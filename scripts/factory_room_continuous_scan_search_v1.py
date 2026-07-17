#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Mission-gated continuous OCR sweep with an explicit parking handoff."""

import json
import threading
import time

import rospy
from std_msgs.msg import String

from factory_room_continuous_handoff_core_v1 import (
    canonical_workshop,
    ocr_matches_target,
    parse_success_task,
)
from fixed_point_continuous_ocr_sweep_route_test_v2 import (
    FixedPointContinuousOcrSweepRouteTestV2,
)


class FactoryRoomContinuousScanSearchV1(
        FixedPointContinuousOcrSweepRouteTestV2):
    WAITING = "WAITING"
    ARMED = "ARMED"
    ACTIVE = "ACTIVE"
    HANDOFF = "HANDOFF"
    HOLD = "HOLD"

    def __init__(self):
        # The navigation base creates a timer before its constructor returns.
        self.integration_ready = False
        self.integration_phase = self.WAITING
        self.integration_lock = threading.RLock()
        self.task_payload = None
        self.target_item = ""
        self.target_warehouse = ""
        self.start_not_before = None
        self.search_completion_reported = False
        self.ocr_candidate_hold_until = None
        self.ocr_candidate_label = ""
        self.ocr_ignore_label = ""
        self.ocr_ignore_until = None
        self.ocr_confirmed_non_targets = set()
        self.launch_wall_time = time.time()
        self.status_pub = None
        super(FactoryRoomContinuousScanSearchV1, self).__init__()

        self.task_result_topic = rospy.get_param(
            "~task_result_topic", "/factory/subtask1_result")
        self.status_topic = rospy.get_param(
            "~search_status_topic", "/factory_room/continuous_scan_status")
        self.control_topic = rospy.get_param(
            "~search_control_topic", "/factory_room/continuous_scan_control")
        self.start_after_tts_s = float(rospy.get_param(
            "~start_after_tts_s", 7.0))
        self.ocr_candidate_hold_s = float(rospy.get_param(
            "~ocr_candidate_hold_s", 1.6))
        self.ocr_non_target_ignore_s = float(rospy.get_param(
            "~ocr_non_target_ignore_s", 1.0))
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True)
        rospy.Subscriber(
            self.task_result_topic, String, self.task_callback, queue_size=5)
        rospy.Subscriber(
            self.control_topic, String, self.control_callback, queue_size=5)

        self.integration_ready = True
        self._set_ocr(False)
        self.publish_integration_status("WAITING_TASK")
        rospy.logwarn(
            "FACTORY_CONTINUOUS_SEARCH_READY task=%s status=%s "
            "motion_before_task=false",
            self.task_result_topic, self.status_topic)

    def publish_integration_status(self, state, **extra):
        if self.status_pub is None:
            return
        payload = {
            "state": state,
            "stamp": time.time(),
            "selected_item": self.target_item,
            "target_warehouse": self.target_warehouse,
        }
        payload.update(extra)
        self.status_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn(
            "FACTORY_CONTINUOUS_SEARCH_STATE %s",
            json.dumps(payload, ensure_ascii=False))

    def task_callback(self, msg):
        try:
            payload = parse_success_task(msg.data)
        except Exception as exc:
            rospy.logwarn_throttle(
                2.0, "continuous search ignored task result: %s", exc)
            return
        stamp = float(payload.get("stamp", time.time()))
        if stamp < self.launch_wall_time - 2.0:
            rospy.logwarn(
                "continuous search ignored stale task result stamp=%.3f", stamp)
            return
        with self.integration_lock:
            if self.integration_phase != self.WAITING:
                return
            self.task_payload = payload
            self.target_item = payload["selected_item"]
            self.target_warehouse = payload["target_warehouse"]
            self.start_not_before = rospy.Time.now() + rospy.Duration(
                max(0.0, self.start_after_tts_s))
            self.integration_phase = self.ARMED
        self.publish_integration_status(
            "SEARCH_ARMED", wait_s=self.start_after_tts_s)

    def control_callback(self, msg):
        command = str(msg.data).strip().lower()
        if command not in ("stop", "abort", "hold"):
            return
        with self.integration_lock:
            if self.integration_phase in (self.HANDOFF, self.HOLD):
                return
            self.integration_phase = self.HOLD
            self.scan_active = False
            self.finished = True
            self.path_world = []
        self._set_scan_safety_mode(False)
        self._set_ocr(False)
        self.publish_zero("FACTORY_SEARCH_EXTERNAL_STOP")
        self.publish_integration_status(
            "SEARCH_ABORTED", reason="external stop request")

    def _start_search(self):
        with self.integration_lock:
            if self.integration_phase != self.ARMED:
                return
            self.route_index = 0
            self.finished = False
            self.scan_active = False
            self.scanned_route_indexes.clear()
            self.scan_candidate_attempts.clear()
            self.search_completion_reported = False
            self.ocr_candidate_hold_until = None
            self.ocr_candidate_label = ""
            self.ocr_ignore_label = ""
            self.ocr_ignore_until = None
            self.ocr_confirmed_non_targets.clear()
            self._activate_route_point(clear_path=True)
            self.scan_route_started = rospy.Time.now()
            self.last_progress_time = rospy.Time.now()
            self.route_transition_hold_until = None
            self.candidate_waiting = False
            self.candidate_retry_not_before = None
            self._reset_translation_progress()
            self.integration_phase = self.ACTIVE
        self._set_ocr(False)
        self.publish_zero("FACTORY_CONTINUOUS_SEARCH_START")
        self.publish_integration_status(
            "SEARCH_STARTED",
            route=[point["name"] for point in self.route_points])

    def _handoff_to_parking(self, payload):
        with self.integration_lock:
            if self.integration_phase != self.ACTIVE:
                return
            point = self.current_route_point()["name"]
            yaw = None if self.pose is None else self.pose[2]
            self.integration_phase = self.HANDOFF
            self.scan_active = False
            self.finished = True
            self.path_world = []
            self.path_index = 0
        self._set_scan_safety_mode(False)
        # Publish the stop before announcing handoff. The parking manager cannot
        # start until this callback has fully relinquished the command stream.
        for _ in range(6):
            self.publish_zero("TARGET_WORKSHOP_HANDOFF_STOP")
            rospy.sleep(0.02)
        self.publish_integration_status(
            "TARGET_FOUND", point=point, yaw=yaw,
            label=payload.get("label"), bbox=payload.get("bbox"),
            score=float(payload.get("score", 0.0)))
        rospy.logwarn(
            "FACTORY_CONTINUOUS_TARGET_HANDOFF point=%s target=%s "
            "navigator_silent=true",
            point, self.target_warehouse)

    def ocr_callback(self, msg):
        super(FactoryRoomContinuousScanSearchV1, self).ocr_callback(msg)
        with self.integration_lock:
            active = (self.integration_phase == self.ACTIVE and
                      self.scan_active)
            target = self.target_warehouse
        if not active:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if ocr_matches_target(payload, target):
            self._handoff_to_parking(payload)
            return

        stable_label = (canonical_workshop(payload.get("label", ""))
                        if payload.get("stable") else "")
        if stable_label and stable_label != target:
            now = rospy.Time.now()
            with self.integration_lock:
                if stable_label in self.ocr_confirmed_non_targets:
                    return
                self.ocr_confirmed_non_targets.add(stable_label)
                self.ocr_candidate_hold_until = None
                self.ocr_candidate_label = ""
                self.ocr_ignore_label = stable_label
                self.ocr_ignore_until = (
                    now + rospy.Duration(self.ocr_non_target_ignore_s))
            # Do not let a confirmed non-target sign remain in the vote window
            # when the sweep reaches the next wall sign.
            self.ocr_control_pub.publish(String(data="reset"))
            rospy.logwarn(
                "OCR_FACTORY_NON_TARGET_CONFIRMED label=%s target=%s; "
                "votes_cleared sweep_resumes=true",
                stable_label, target)
            return

        frame_label = canonical_workshop(payload.get("frame_label", ""))
        bbox = payload.get("bbox")
        if (frame_label and isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            now = rospy.Time.now()
            with self.integration_lock:
                if self.integration_phase != self.ACTIVE or not self.scan_active:
                    return
                if frame_label in self.ocr_confirmed_non_targets:
                    return
                if (self.ocr_ignore_label == frame_label and
                        self.ocr_ignore_until is not None and
                        now < self.ocr_ignore_until):
                    return
                if self.ocr_candidate_hold_until is not None:
                    return
                self.ocr_candidate_label = frame_label
                self.ocr_candidate_hold_until = (
                    now + rospy.Duration(
                        max(0.4, self.ocr_candidate_hold_s)))
            # Start this sign with an uncontaminated vote window. The triggering
            # frame is intentionally discarded; the stationary frames decide.
            self.ocr_control_pub.publish(String(data="reset"))
            rospy.logwarn(
                "OCR_FACTORY_CANDIDATE_HOLD label=%s target=%s "
                "old_votes=%s/%s hold=%.2fs bbox=%s",
                frame_label, target, payload.get("votes", 0),
                payload.get("vote_window", 0), self.ocr_candidate_hold_s,
                bbox)

    def _scan_control_loop(self):
        with self.integration_lock:
            hold_until = self.ocr_candidate_hold_until
            candidate = self.ocr_candidate_label
        now = rospy.Time.now()
        if hold_until is not None and now < hold_until:
            # Keep the target sign fixed in view long enough for several fresh
            # OCR frames. Reset sweep timing so this intentional pause cannot
            # be mistaken for a blocked rectangular rotation.
            self.scan_command_wz = 0.0
            self.scan_last_tick = now
            self.scan_last_motion = now
            if self.pose is not None:
                self.scan_last_yaw = self.pose[2]
            self.publish_cmd(0.0, 0.0)
            rospy.logwarn_throttle(
                0.35,
                "OCR_FACTORY_CANDIDATE_ACCUMULATING label=%s remaining=%.2fs",
                candidate, (hold_until - now).to_sec())
            return
        if hold_until is not None:
            with self.integration_lock:
                self.ocr_candidate_hold_until = None
                self.ocr_candidate_label = ""
                self.ocr_ignore_label = candidate
                self.ocr_ignore_until = (
                    now + rospy.Duration(self.ocr_non_target_ignore_s))
            self.ocr_control_pub.publish(String(data="reset"))
            rospy.logwarn(
                "OCR_FACTORY_CANDIDATE_TIMEOUT label=%s stable=false; "
                "votes_cleared sweep_resumes=true",
                candidate)
        super(FactoryRoomContinuousScanSearchV1, self)._scan_control_loop()

    def _finish_scan(self):
        final_scan = self.route_index >= len(self.route_points) - 1
        final_name = self.current_route_point()["name"]
        super(FactoryRoomContinuousScanSearchV1, self)._finish_scan()
        with self.integration_lock:
            self.ocr_candidate_hold_until = None
            self.ocr_candidate_label = ""
            self.ocr_ignore_label = ""
            self.ocr_ignore_until = None
        with self.integration_lock:
            if (not final_scan or self.integration_phase != self.ACTIVE or
                    self.search_completion_reported):
                return
            self.search_completion_reported = True
            self.integration_phase = self.HOLD
        self._set_scan_safety_mode(False)
        self._set_ocr(False)
        self.publish_zero("FACTORY_SEARCH_ALL_POINTS_COMPLETE")
        self.publish_integration_status(
            "SEARCH_COMPLETE", last_point=final_name,
            scanned_points=sorted(self.scan_specs))

    def control_loop(self, event):
        if not self.integration_ready:
            return
        with self.integration_lock:
            phase = self.integration_phase
            start_not_before = self.start_not_before
        if phase == self.WAITING or phase == self.HANDOFF:
            return
        if phase == self.ARMED:
            if (start_not_before is None or
                    rospy.Time.now() < start_not_before):
                return
            self._start_search()
            return
        if phase == self.HOLD:
            self.publish_zero("FACTORY_CONTINUOUS_SEARCH_HOLD")
            return
        super(FactoryRoomContinuousScanSearchV1, self).control_loop(event)


if __name__ == "__main__":
    FactoryRoomContinuousScanSearchV1().spin()
