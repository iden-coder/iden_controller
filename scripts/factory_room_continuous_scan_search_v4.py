#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Continuous search that actively recenters edge-visible target signs."""

import json

import rospy

from factory_room_continuous_handoff_core_v1 import canonical_workshop
from factory_room_continuous_scan_search_v3 import (
    CandidateGatedContinuousScanSearchV3,
)
from factory_room_edge_recenter_core_v10 import (
    edge_target_recenter_decision,
)
from fixed_point_continuous_ocr_sweep_route_test_v2 import (
    FixedPointContinuousOcrSweepRouteTestV2,
)


class EdgeRecenteringContinuousScanSearchV4(
        CandidateGatedContinuousScanSearchV3):
    """Move a confirmed target away from the frame edge before handoff."""

    def __init__(self):
        # The inherited navigation timer can run before construction returns.
        self.edge_recenter_state = ""
        self.edge_recenter_command = 0.0
        self.edge_recenter_last_seen = None
        self.edge_recenter_fresh_s = 0.65
        self.edge_recenter_gain = 0.28
        self.edge_recenter_min_speed = 0.08
        self.edge_recenter_max_speed = 0.20
        super(EdgeRecenteringContinuousScanSearchV4, self).__init__()

        self.edge_recenter_fresh_s = float(rospy.get_param(
            "~edge_target_recenter_fresh_s", 0.65))
        self.edge_recenter_gain = float(rospy.get_param(
            "~edge_target_recenter_gain", 0.28))
        self.edge_recenter_min_speed = float(rospy.get_param(
            "~edge_target_recenter_min_rps", 0.08))
        self.edge_recenter_max_speed = float(rospy.get_param(
            "~edge_target_recenter_max_rps", 0.20))
        rospy.logwarn(
            "FACTORY_CONTINUOUS_SEARCH_V4_READY edge_target_recenter=true "
            "speed=(%.2f,%.2f)rps fresh=%.2fs",
            self.edge_recenter_min_speed,
            self.edge_recenter_max_speed,
            self.edge_recenter_fresh_s)

    def _clear_edge_recenter(self):
        self.edge_recenter_state = ""
        self.edge_recenter_command = 0.0
        self.edge_recenter_last_seen = None

    def _cancel_target_hold(self, target):
        with self.integration_lock:
            if self.ocr_candidate_label == target:
                self.ocr_candidate_hold_until = None
                self.ocr_candidate_label = ""

    def ocr_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return super(EdgeRecenteringContinuousScanSearchV4,
                         self).ocr_callback(msg)

        with self.integration_lock:
            active = (self.integration_phase == self.ACTIVE and
                      self.scan_active)
            target = self.target_warehouse
        frame_label = canonical_workshop(payload.get("frame_label", ""))
        stable_label = (canonical_workshop(payload.get("label", ""))
                        if payload.get("stable") else "")
        target_visible = bool(
            active and target and target in (frame_label, stable_label))

        if target_visible:
            state, command = edge_target_recenter_decision(
                payload.get("bbox"), payload.get("image_width", 0),
                self.handoff_edge_margin_px,
                self.handoff_min_bbox_width_px,
                self.edge_recenter_gain,
                self.edge_recenter_min_speed,
                self.edge_recenter_max_speed)
            if state in ("recenter", "partial"):
                # Preserve OCR diagnostics without invoking V1's stationary
                # candidate hold or V2's edge handoff rejection path.
                FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(
                    self, msg)
                self._cancel_target_hold(target)
                if state == "recenter":
                    self.edge_recenter_state = state
                    self.edge_recenter_command = command
                    self.edge_recenter_last_seen = rospy.Time.now()
                    rospy.logwarn_throttle(
                        0.35,
                        "OCR_TARGET_EDGE_RECENTER label=%s bbox=%s "
                        "cmd_wz=%.3f parking_handoff=false",
                        target, payload.get("bbox"), command)
                else:
                    self._clear_edge_recenter()
                    rospy.logwarn_throttle(
                        0.5,
                        "OCR_TARGET_PARTIAL_CONTINUE label=%s bbox=%s",
                        target, payload.get("bbox"))
                return
            if state == "usable":
                self._clear_edge_recenter()

        return super(EdgeRecenteringContinuousScanSearchV4,
                     self).ocr_callback(msg)

    def _scan_control_loop(self):
        now = rospy.Time.now()
        if (self.edge_recenter_state == "recenter" and
                self.edge_recenter_last_seen is not None and
                (now - self.edge_recenter_last_seen).to_sec() <=
                self.edge_recenter_fresh_s):
            # Recenter is intentional scan motion. Keep the sweep's stall and
            # progress clocks coherent so it cannot look like a blocked turn.
            self.scan_command_wz = self.edge_recenter_command
            self.scan_last_tick = now
            self.scan_last_motion = now
            if self.pose is not None:
                self.scan_last_yaw = self.pose[2]
            self.publish_cmd(0.0, self.edge_recenter_command)
            rospy.logwarn_throttle(
                0.35,
                "OCR_TARGET_RECENTERING cmd_wz=%.3f ocr=enabled",
                self.edge_recenter_command)
            return
        if self.edge_recenter_state:
            rospy.logwarn(
                "OCR_TARGET_RECENTER_STALE; normal_sweep_resumes=true")
            self._clear_edge_recenter()
        super(EdgeRecenteringContinuousScanSearchV4,
              self)._scan_control_loop()

    def _start_search(self):
        self._clear_edge_recenter()
        return super(EdgeRecenteringContinuousScanSearchV4,
                     self)._start_search()

    def _handoff_to_parking(self, payload):
        self._clear_edge_recenter()
        return super(EdgeRecenteringContinuousScanSearchV4,
                     self)._handoff_to_parking(payload)

    def _finish_scan(self):
        self._clear_edge_recenter()
        return super(EdgeRecenteringContinuousScanSearchV4,
                     self)._finish_scan()

    def shutdown(self):
        self._clear_edge_recenter()
        return super(EdgeRecenteringContinuousScanSearchV4, self).shutdown()


if __name__ == "__main__":
    EdgeRecenteringContinuousScanSearchV4().spin()

