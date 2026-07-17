#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parking manager that resumes search after a verified false handoff."""

import json
import threading
import time

import rospy
from std_msgs.msg import String

from factory_room_continuous_handoff_core_v1 import canonical_workshop
from factory_room_delivery_manager_continuous_v7 import (
    DirectCenterlineParkingManager,
)
from factory_room_false_handoff_core_v8 import parking_recheck_result


class RecoverableHandoffParkingManager(DirectCenterlineParkingManager):
    RESUME_COMMAND = "resume_after_false_handoff"

    def __init__(self):
        self.search_resumed = threading.Event()
        self.false_handoff_max_resumes = 2
        self.false_handoff_ack_timeout = 2.5
        self.false_handoff_search_timeout = 75.0
        self.false_handoff_recheck_age = 1.8
        self.false_handoff_min_votes = 8
        self.target_parking_max_retries = 1
        self.target_parking_retry_delay = 0.4
        super(RecoverableHandoffParkingManager, self).__init__()
        self.false_handoff_max_resumes = int(rospy.get_param(
            "~parking_false_handoff_max_resumes", 2))
        self.false_handoff_ack_timeout = float(rospy.get_param(
            "~parking_false_handoff_ack_timeout_s", 2.5))
        self.false_handoff_search_timeout = float(rospy.get_param(
            "~parking_false_handoff_search_timeout_s", 75.0))
        self.false_handoff_recheck_age = float(rospy.get_param(
            "~parking_false_handoff_recheck_age_s", 1.8))
        self.false_handoff_min_votes = int(rospy.get_param(
            "~parking_false_handoff_min_votes", 8))
        self.target_parking_max_retries = int(rospy.get_param(
            "~parking_target_max_retries", 1))
        self.target_parking_retry_delay = float(rospy.get_param(
            "~parking_target_retry_delay_s", 0.4))
        rospy.logwarn(
            "ROOM_FALSE_HANDOFF_RECOVERY_V8_READY max_resumes=%d "
            "min_votes=%d audible_during_resume=false",
            self.false_handoff_max_resumes,
            self.false_handoff_min_votes)

    def search_status_callback(self, msg):
        super(RecoverableHandoffParkingManager,
              self).search_status_callback(msg)
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if str(payload.get("state", "")) == \
                "SEARCH_RESUMED_AFTER_FALSE_HANDOFF":
            self.search_resumed.set()
        if str(payload.get("state", "")) == "TARGET_FOUND":
            self.refresh_room_watchdog(
                "verified_target_parking", self.room_parking_timeout)

    def _fresh_parking_recheck(self, attempt_started):
        with self.lock:
            payload = (None if self.latest_ocr is None
                       else dict(self.latest_ocr))
        if payload is not None:
            payload["label"] = canonical_workshop(
                payload.get("label", ""))
        target = canonical_workshop(self.target_warehouse)
        return parking_recheck_result(
            payload, target, time.time(), attempt_started,
            self.false_handoff_recheck_age,
            self.false_handoff_min_votes)

    def _watchdog_aborted(self):
        lock = getattr(self, "_room_watchdog_lock", None)
        if lock is None:
            return False
        with lock:
            return bool(getattr(self, "_room_watchdog_aborted", False))

    def _resume_search_and_wait(self, observed_label, resume_index):
        self.refresh_room_watchdog(
            "false_handoff_search",
            self.false_handoff_search_timeout + 10.0)
        self.search_target_found.clear()
        self.search_complete.clear()
        self.search_aborted.clear()
        self.search_resumed.clear()
        command = {
            "command": self.RESUME_COMMAND,
            "observed_label": observed_label,
            "resume_index": resume_index,
            "stamp": time.time(),
        }
        self.search_control_pub.publish(String(
            data=json.dumps(command, ensure_ascii=False)))
        self.publish_state(
            "FALSE_HANDOFF_RESUME_REQUESTED",
            observed_label=observed_label, resume_index=resume_index)

        ack_deadline = time.time() + self.false_handoff_ack_timeout
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and time.time() < ack_deadline and
               not self.search_resumed.is_set() and
               not self.search_complete.is_set()):
            if self._watchdog_aborted():
                return False
            self.publish_zero(1)
            rate.sleep()
        if (not self.search_resumed.is_set() and
                not self.search_complete.is_set()):
            rospy.logerr("FALSE_HANDOFF_RESUME_ACK_TIMEOUT")
            return False

        deadline = time.time() + self.false_handoff_search_timeout
        self.publish_state(
            "FALSE_HANDOFF_SEARCH_CONTINUING",
            observed_label=observed_label)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self._watchdog_aborted():
                return False
            if self.search_target_found.is_set():
                self.publish_zero(5)
                self.publish_state(
                    "FALSE_HANDOFF_NEW_TARGET_FOUND",
                    scan_status=self.search_status_payload)
                return True
            if (self.search_complete.is_set() or
                    self.search_aborted.is_set()):
                return False
            rate.sleep()
        rospy.logerr("FALSE_HANDOFF_RESUMED_SEARCH_TIMEOUT")
        return False

    def park_inside_square(self):
        resumes = 0
        target_retries = 0
        while not rospy.is_shutdown():
            if not self.refresh_room_watchdog(
                    "centerline_parking", self.room_parking_timeout):
                return False
            attempt_started = time.time()
            if super(RecoverableHandoffParkingManager,
                     self).park_inside_square():
                return True

            verdict, observed = self._fresh_parking_recheck(attempt_started)
            rospy.logwarn(
                "PARKING_RECHECK_V8 verdict=%s observed=%s target=%s",
                verdict, observed, self.target_warehouse)
            if (verdict == "target" and
                    target_retries < max(0, self.target_parking_max_retries)):
                target_retries += 1
                self.publish_zero(12)
                self.publish_state(
                    "TARGET_PARKING_RETRY",
                    attempt=target_retries,
                    max_retries=self.target_parking_max_retries)
                rospy.logwarn(
                    "TARGET_PARKING_RETRY attempt=%d/%d reason=verified_target",
                    target_retries, self.target_parking_max_retries)
                rospy.sleep(max(0.0, self.target_parking_retry_delay))
                if not self.wait_for_inputs(timeout_s=5.0):
                    return False
                continue
            if verdict != "non_target":
                return False
            if resumes >= max(0, self.false_handoff_max_resumes):
                rospy.logerr(
                    "FALSE_HANDOFF_RESUME_LIMIT observed=%s", observed)
                return False

            resumes += 1
            self.ocr_control("disable")
            self.publish_zero(12)
            if not self._resume_search_and_wait(observed, resumes):
                return False
            if not self.wait_for_inputs(timeout_s=8.0):
                return False
            if not self.approach_target_wall_with_navigation():
                return False
        return False


if __name__ == "__main__":
    RecoverableHandoffParkingManager().run()
