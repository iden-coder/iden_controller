#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V3: effective wall-heading recovery for the sign/frame assisted parker."""

import math
import time

import rospy
from std_msgs.msg import Bool

from factory_sign_center_parking_test_v1 import clamp
from factory_sign_center_parking_test_v2 import AssistedSignParkingTest


class EffectiveHeadingParkingTest(AssistedSignParkingTest):
    def __init__(self):
        super(EffectiveHeadingParkingTest, self).__init__()
        self.heading_min_wz = float(rospy.get_param(
            "~heading_min_wz", 0.125))
        self.heading_stuck_wz = float(rospy.get_param(
            "~heading_stuck_wz", 0.165))
        self.heading_progress_deg = float(rospy.get_param(
            "~heading_progress_deg", 0.35))
        self.heading_stuck_after_s = float(rospy.get_param(
            "~heading_stuck_after_s", 1.6))
        self.lateral_realign_deg = float(rospy.get_param(
            "~lateral_realign_deg", 5.5))
        self.lateral_recovery_tolerance_deg = float(rospy.get_param(
            "~lateral_recovery_tolerance_deg", 2.6))

    def align_wall_heading(self, state_name):
        self.publish_state(state_name, controller="effective_heading_v3")
        stable = 0
        start = time.time()
        best_error = float("inf")
        last_progress = start
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < self.heading_timeout_s:
            snapshot = self.snapshot()
            if not self.sensors_fresh(snapshot, require_ocr=False):
                self.publish_zero()
                stable = 0
                rate.sleep()
                continue
            error = snapshot["wall"]["heading_error"]
            abs_error = abs(error)
            if abs_error + math.radians(self.heading_progress_deg) < best_error:
                best_error = abs_error
                last_progress = time.time()
            if abs_error <= self.heading_tolerance:
                stable += 1
                command_wz = 0.0
                self.publish_command()
            else:
                stable = 0
                minimum = self.heading_min_wz
                if time.time() - last_progress >= self.heading_stuck_after_s:
                    minimum = self.heading_stuck_wz
                command_wz = self.heading_kp * error
                if abs(command_wz) < minimum:
                    command_wz = math.copysign(minimum, error)
                command_wz = clamp(
                    command_wz, -self.heading_max_wz, self.heading_max_wz)
                self.publish_command(wz=command_wz)
            rospy.logwarn_throttle(
                0.30,
                "SIGN_PARK_V3_HEADING wall=%.3f error=%.2fdeg stable=%d/%d "
                "cmd_w=%.3f no_progress=%.1fs",
                snapshot["wall"]["distance"], math.degrees(error), stable,
                self.heading_stable_frames, command_wz,
                time.time() - last_progress)
            if stable >= self.heading_stable_frames:
                self.publish_zero(12)
                rospy.logwarn(
                    "SIGN_PARK_V3_HEADING_OK error=%.2fdeg elapsed=%.2fs",
                    math.degrees(error), time.time() - start)
                return True
            rate.sleep()
        self.publish_zero(30)
        rospy.logerr(
            "SIGN_PARK_V3_HEADING_FAILED best_error=%.2fdeg timeout=%.1fs",
            math.degrees(best_error), self.heading_timeout_s)
        return False

    def center_under_sign(self):
        # Lateral mecanum motion can perturb the lidar wall estimate by several
        # degrees. Do not interrupt useful centering for a small transient.
        strict_tolerance = self.heading_tolerance
        recovery_tolerance = math.radians(
            self.lateral_recovery_tolerance_deg)
        # V2 recovers at 2*tolerance, so this also encodes the V3 threshold.
        recovery_tolerance = max(
            recovery_tolerance, 0.5 * math.radians(self.lateral_realign_deg))
        self.heading_tolerance = recovery_tolerance
        try:
            return super(EffectiveHeadingParkingTest, self).center_under_sign()
        finally:
            # The final heading alignment and wall approach remain strict.
            self.heading_tolerance = strict_tolerance


if __name__ == "__main__":
    node = EffectiveHeadingParkingTest()
    try:
        rospy.sleep(1.0)
        node.success = node.run()
        node.finished = True
        node.publish_zero(30)
        if not node.success:
            rospy.logerr("SIGN_PARK_V3_TEST_FAILED; vehicle remains stopped")
        rospy.logwarn("SIGN_PARK_V3_TEST_FINISHED; node holds zero")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            node.publish_zero()
            node.parking_pub.publish(Bool(data=True))
            rospy.logwarn_throttle(
                3.0, "SIGN_PARK_V3 finished=%s success=%s HOLDING_ZERO",
                str(node.finished), str(node.success))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

