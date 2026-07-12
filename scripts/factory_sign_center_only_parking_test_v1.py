#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone parking driven only by the OCR workshop-sign centerline."""

import math
import time

import rospy
from std_msgs.msg import Bool

from factory_sign_center_parking_test_v1 import (
    SignCenterParkingTest,
    clamp,
)


class SignCenterOnlyParkingTest(SignCenterParkingTest):
    def __init__(self):
        # Scan callbacks can arrive while the base constructor is returning.
        self.robot_half_length = 0.171
        self.robot_half_width = 0.128
        self.laser_offset_x = -0.11
        self.swept_front_margin = 0.10
        self.swept_rear_margin = 0.08
        self.left_swept_clear = float("inf")
        self.right_swept_clear = float("inf")
        super(SignCenterOnlyParkingTest, self).__init__()

        self.robot_half_length = float(rospy.get_param(
            "~robot_half_length_m", 0.171))
        self.robot_half_width = float(rospy.get_param(
            "~robot_half_width_m", 0.128))
        self.laser_offset_x = float(rospy.get_param(
            "~laser_offset_x_m", -0.11))
        self.swept_front_margin = float(rospy.get_param(
            "~swept_front_margin_m", 0.10))
        self.swept_rear_margin = float(rospy.get_param(
            "~swept_rear_margin_m", 0.08))
        self.lateral_hard_clearance = float(rospy.get_param(
            "~lateral_hard_clearance_m", 0.10))
        self.lateral_slow_clearance = float(rospy.get_param(
            "~lateral_slow_clearance_m", 0.34))
        self.obstacle_wait_timeout_s = float(rospy.get_param(
            "~obstacle_wait_timeout_s", 3.0))
        self.near_center_band_px = float(rospy.get_param(
            "~near_center_band_px", 48.0))
        self.near_center_min_speed = float(rospy.get_param(
            "~near_center_min_speed", 0.014))
        self.lateral_realign_deg = float(rospy.get_param(
            "~lateral_realign_deg", 7.0))
        self.heading_min_wz = float(rospy.get_param(
            "~heading_min_wz", 0.115))
        self.heading_stuck_wz = float(rospy.get_param(
            "~heading_stuck_wz", 0.155))
        self.heading_progress_deg = float(rospy.get_param(
            "~heading_progress_deg", 0.35))
        self.heading_stuck_after_s = float(rospy.get_param(
            "~heading_stuck_after_s", 1.6))

    def scan_callback(self, msg):
        super(SignCenterOnlyParkingTest, self).scan_callback(msg)
        left_clear = float("inf")
        right_clear = float("inf")
        min_x = -self.robot_half_length - self.swept_rear_margin
        max_x = self.robot_half_length + self.swept_front_margin
        for index, distance in enumerate(msg.ranges):
            if (not math.isfinite(distance) or distance < msg.range_min or
                    distance > msg.range_max):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            x_base = distance * math.cos(angle) + self.laser_offset_x
            y_base = distance * math.sin(angle)
            if x_base < min_x or x_base > max_x:
                continue
            if y_base > self.robot_half_width:
                left_clear = min(left_clear, y_base - self.robot_half_width)
            elif y_base < -self.robot_half_width:
                right_clear = min(right_clear, -y_base - self.robot_half_width)
        with self.lock:
            self.left_swept_clear = left_clear
            self.right_swept_clear = right_clear

    def snapshot(self):
        result = super(SignCenterOnlyParkingTest, self).snapshot()
        with self.lock:
            result["left_swept_clear"] = self.left_swept_clear
            result["right_swept_clear"] = self.right_swept_clear
        return result

    def guarded_lateral_speed(self, requested, snapshot):
        if abs(requested) < 1.0e-5:
            return 0.0, float("inf"), "clear"
        clearance = (snapshot["left_swept_clear"] if requested > 0.0
                     else snapshot["right_swept_clear"])
        if clearance <= self.lateral_hard_clearance:
            return 0.0, clearance, "blocked"
        if clearance >= self.lateral_slow_clearance or not math.isfinite(clearance):
            return requested, clearance, "clear"
        ratio = clamp(
            (clearance - self.lateral_hard_clearance) /
            max(0.01, self.lateral_slow_clearance - self.lateral_hard_clearance),
            0.22, 1.0)
        return requested * ratio, clearance, "slow"

    def align_wall_heading(self, state_name):
        self.publish_state(state_name, controller="centerline_only")
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
                "CENTER_ONLY_HEADING wall=%.3f error=%.2fdeg stable=%d/%d "
                "cmd_w=%.3f",
                snapshot["wall"]["distance"], math.degrees(error), stable,
                self.heading_stable_frames, command_wz)
            if stable >= self.heading_stable_frames:
                self.publish_zero(12)
                rospy.logwarn("CENTER_ONLY_HEADING_OK error=%.2fdeg",
                              math.degrees(error))
                return True
            rate.sleep()
        self.publish_zero(30)
        rospy.logerr("CENTER_ONLY_HEADING_FAILED")
        return False

    def center_under_sign(self):
        self.publish_state("CENTER_ONLY_LATERAL_ALIGNMENT")
        start_xy = self.snapshot()["odom_xy"]
        stable = 0
        obstacle_since = None
        start = time.time()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() - start < self.lateral_timeout_s:
            snapshot = self.snapshot()
            if not self.sensors_fresh(snapshot, require_ocr=True):
                self.publish_zero()
                stable = 0
                rospy.logwarn_throttle(1.0, "CENTER_ONLY paused: stale OCR/lidar/odom")
                rate.sleep()
                continue
            moved = self.odom_displacement(start_xy, snapshot["odom_xy"])
            if moved > self.lateral_limit_m:
                rospy.logerr("CENTER_ONLY lateral travel limit %.3fm", moved)
                self.publish_zero(30)
                return False
            heading_error = snapshot["wall"]["heading_error"]
            if abs(heading_error) > math.radians(self.lateral_realign_deg):
                self.publish_zero(8)
                if not self.align_wall_heading("CENTER_ONLY_RECOVER_HEADING"):
                    return False
                stable = 0
                continue

            ocr = snapshot["ocr"]
            width = float(ocr["image_width"])
            center = float(ocr["filtered_center_x"])
            target = width * self.optical_center_x_ratio
            pixel_error = center - target
            if abs(pixel_error) <= self.sign_center_tolerance_px:
                stable += 1
                obstacle_since = None
                cmd_y = 0.0
                clearance = float("inf")
                guard = "centered"
                self.publish_command()
            else:
                stable = 0
                normalized = pixel_error / max(1.0, 0.5 * width)
                requested = self.lateral_command_sign * self.lateral_kp * normalized
                minimum = (self.near_center_min_speed
                           if abs(pixel_error) <= self.near_center_band_px
                           else self.lateral_min_speed)
                if abs(requested) < minimum:
                    requested = math.copysign(minimum, requested)
                requested = clamp(requested, -self.lateral_max_speed,
                                  self.lateral_max_speed)
                cmd_y, clearance, guard = self.guarded_lateral_speed(
                    requested, snapshot)
                if guard == "blocked":
                    self.publish_zero()
                    if obstacle_since is None:
                        obstacle_since = time.time()
                    rospy.logerr_throttle(
                        0.4,
                        "CENTER_ONLY_LATERAL_BLOCKED clear=%.3fm direction=%s",
                        clearance, "left" if requested > 0.0 else "right")
                    if time.time() - obstacle_since >= self.obstacle_wait_timeout_s:
                        self.publish_zero(30)
                        return False
                else:
                    obstacle_since = None
                    self.publish_command(y=cmd_y)
            rospy.logwarn_throttle(
                0.30,
                "CENTER_ONLY_ALIGN label=%s u=%.1f target=%.1f error=%.1fpx "
                "stable=%d/%d clear=%.3f guard=%s cmd_y=%.3f",
                ocr.get("label"), center, target, pixel_error, stable,
                self.sign_stable_frames, clearance, guard, cmd_y)
            if stable >= self.sign_stable_frames:
                self.publish_zero(15)
                self.publish_state(
                    "CENTER_ONLY_ALIGNED", label=ocr.get("label"),
                    pixel_error=pixel_error, lateral_motion=moved)
                return True
            rate.sleep()
        self.publish_zero(30)
        return False


if __name__ == "__main__":
    node = SignCenterOnlyParkingTest()
    try:
        rospy.sleep(1.0)
        node.success = node.run()
        node.finished = True
        node.publish_zero(30)
        if not node.success:
            rospy.logerr("CENTER_ONLY_TEST_FAILED; vehicle remains stopped")
        rospy.logwarn("CENTER_ONLY_TEST_FINISHED; node holds zero")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            node.publish_zero()
            node.parking_pub.publish(Bool(data=True))
            rospy.logwarn_throttle(
                3.0, "CENTER_ONLY finished=%s success=%s HOLDING_ZERO",
                str(node.finished), str(node.success))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

