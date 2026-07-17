#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fast centerline parking with clearance-producing straight retreat."""

import math
import time

import rospy

from factory_room_delivery_manager_center_only_v1 import clamp
from factory_room_delivery_manager_center_only_v2 import (
    RobustCenterlineFactoryDeliveryManager,
)


class FastClearanceCenterlineManager(
        RobustCenterlineFactoryDeliveryManager):
    def __init__(self):
        self.rear_swept_clear = float("inf")
        super(FastClearanceCenterlineManager, self).__init__()
        self.precenter_clearance = float(rospy.get_param(
            "~center_precenter_clearance_m", 0.22))
        self.precenter_retreat_distance = float(rospy.get_param(
            "~center_precenter_retreat_distance_m", 0.16))
        self.precenter_retreat_speed = float(rospy.get_param(
            "~center_precenter_retreat_speed_mps", 0.070))
        self.precenter_retreat_timeout = float(rospy.get_param(
            "~center_precenter_retreat_timeout_s", 4.5))
        self.precenter_rear_clear = float(rospy.get_param(
            "~center_precenter_rear_clearance_m", 0.11))
        self.quick_handoff_settle = float(rospy.get_param(
            "~center_quick_handoff_settle_s", 0.25))
        rospy.logwarn(
            "CENTERLINE_MANAGER_V3 fast=true preclear=%.2fm retreat=%.2fm",
            self.precenter_clearance, self.precenter_retreat_distance)

    def scan_callback(self, msg):
        super(FastClearanceCenterlineManager, self).scan_callback(msg)
        if not self.scan_tf_ready:
            return
        tx, ty, tf_yaw, _ = self._scan_transform(msg)
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        rear_clear = float("inf")
        corridor = self.robot_half_width + 0.035
        for index, distance in enumerate(msg.ranges):
            if (not math.isfinite(distance) or distance < msg.range_min or
                    distance > msg.range_max):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)
            bx = tx + cos_yaw * lx - sin_yaw * ly
            by = ty + sin_yaw * lx + cos_yaw * ly
            if abs(by) <= corridor and bx < -self.robot_half_length:
                rear_clear = min(
                    rear_clear, -bx - self.robot_half_length)
        with self.lock:
            self.rear_swept_clear = rear_clear

    def snapshot_center(self):
        snapshot = super(FastClearanceCenterlineManager, self).snapshot_center()
        with self.lock:
            snapshot["rear_clear"] = self.rear_swept_clear
        return snapshot

    def approach_target_wall_with_navigation(self):
        # The workshop sign is already visible. Wall fitting is faster and
        # more repeatable than another image-only heading servo here.
        self.move_client.cancel_all_goals()
        self.publish_zero(4)
        self.publish_state(
            "CENTERLINE_FAST_PARKING_HANDOFF",
            warehouse=self.target_warehouse)
        self.ocr_control("reset")
        self.ocr_control("enable")
        rospy.sleep(max(0.1, self.quick_handoff_settle))
        return True

    def _needed_lateral_direction(self):
        sign = self.target_sign_error()
        if sign is None or abs(sign["error_px"]) <= self.sign_tolerance_px:
            return 0.0, sign
        normalized = sign["error_px"] / max(1.0, 0.5 * sign["width"])
        requested = self.lateral_command_sign * self.lateral_kp * normalized
        return math.copysign(1.0, requested), sign

    def _directional_clearance(self, snapshot, direction):
        return (snapshot["left_clear"] if direction > 0.0
                else snapshot["right_clear"])

    def retreat_for_lateral_clearance(self):
        direction, sign = self._needed_lateral_direction()
        if direction == 0.0:
            return True
        snapshot = self.snapshot_center()
        clearance = self._directional_clearance(snapshot, direction)
        if clearance >= self.precenter_clearance:
            return True

        self.publish_state(
            "CENTERLINE_CLEARANCE_RETREAT",
            side="left" if direction > 0.0 else "right",
            clearance=clearance)
        start_xy = snapshot["odom_xy"]
        progress_xy = start_xy
        last_progress = time.time()
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.precenter_retreat_timeout):
            snapshot = self.snapshot_center()
            clearance = self._directional_clearance(snapshot, direction)
            travelled = self.distance_between(start_xy, snapshot["odom_xy"])
            if clearance >= self.precenter_clearance:
                self.publish_zero(8)
                rospy.logwarn(
                    "CENTERLINE_RETREAT_CLEAR side_clear=%.3f travel=%.3f",
                    clearance, travelled)
                return True
            if travelled >= self.precenter_retreat_distance:
                self.publish_zero(8)
                rospy.logwarn(
                    "CENTERLINE_RETREAT_DISTANCE side_clear=%.3f travel=%.3f",
                    clearance, travelled)
                return clearance > self.lateral_hard_clearance
            if snapshot["rear_clear"] < self.precenter_rear_clear:
                self.publish_zero(12)
                rospy.logerr(
                    "CENTERLINE_RETREAT_REAR_BLOCKED rear=%.3f",
                    snapshot["rear_clear"])
                return False
            if self.distance_between(
                    progress_xy, snapshot["odom_xy"]) >= 0.012:
                progress_xy = snapshot["odom_xy"]
                last_progress = time.time()
            elif time.time() - last_progress > 1.4:
                self.publish_zero(12)
                rospy.logerr("CENTERLINE_RETREAT_NO_PROGRESS")
                return False
            self.publish_center_command(x=-self.precenter_retreat_speed)
            rospy.logwarn_throttle(
                0.30,
                "CENTERLINE_RETREAT travel=%.3f/%.3f side_clear=%.3f "
                "rear_clear=%.3f cmd_x=-%.3f",
                travelled, self.precenter_retreat_distance, clearance,
                snapshot["rear_clear"], self.precenter_retreat_speed)
            rate.sleep()
        self.publish_zero(12)
        return False

    def park_inside_square(self):
        self.move_client.cancel_all_goals()
        self.publish_zero(5)
        self.center_ocr_value = None
        self.center_ocr_stamp = None
        self.set_parking_mode(True)
        self.publish_state(
            "CENTERLINE_FAST_PARKING_START",
            warehouse=self.target_warehouse)
        success = False
        try:
            if not self.align_real_wall("CENTERLINE_INITIAL_WALL_ALIGNMENT"):
                return False
            if not self.retreat_for_lateral_clearance():
                return False
            if not self.align_real_wall(
                    "CENTERLINE_POST_RETREAT_ALIGNMENT",
                    tolerance=math.radians(3.2)):
                return False
            if not self.center_on_target_sign():
                return False
            if not self.align_real_wall(
                    "CENTERLINE_FINAL_WALL_ALIGNMENT",
                    tolerance=math.radians(2.6)):
                return False
            if not self.approach_centered_wall():
                return False
            success = True
            return True
        finally:
            self.publish_zero(15)
            self.set_parking_mode(False)
            rospy.logwarn(
                "CENTERLINE_FAST_PARKING_END success=%s", str(success))


if __name__ == "__main__":
    FastClearanceCenterlineManager().run()
