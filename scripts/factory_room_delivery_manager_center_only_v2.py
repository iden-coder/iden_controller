#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Robust centerline delivery manager with TF geometry and 180 degree search."""

import math
import time

import rospy
import tf
from geometry_msgs.msg import Twist

from factory_room_delivery_manager import FactoryRoomDeliveryManager, norm_angle
from factory_room_delivery_manager_center_only_v1 import (
    CenterlineFactoryDeliveryManager,
)


class RobustCenterlineFactoryDeliveryManager(
        CenterlineFactoryDeliveryManager):
    def __init__(self):
        self.scan_tf_ready = False
        self.scan_tf_listener = None
        self.mission_deadline = None
        self.mission_timeout_hit = False
        super(RobustCenterlineFactoryDeliveryManager, self).__init__()
        self.scan_total_angle = math.radians(float(rospy.get_param(
            "~workshop_scan_total_deg", 180.0)))
        self.scan_tf_fallback_x = float(rospy.get_param(
            "~center_laser_offset_x_m", 0.11))
        self.scan_tf_fallback_y = float(rospy.get_param(
            "~center_laser_offset_y_m", 0.0))
        self.scan_tf_fallback_yaw = float(rospy.get_param(
            "~center_laser_yaw_rad", -0.07))
        self.scan_tf_listener = tf.TransformListener()
        self.scan_tf_ready = True
        rospy.logwarn(
            "CENTERLINE_MANAGER_V2 scan=%.0fdeg tf_geometry=true "
            "mission_timeout=%.1fs",
            math.degrees(self.scan_total_angle), self.mission_timeout_s)

    def _scan_transform(self, msg):
        frame_id = msg.header.frame_id or "laser_frame"
        if self.scan_tf_ready:
            try:
                trans, rotation = self.scan_tf_listener.lookupTransform(
                    "base_link", frame_id, rospy.Time(0))
                yaw = tf.transformations.euler_from_quaternion(rotation)[2]
                return trans[0], trans[1], yaw, True
            except Exception as exc:
                rospy.logwarn_throttle(
                    1.0, "CENTERLINE_SCAN_TF_FALLBACK frame=%s error=%s",
                    frame_id, exc)
        return (self.scan_tf_fallback_x, self.scan_tf_fallback_y,
                self.scan_tf_fallback_yaw, False)

    def scan_callback(self, msg):
        FactoryRoomDeliveryManager.scan_callback(self, msg)
        if not self.scan_tf_ready:
            return
        tx, ty, tf_yaw, _ = self._scan_transform(msg)
        cos_yaw = math.cos(tf_yaw)
        sin_yaw = math.sin(tf_yaw)
        wall_points = []
        left_clear = float("inf")
        right_clear = float("inf")
        min_x = -self.robot_half_length - self.swept_rear_margin
        max_x = self.robot_half_length + self.swept_front_margin
        for index, distance in enumerate(msg.ranges):
            if (not math.isfinite(distance) or distance < msg.range_min or
                    distance > msg.range_max):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            x_laser = distance * math.cos(angle)
            y_laser = distance * math.sin(angle)
            x_base = tx + cos_yaw * x_laser - sin_yaw * y_laser
            y_base = ty + sin_yaw * x_laser + cos_yaw * y_laser
            if (abs(angle) <= self.wall_fit_sector and
                    self.wall_min_range <= distance <= self.wall_max_range):
                wall_points.append((x_laser, y_laser))
            if min_x <= x_base <= max_x:
                if y_base > self.robot_half_width:
                    left_clear = min(
                        left_clear, y_base - self.robot_half_width)
                elif y_base < -self.robot_half_width:
                    right_clear = min(
                        right_clear, -y_base - self.robot_half_width)

        model = self.fit_front_wall(wall_points)
        now = time.time()
        with self.lock:
            if model is not None:
                model["heading_error"] = norm_angle(
                    model["heading_error"] + tf_yaw)
                previous = self.center_wall_model
                if previous is not None:
                    alpha = 0.32
                    delta = norm_angle(
                        model["heading_error"] - previous["heading_error"])
                    model["heading_error"] = (
                        previous["heading_error"] + alpha * delta)
                    model["distance"] = (
                        previous["distance"] + alpha *
                        (model["distance"] - previous["distance"]))
                self.center_wall_model = model
                self.center_scan_time = now
            self.left_swept_clear = left_clear
            self.right_swept_clear = right_clear

    def rotate_relative(self, delta_yaw):
        _, start_yaw = self.snapshot_odom()
        if start_yaw is None:
            return False
        target = norm_angle(start_yaw + delta_yaw)
        timeout = max(
            4.0, abs(delta_yaw) / max(self.scan_turn_speed, 0.1) * 2.4)
        start = time.time()
        rate = rospy.Rate(16)
        while not rospy.is_shutdown() and time.time() - start < timeout:
            if self.target_found.is_set():
                self.publish_zero(3)
                return True
            _, yaw = self.snapshot_odom()
            if yaw is None:
                break
            error = norm_angle(target - yaw)
            if abs(error) <= math.radians(3.0):
                self.publish_zero(3)
                return True
            command = Twist()
            command.angular.z = math.copysign(
                max(0.13, min(self.scan_turn_speed, abs(error) * 1.2)),
                error)
            self.cmd_pub.publish(command)
            rate.sleep()
        self.publish_zero(5)
        return False

    def scan_for_target_at_current_pose(self, point_name):
        self.publish_state("SCANNING_WORKSHOP", point=point_name,
                           target=self.target_warehouse,
                           sweep_deg=math.degrees(self.scan_total_angle))
        self.target_found.clear()
        self.ocr_control("reset")
        self.ocr_control("enable")
        rospy.sleep(1.5)
        if self.target_found.is_set():
            self.publish_zero(3)
            return True
        steps = max(2, int(round(
            math.degrees(self.scan_total_angle) /
            max(self.scan_step_deg, 10.0))))
        step_radians = self.scan_total_angle / float(steps)
        for step in range(steps):
            if rospy.is_shutdown() or self._mission_expired():
                return False
            if self.target_found.is_set():
                self.publish_zero(3)
                return True
            if not self.rotate_relative(step_radians):
                if self.target_found.is_set():
                    return True
                rospy.logwarn(
                    "scan rotation safely blocked at %s step=%d/%d",
                    point_name, step + 1, steps)
                break
            rospy.sleep(self.scan_settle_s)
        return self.target_found.is_set()

    def _mission_expired(self):
        if self.mission_deadline is None:
            return False
        expired = time.time() >= self.mission_deadline
        if expired:
            self.mission_timeout_hit = True
        return expired

    def find_target_workshop(self):
        for cycle in range(max(1, self.search_cycles)):
            points = (self.observation_points if cycle % 2 == 0
                      else list(reversed(self.observation_points)))
            for point in points:
                if rospy.is_shutdown() or self._mission_expired():
                    return False
                self.ocr_control("disable")
                reached = self.navigate(
                    point["name"], point["x"], point["y"], point["yaw"],
                    allow_offsets=True)
                if not reached:
                    continue
                if self.scan_for_target_at_current_pose(point["name"]):
                    return True
            rospy.logwarn(
                "WORKSHOP_SEARCH_CYCLE_COMPLETE cycle=%d/%d target=%s",
                cycle + 1, self.search_cycles, self.target_warehouse)
            self.clear_navigation_costmaps()
        return False

    def mission_thread(self):
        self.mission_timeout_hit = False
        self.mission_deadline = time.time() + self.mission_timeout_s
        try:
            super(RobustCenterlineFactoryDeliveryManager, self).mission_thread()
        finally:
            self.mission_deadline = None

    def fail(self, reason):
        if self.mission_timeout_hit:
            reason = "房间任务达到安全时限，已停止并释放后续任务"
        super(RobustCenterlineFactoryDeliveryManager, self).fail(reason)


if __name__ == "__main__":
    RobustCenterlineFactoryDeliveryManager().run()
