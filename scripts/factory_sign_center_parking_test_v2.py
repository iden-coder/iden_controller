#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Faster sign-primary parking with white-frame and swept-side assistance."""

import math
import threading
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool

from factory_room_vision_parking_v7 import AdaptiveRangeParkingDetector
from factory_sign_center_parking_test_v1 import (
    SignCenterParkingTest,
    clamp,
)


def decode_image(msg):
    encoding = (msg.encoding or "").lower()
    channels = 4 if encoding in ("rgba8", "bgra8") else 3
    if encoding in ("mono8", "8uc1"):
        channels = 1
    try:
        raw = np.frombuffer(msg.data, dtype=np.uint8)
        row_bytes = int(msg.step) if msg.step else int(msg.width) * channels
        rows = raw[:int(msg.height) * row_bytes].reshape(
            (int(msg.height), row_bytes))
        packed = rows[:, :int(msg.width) * channels]
        if channels == 1:
            gray = packed.reshape((int(msg.height), int(msg.width)))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        image = packed.reshape((int(msg.height), int(msg.width), channels))
        if encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if encoding == "rgba8":
            return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        if encoding == "bgra8":
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if encoding == "bgr8":
            return np.ascontiguousarray(image)
    except (ValueError, IndexError):
        return None
    return None


class AssistedSignParkingTest(SignCenterParkingTest):
    def __init__(self):
        self.latest_frame = None
        self.image_time = 0.0
        self.image_lock = threading.RLock()
        self.last_frame_detection_time = 0.0
        self.cached_frame_result = None
        self.frame_stable_count = 0
        self.previous_frame_center = None
        self.left_swept_clear = float("inf")
        self.right_swept_clear = float("inf")
        # Defaults are needed before ROS subscribers created by V1 can deliver
        # the first scan callback. They are replaced by ROS params below.
        self.robot_half_length = 0.171
        self.robot_half_width = 0.128
        self.laser_offset_x = -0.11
        self.swept_front_margin = 0.10
        self.swept_rear_margin = 0.08
        super(AssistedSignParkingTest, self).__init__()

        self.image_topic = rospy.get_param(
            "~image_topic", "/ucar_camera/image_raw")
        self.debug_topic = rospy.get_param(
            "~debug_topic", "/factory/sign_parking_debug")
        self.frame_detector = AdaptiveRangeParkingDetector(max_width=800)
        self.frame_min_confidence = float(rospy.get_param(
            "~frame_min_confidence", 0.62))
        self.frame_stable_need = int(rospy.get_param(
            "~frame_stable_need", 3))
        self.frame_max_center_jump_px = float(rospy.get_param(
            "~frame_max_center_jump_px", 42.0))
        self.frame_assist_weight = float(rospy.get_param(
            "~frame_assist_weight", 0.25))
        self.frame_conflict_normalized = float(rospy.get_param(
            "~frame_conflict_normalized", 0.22))
        self.frame_process_period_s = float(rospy.get_param(
            "~frame_process_period_s", 0.16))
        self.approach_frame_max_y = float(rospy.get_param(
            "~approach_frame_max_lateral_speed", 0.028))
        self.approach_frame_min_wall_m = float(rospy.get_param(
            "~approach_frame_min_wall_m", 0.30))

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

        self.debug_pub = rospy.Publisher(
            self.debug_topic, Image, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_callback,
                         queue_size=1, buff_size=4 * 1024 * 1024)

    def image_callback(self, msg):
        frame = decode_image(msg)
        if frame is None:
            rospy.logwarn_throttle(2.0, "SIGN_PARK_V2 unsupported image")
            return
        with self.image_lock:
            self.latest_frame = frame
            self.image_time = time.time()

    def scan_callback(self, msg):
        super(AssistedSignParkingTest, self).scan_callback(msg)
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
        result = super(AssistedSignParkingTest, self).snapshot()
        with self.lock:
            result["left_swept_clear"] = self.left_swept_clear
            result["right_swept_clear"] = self.right_swept_clear
        return result

    def publish_debug(self, frame, result, sign_center=None):
        debug = result.get("debug") if result else None
        if debug is None:
            debug = frame.copy()
        else:
            debug = debug.copy()
        height, width = debug.shape[:2]
        target_x = int(round(width * self.optical_center_x_ratio))
        cv2.line(debug, (target_x, 0), (target_x, height - 1),
                 (255, 255, 0), 2)
        if sign_center is not None:
            sign_x = int(round(sign_center))
            cv2.line(debug, (sign_x, 0), (sign_x, height - 1),
                     (255, 0, 255), 2)
        rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        message = Image()
        message.header.stamp = rospy.Time.now()
        message.header.frame_id = "sign_parking_v2"
        message.height, message.width = rgb.shape[:2]
        message.encoding = "rgb8"
        message.is_bigendian = False
        message.step = message.width * 3
        message.data = np.ascontiguousarray(rgb).tobytes()
        self.debug_pub.publish(message)

    def detect_frame(self, sign_center=None):
        now = time.time()
        if (self.cached_frame_result is not None and
                now - self.last_frame_detection_time < self.frame_process_period_s):
            return self.cached_frame_result
        with self.image_lock:
            if self.latest_frame is None or now - self.image_time > 1.0:
                self.frame_stable_count = 0
                return None
            frame = self.latest_frame.copy()
        try:
            result = self.frame_detector.detect(frame)
        except Exception as exc:
            rospy.logwarn_throttle(1.0, "SIGN_PARK_V2 frame detector failed: %s", exc)
            self.frame_stable_count = 0
            return None
        self.last_frame_detection_time = now
        center = result.get("center_near_px")
        valid = (bool(result.get("found")) and center is not None and
                 float(result.get("confidence", 0.0)) >= self.frame_min_confidence)
        if valid:
            center = float(center)
            if (self.previous_frame_center is None or
                    abs(center - self.previous_frame_center) <=
                    self.frame_max_center_jump_px):
                self.frame_stable_count += 1
            else:
                self.frame_stable_count = 1
            self.previous_frame_center = center
        else:
            self.frame_stable_count = 0
            self.previous_frame_center = None
        result["assist_stable"] = (
            valid and self.frame_stable_count >= self.frame_stable_need)
        result["source_width"] = int(result.get("source_width", frame.shape[1]))
        self.cached_frame_result = result
        self.publish_debug(frame, result, sign_center=sign_center)
        return result

    def fused_center_error(self, ocr):
        width = float(ocr["image_width"])
        sign_center = float(ocr["filtered_center_x"])
        sign_normalized = ((sign_center - width * self.optical_center_x_ratio) /
                           max(1.0, 0.5 * width))
        result = self.detect_frame(sign_center=sign_center)
        if not result or not result.get("assist_stable"):
            return sign_normalized, "sign_only", None
        frame_width = float(result.get("source_width", width))
        frame_center = float(result["center_near_px"])
        frame_normalized = (
            (frame_center - frame_width * self.optical_center_x_ratio) /
            max(1.0, 0.5 * frame_width))
        conflict = (sign_normalized * frame_normalized < 0.0 and
                    abs(sign_normalized - frame_normalized) >
                    self.frame_conflict_normalized)
        if conflict:
            rospy.logwarn_throttle(
                0.7,
                "SIGN_PARK_V2 frame conflict ignored sign=%.3f frame=%.3f",
                sign_normalized, frame_normalized)
            return sign_normalized, "sign_primary_conflict", frame_normalized
        weight = clamp(self.frame_assist_weight, 0.0, 0.45)
        fused = (1.0 - weight) * sign_normalized + weight * frame_normalized
        return fused, "sign_plus_frame", frame_normalized

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

    def center_under_sign(self):
        self.publish_state("V2_CENTERING_SIGN_PRIMARY_FRAME_ASSIST")
        start_snapshot = self.snapshot()
        start_xy = start_snapshot["odom_xy"]
        stable = 0
        obstacle_since = None
        start = time.time()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() - start < self.lateral_timeout_s:
            snapshot = self.snapshot()
            if not self.sensors_fresh(snapshot, require_ocr=True):
                self.publish_zero()
                stable = 0
                rate.sleep()
                continue
            moved = self.odom_displacement(start_xy, snapshot["odom_xy"])
            if moved > self.lateral_limit_m:
                rospy.logerr("SIGN_PARK_V2 lateral travel limit %.3fm", moved)
                self.publish_zero(20)
                return False
            wall_error = snapshot["wall"]["heading_error"]
            if abs(wall_error) > 2.0 * self.heading_tolerance:
                self.publish_zero(8)
                if not self.align_wall_heading("V2_RECOVERING_WALL_HEADING"):
                    return False
                stable = 0
                continue
            ocr = snapshot["ocr"]
            normalized, mode, frame_normalized = self.fused_center_error(ocr)
            width = float(ocr["image_width"])
            pixel_error = normalized * 0.5 * width
            if abs(pixel_error) <= self.sign_center_tolerance_px:
                stable += 1
                obstacle_since = None
                cmd_y = 0.0
                clearance = float("inf")
                guard = "centered"
                self.publish_command()
            else:
                stable = 0
                requested = self.lateral_command_sign * self.lateral_kp * normalized
                if abs(requested) < self.lateral_min_speed:
                    requested = math.copysign(self.lateral_min_speed, requested)
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
                        "SIGN_PARK_V2 LATERAL_BLOCKED side_clear=%.3fm direction=%s",
                        clearance, "left" if requested > 0.0 else "right")
                    if time.time() - obstacle_since >= self.obstacle_wait_timeout_s:
                        self.publish_zero(30)
                        return False
                else:
                    obstacle_since = None
                    self.publish_command(y=cmd_y)
            rospy.logwarn_throttle(
                0.3,
                "SIGN_PARK_V2_CENTER mode=%s error=%.1fpx frame=%s "
                "stable=%d/%d clear=%.3f guard=%s cmd_y=%.3f",
                mode, pixel_error,
                "none" if frame_normalized is None else "%.3f" % frame_normalized,
                stable, self.sign_stable_frames, clearance, guard, cmd_y)
            if stable >= self.sign_stable_frames:
                self.publish_zero(15)
                self.publish_state("V2_SIGN_CENTERED", mode=mode,
                                   pixel_error=pixel_error,
                                   lateral_motion=moved)
                return True
            rate.sleep()
        self.publish_zero(30)
        return False

    def approach_frame_command(self, wall_distance, snapshot):
        if wall_distance <= self.approach_frame_min_wall_m:
            return 0.0, "disabled_close"
        result = self.detect_frame()
        if not result or not result.get("assist_stable"):
            return 0.0, "not_stable"
        width = float(result.get("source_width", 800.0))
        center = float(result["center_near_px"])
        normalized = ((center - width * self.optical_center_x_ratio) /
                      max(1.0, 0.5 * width))
        if abs(normalized * 0.5 * width) <= self.sign_center_tolerance_px:
            return 0.0, "centered"
        requested = self.lateral_command_sign * 0.10 * normalized
        requested = clamp(requested, -self.approach_frame_max_y,
                          self.approach_frame_max_y)
        command, clearance, guard = self.guarded_lateral_speed(
            requested, snapshot)
        if guard == "blocked":
            return 0.0, "blocked"
        return command, "frame_%s" % guard

    def approach_wall(self):
        self.publish_state("V2_APPROACHING_WALL_WITH_FRAME_ASSIST",
                           target_distance=self.final_wall_distance_m)
        start_xy = self.snapshot()["odom_xy"]
        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < self.approach_timeout_s:
            snapshot = self.snapshot()
            if not self.sensors_fresh(snapshot, require_ocr=False):
                self.publish_zero()
                stable = 0
                rate.sleep()
                continue
            wall = snapshot["wall"]["distance"]
            heading_error = snapshot["wall"]["heading_error"]
            travelled = self.odom_displacement(start_xy, snapshot["odom_xy"])
            distance_error = wall - self.final_wall_distance_m
            if (snapshot["front_min"] < self.front_emergency_m or
                    wall < self.final_wall_distance_m - self.final_fail_tolerance_m or
                    travelled > self.approach_max_travel_m):
                self.publish_zero(30)
                rospy.logerr("SIGN_PARK_V2 approach safety stop wall=%.3f front=%.3f travel=%.3f",
                             wall, snapshot["front_min"], travelled)
                return False
            cmd_x = 0.0
            cmd_y = 0.0
            wz = 0.0
            assist = "off"
            if abs(heading_error) > 2.0 * self.heading_tolerance:
                stable = 0
                wz = clamp(self.heading_kp * heading_error,
                           -self.heading_max_wz, self.heading_max_wz)
            elif abs(distance_error) <= self.final_wall_tolerance_m:
                stable += 1
            elif distance_error < 0.0:
                stable = 0
            else:
                stable = 0
                ratio = clamp(distance_error / self.approach_slow_error_m,
                              0.0, 1.0)
                cmd_x = (self.approach_slow_speed + ratio *
                         (self.approach_fast_speed - self.approach_slow_speed))
                wz = clamp(self.heading_kp * heading_error, -0.05, 0.05)
                cmd_y, assist = self.approach_frame_command(wall, snapshot)
            self.publish_command(x=cmd_x, y=cmd_y, wz=wz)
            rospy.logwarn_throttle(
                0.3,
                "SIGN_PARK_V2_APPROACH wall=%.3f error=%.3f heading=%.2fdeg "
                "assist=%s cmd=(%.3f,%.3f,%.3f) stable=%d/%d",
                wall, distance_error, math.degrees(heading_error), assist,
                cmd_x, cmd_y, wz, stable, self.final_stable_frames)
            if stable >= self.final_stable_frames:
                self.publish_zero(30)
                self.publish_state(
                    "PARKING_SUCCESS", wall_distance=wall,
                    nose_gap_estimate=wall - 0.061,
                    heading_error_deg=math.degrees(heading_error),
                    forward_motion=travelled,
                    controller="sign_primary_frame_assist_v2")
                rospy.logwarn(
                    "SIGN_PARK_V2_SUCCESS wall=%.3fm nose_gap=%.3fm heading=%.2fdeg",
                    wall, wall - 0.061, math.degrees(heading_error))
                return True
            rate.sleep()
        self.publish_zero(30)
        return False


if __name__ == "__main__":
    node = AssistedSignParkingTest()
    try:
        rospy.sleep(1.0)
        node.success = node.run()
        node.finished = True
        node.publish_zero(30)
        if not node.success:
            rospy.logerr("SIGN_PARK_V2_TEST_FAILED; vehicle remains stopped")
        rospy.logwarn("SIGN_PARK_V2_TEST_FINISHED; node holds zero")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            node.publish_zero()
            node.parking_pub.publish(Bool(data=True))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
