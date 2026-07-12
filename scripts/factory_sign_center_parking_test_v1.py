#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone workshop-sign parking using OCR, lidar wall fitting and odometry."""

import json
import math
import threading
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String


WORKSHOP_LABELS = (
    "食品加工车间",
    "日用品加工车间",
    "电子产品生产车间",
)


def clamp(value, low, high):
    return max(low, min(high, value))


def norm_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class SignCenterParkingTest(object):
    def __init__(self):
        rospy.init_node("factory_sign_center_parking_test_v1")
        self.lock = threading.RLock()

        self.ocr_topic = rospy.get_param(
            "~ocr_topic", "/factory_room/ocr_result")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel_raw")
        self.parking_mode_topic = rospy.get_param(
            "~parking_mode_topic", "/factory/parking_close_mode")
        self.state_topic = rospy.get_param(
            "~state_topic", "/factory/sign_center_parking_state")

        self.target_label = str(rospy.get_param("~target_label", "")).strip()
        self.start_delay_s = float(rospy.get_param("~start_delay_s", 5.0))
        self.sensor_timeout_s = float(rospy.get_param("~sensor_timeout_s", 1.0))
        self.input_wait_timeout_s = float(rospy.get_param(
            "~input_wait_timeout_s", 45.0))

        self.wall_sector_deg = float(rospy.get_param("~wall_sector_deg", 42.0))
        self.wall_min_range_m = float(rospy.get_param(
            "~wall_min_range_m", 0.10))
        self.wall_max_range_m = float(rospy.get_param(
            "~wall_max_range_m", 1.60))
        self.wall_inlier_m = float(rospy.get_param("~wall_inlier_m", 0.025))
        self.wall_min_inliers = int(rospy.get_param("~wall_min_inliers", 18))
        self.wall_min_span_m = float(rospy.get_param(
            "~wall_min_span_m", 0.28))

        self.heading_tolerance = math.radians(float(rospy.get_param(
            "~heading_tolerance_deg", 1.8)))
        self.heading_abort = math.radians(float(rospy.get_param(
            "~heading_abort_deg", 18.0)))
        self.heading_kp = float(rospy.get_param("~heading_kp", 1.35))
        self.heading_max_wz = float(rospy.get_param("~heading_max_wz", 0.18))
        self.heading_stable_frames = int(rospy.get_param(
            "~heading_stable_frames", 6))
        self.heading_timeout_s = float(rospy.get_param(
            "~heading_timeout_s", 12.0))
        self.cardinal_warn_deg = float(rospy.get_param(
            "~cardinal_warn_deg", 8.0))

        self.image_mirrored = bool(rospy.get_param("~image_mirrored", True))
        default_sign = 1.0 if self.image_mirrored else -1.0
        self.lateral_command_sign = float(rospy.get_param(
            "~lateral_command_sign", default_sign))
        self.optical_center_x_ratio = float(rospy.get_param(
            "~optical_center_x_ratio", 0.5))
        self.sign_center_tolerance_px = float(rospy.get_param(
            "~sign_center_tolerance_px", 14.0))
        self.sign_stable_frames = int(rospy.get_param(
            "~sign_stable_frames", 5))
        self.lateral_kp = float(rospy.get_param("~lateral_kp", 0.16))
        self.lateral_min_speed = float(rospy.get_param(
            "~lateral_min_speed", 0.018))
        self.lateral_max_speed = float(rospy.get_param(
            "~lateral_max_speed", 0.055))
        self.lateral_limit_m = float(rospy.get_param(
            "~lateral_limit_m", 0.45))
        self.lateral_timeout_s = float(rospy.get_param(
            "~lateral_timeout_s", 20.0))
        self.side_hard_clearance_m = float(rospy.get_param(
            "~side_hard_clearance_m", 0.15))
        self.ocr_ema_alpha = float(rospy.get_param("~ocr_ema_alpha", 0.35))

        self.final_wall_distance_m = float(rospy.get_param(
            "~final_wall_distance_m", 0.171))
        self.final_wall_tolerance_m = float(rospy.get_param(
            "~final_wall_tolerance_m", 0.008))
        self.final_fail_tolerance_m = float(rospy.get_param(
            "~final_fail_tolerance_m", 0.025))
        self.front_emergency_m = float(rospy.get_param(
            "~front_emergency_m", 0.105))
        self.approach_fast_speed = float(rospy.get_param(
            "~approach_fast_speed", 0.085))
        self.approach_slow_speed = float(rospy.get_param(
            "~approach_slow_speed", 0.025))
        self.approach_slow_error_m = float(rospy.get_param(
            "~approach_slow_error_m", 0.18))
        self.approach_max_travel_m = float(rospy.get_param(
            "~approach_max_travel_m", 0.90))
        self.approach_timeout_s = float(rospy.get_param(
            "~approach_timeout_s", 24.0))
        self.final_stable_frames = int(rospy.get_param(
            "~final_stable_frames", 6))

        self.latest_ocr = None
        self.ocr_time = 0.0
        self.ocr_center_ema = None
        self.wall_model = None
        self.scan_time = 0.0
        self.front_min = float("inf")
        self.left_min = float("inf")
        self.right_min = float("inf")
        self.odom_xy = None
        self.odom_yaw = None
        self.odom_time = 0.0
        self.finished = False
        self.success = False

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.parking_pub = rospy.Publisher(
            self.parking_mode_topic, Bool, queue_size=1, latch=True)
        self.state_pub = rospy.Publisher(
            self.state_topic, String, queue_size=5, latch=True)
        rospy.Subscriber(self.ocr_topic, String, self.ocr_callback, queue_size=5)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback,
                         queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback,
                         queue_size=1)
        rospy.on_shutdown(self.shutdown)

    def publish_state(self, state, **values):
        payload = {"state": state, "stamp": time.time()}
        payload.update(values)
        self.state_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn("SIGN_PARK_STATE %s",
                      json.dumps(payload, ensure_ascii=False))

    def publish_zero(self, repeats=1):
        for _ in range(max(1, repeats)):
            self.cmd_pub.publish(Twist())
            if repeats > 1:
                rospy.sleep(0.025)

    def publish_command(self, x=0.0, y=0.0, wz=0.0):
        command = Twist()
        command.linear.x = float(x)
        command.linear.y = float(y)
        command.angular.z = float(wz)
        self.parking_pub.publish(Bool(data=True))
        self.cmd_pub.publish(command)

    def ocr_callback(self, msg):
        try:
            payload = json.loads(msg.data)
            bbox = payload.get("bbox")
            width = float(payload.get("image_width", 0.0))
            label = str(payload.get("label", "")).strip()
            if (not payload.get("stable") or label not in WORKSHOP_LABELS or
                    not bbox or len(bbox) != 4 or width <= 1.0):
                return
            if self.target_label and label != self.target_label:
                return
            center = 0.5 * (float(bbox[0]) + float(bbox[2]))
            with self.lock:
                if self.ocr_center_ema is None:
                    self.ocr_center_ema = center
                else:
                    alpha = clamp(self.ocr_ema_alpha, 0.05, 1.0)
                    self.ocr_center_ema = (
                        alpha * center + (1.0 - alpha) * self.ocr_center_ema)
                payload["filtered_center_x"] = self.ocr_center_ema
                self.latest_ocr = payload
                self.ocr_time = time.time()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "SIGN_PARK bad OCR payload: %s", exc)

    def odom_callback(self, msg):
        pose = msg.pose.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        with self.lock:
            self.odom_xy = (pose.position.x, pose.position.y)
            self.odom_yaw = yaw
            self.odom_time = time.time()

    def scan_callback(self, msg):
        points = []
        front = []
        left = []
        right = []
        sector = math.radians(self.wall_sector_deg)
        for index, distance in enumerate(msg.ranges):
            if not math.isfinite(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            if abs(angle) <= math.radians(10.0):
                front.append(distance)
            if math.radians(48.0) <= angle <= math.radians(112.0):
                left.append(distance)
            if math.radians(-112.0) <= angle <= math.radians(-48.0):
                right.append(distance)
            if (abs(angle) <= sector and
                    self.wall_min_range_m <= distance <= self.wall_max_range_m):
                points.append((distance * math.cos(angle),
                               distance * math.sin(angle)))
        model = self.fit_wall(points)
        now = time.time()
        with self.lock:
            self.wall_model = model
            self.scan_time = now
            self.front_min = min(front) if front else float("inf")
            self.left_min = min(left) if left else float("inf")
            self.right_min = min(right) if right else float("inf")

    def fit_wall(self, points):
        if len(points) < self.wall_min_inliers:
            return None
        data = np.asarray(points, dtype=np.float64)
        count = len(data)
        stride = max(1, count // 34)
        candidates = range(0, count, stride)
        best_indices = None
        best_score = -1.0
        threshold = self.wall_inlier_m
        for first in candidates:
            for second in candidates:
                if second <= first + max(2, stride):
                    continue
                dx = data[second, 0] - data[first, 0]
                dy = data[second, 1] - data[first, 1]
                length = math.hypot(dx, dy)
                if length < self.wall_min_span_m * 0.55:
                    continue
                normal = np.array([-dy / length, dx / length])
                residual = np.abs((data - data[first]).dot(normal))
                indices = np.flatnonzero(residual <= threshold)
                if len(indices) < self.wall_min_inliers:
                    continue
                tangent = np.array([dx / length, dy / length])
                projected = data[indices].dot(tangent)
                span = float(projected.max() - projected.min())
                if span < self.wall_min_span_m:
                    continue
                score = len(indices) + 12.0 * span
                if score > best_score:
                    best_score = score
                    best_indices = indices
        if best_indices is None:
            return None
        inliers = data[best_indices]
        centroid = inliers.mean(axis=0)
        covariance = np.cov((inliers - centroid).T)
        values, vectors = np.linalg.eigh(covariance)
        normal = vectors[:, int(np.argmin(values))]
        if float(normal.dot(centroid)) < 0.0:
            normal = -normal
        tangent = np.array([-normal[1], normal[0]])
        projected = inliers.dot(tangent)
        span = float(projected.max() - projected.min())
        residual = np.abs((inliers - centroid).dot(normal))
        distance = float(normal.dot(centroid))
        heading_error = math.atan2(float(normal[1]), float(normal[0]))
        if distance <= 0.0 or abs(heading_error) > self.heading_abort:
            return None
        return {
            "distance": distance,
            "heading_error": heading_error,
            "inliers": int(len(inliers)),
            "span": span,
            "rms": float(math.sqrt(np.mean(residual * residual))),
        }

    def snapshot(self):
        with self.lock:
            return {
                "ocr": None if self.latest_ocr is None else dict(self.latest_ocr),
                "ocr_time": self.ocr_time,
                "wall": None if self.wall_model is None else dict(self.wall_model),
                "scan_time": self.scan_time,
                "front_min": self.front_min,
                "left_min": self.left_min,
                "right_min": self.right_min,
                "odom_xy": self.odom_xy,
                "odom_yaw": self.odom_yaw,
                "odom_time": self.odom_time,
            }

    @staticmethod
    def odom_displacement(start_xy, current_xy):
        if start_xy is None or current_xy is None:
            return float("inf")
        return math.hypot(current_xy[0] - start_xy[0],
                          current_xy[1] - start_xy[1])

    def sensors_fresh(self, snapshot, require_ocr=True):
        now = time.time()
        if (snapshot["wall"] is None or
                now - snapshot["scan_time"] > self.sensor_timeout_s or
                snapshot["odom_xy"] is None or
                now - snapshot["odom_time"] > self.sensor_timeout_s):
            return False
        if require_ocr and (snapshot["ocr"] is None or
                            now - snapshot["ocr_time"] > self.sensor_timeout_s):
            return False
        return True

    def wait_for_inputs(self):
        self.publish_state("WAITING_INPUTS")
        start = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start < self.input_wait_timeout_s:
            self.publish_zero()
            self.parking_pub.publish(Bool(data=True))
            snapshot = self.snapshot()
            if self.sensors_fresh(snapshot, require_ocr=True):
                ocr = snapshot["ocr"]
                wall = snapshot["wall"]
                rospy.logwarn(
                    "SIGN_PARK_INPUTS_OK label=%s bbox=%s wall=%.3fm heading=%.2fdeg",
                    ocr.get("label"), ocr.get("bbox"), wall["distance"],
                    math.degrees(wall["heading_error"]))
                return True
            rospy.logwarn_throttle(
                1.0, "SIGN_PARK_WAITING OCR, lidar wall and odometry")
            rate.sleep()
        return False

    def countdown(self):
        self.publish_state("READY_COUNTDOWN", delay_s=self.start_delay_s)
        end = time.time() + self.start_delay_s
        while not rospy.is_shutdown() and time.time() < end:
            self.publish_zero()
            self.parking_pub.publish(Bool(data=True))
            rospy.logwarn_throttle(
                1.0, "SIGN_PARK starts in %.1fs; vehicle held stopped",
                max(0.0, end - time.time()))
            rospy.sleep(0.05)

    def align_wall_heading(self, state_name):
        self.publish_state(state_name)
        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < self.heading_timeout_s:
            snapshot = self.snapshot()
            if not self.sensors_fresh(snapshot, require_ocr=False):
                self.publish_zero()
                stable = 0
                rate.sleep()
                continue
            error = snapshot["wall"]["heading_error"]
            if abs(error) <= self.heading_tolerance:
                stable += 1
                self.publish_command()
            else:
                stable = 0
                wz = clamp(self.heading_kp * error,
                           -self.heading_max_wz, self.heading_max_wz)
                self.publish_command(wz=wz)
            rospy.logwarn_throttle(
                0.35,
                "SIGN_PARK_HEADING wall=%.3f error=%.2fdeg stable=%d/%d",
                snapshot["wall"]["distance"], math.degrees(error), stable,
                self.heading_stable_frames)
            if stable >= self.heading_stable_frames:
                self.publish_zero(12)
                yaw = snapshot["odom_yaw"]
                nearest = round(yaw / (0.5 * math.pi)) * (0.5 * math.pi)
                cardinal_error = math.degrees(norm_angle(nearest - yaw))
                qz = math.sin(0.5 * yaw)
                qw = math.cos(0.5 * yaw)
                rospy.logwarn(
                    "SIGN_PARK_HEADING_OK odom_yaw=%.2fdeg qz=%.5f qw=%.5f "
                    "nearest_cardinal_error=%.2fdeg",
                    math.degrees(yaw), qz, qw, cardinal_error)
                if abs(cardinal_error) > self.cardinal_warn_deg:
                    rospy.logwarn(
                        "SIGN_PARK cardinal mismatch; lidar perpendicular is retained")
                return True
            rate.sleep()
        self.publish_zero(20)
        return False

    def center_under_sign(self):
        self.publish_state("CENTERING_UNDER_SIGN")
        start_snapshot = self.snapshot()
        start_xy = start_snapshot["odom_xy"]
        stable = 0
        start = time.time()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() - start < self.lateral_timeout_s:
            snapshot = self.snapshot()
            if not self.sensors_fresh(snapshot, require_ocr=True):
                self.publish_zero()
                stable = 0
                rospy.logwarn_throttle(1.0, "SIGN_PARK_CENTER paused: stale input")
                rate.sleep()
                continue
            moved = self.odom_displacement(start_xy, snapshot["odom_xy"])
            if moved > self.lateral_limit_m:
                rospy.logerr("SIGN_PARK_CENTER travel limit %.3fm exceeded", moved)
                self.publish_zero(20)
                return False
            wall_error = snapshot["wall"]["heading_error"]
            if abs(wall_error) > 2.0 * self.heading_tolerance:
                self.publish_zero(8)
                if not self.align_wall_heading("RECOVERING_WALL_HEADING"):
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
                self.publish_command()
                cmd_y = 0.0
            else:
                stable = 0
                normalized = pixel_error / max(1.0, 0.5 * width)
                cmd_y = self.lateral_command_sign * self.lateral_kp * normalized
                if abs(cmd_y) < self.lateral_min_speed:
                    cmd_y = math.copysign(self.lateral_min_speed, cmd_y)
                cmd_y = clamp(cmd_y, -self.lateral_max_speed,
                              self.lateral_max_speed)
                side_clear = (snapshot["left_min"] if cmd_y > 0.0
                              else snapshot["right_min"])
                if side_clear < self.side_hard_clearance_m:
                    rospy.logerr(
                        "SIGN_PARK_CENTER side blocked clear=%.3fm cmd_y=%.3f",
                        side_clear, cmd_y)
                    self.publish_zero(20)
                    return False
                self.publish_command(y=cmd_y)
            rospy.logwarn_throttle(
                0.35,
                "SIGN_PARK_CENTER label=%s u=%.1f target=%.1f error=%.1fpx "
                "stable=%d/%d moved=%.3fm cmd_y=%.3f",
                ocr.get("label"), center, target, pixel_error, stable,
                self.sign_stable_frames, moved, cmd_y)
            if stable >= self.sign_stable_frames:
                self.publish_zero(15)
                self.publish_state(
                    "SIGN_CENTERED", label=ocr.get("label"),
                    pixel_error=pixel_error, lateral_motion=moved)
                return True
            rate.sleep()
        self.publish_zero(20)
        return False

    def approach_wall(self):
        self.publish_state("APPROACHING_WALL",
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
                rospy.logwarn_throttle(1.0, "SIGN_PARK_APPROACH paused: stale lidar/odom")
                rate.sleep()
                continue
            wall = snapshot["wall"]["distance"]
            heading_error = snapshot["wall"]["heading_error"]
            travelled = self.odom_displacement(start_xy, snapshot["odom_xy"])
            distance_error = wall - self.final_wall_distance_m
            if (snapshot["front_min"] < self.front_emergency_m or
                    wall < self.final_wall_distance_m - self.final_fail_tolerance_m):
                rospy.logerr(
                    "SIGN_PARK_APPROACH emergency stop wall=%.3f front=%.3f",
                    wall, snapshot["front_min"])
                self.publish_zero(30)
                return False
            if travelled > self.approach_max_travel_m:
                rospy.logerr("SIGN_PARK_APPROACH travel limit %.3fm exceeded",
                             travelled)
                self.publish_zero(30)
                return False
            if abs(heading_error) > 2.0 * self.heading_tolerance:
                stable = 0
                wz = clamp(self.heading_kp * heading_error,
                           -self.heading_max_wz, self.heading_max_wz)
                self.publish_command(wz=wz)
                cmd_x = 0.0
            elif abs(distance_error) <= self.final_wall_tolerance_m:
                cmd_x = 0.0
                stable += 1
                self.publish_command()
            elif distance_error < 0.0:
                cmd_x = 0.0
                stable = 0
                self.publish_command()
            else:
                stable = 0
                ratio = clamp(distance_error / self.approach_slow_error_m,
                              0.0, 1.0)
                cmd_x = (self.approach_slow_speed + ratio *
                         (self.approach_fast_speed - self.approach_slow_speed))
                wz = clamp(self.heading_kp * heading_error, -0.05, 0.05)
                self.publish_command(x=cmd_x, wz=wz)
            rospy.logwarn_throttle(
                0.3,
                "SIGN_PARK_APPROACH wall=%.3f target=%.3f error=%.3f "
                "heading=%.2fdeg travelled=%.3f stable=%d/%d cmd_x=%.3f",
                wall, self.final_wall_distance_m, distance_error,
                math.degrees(heading_error), travelled, stable,
                self.final_stable_frames, cmd_x)
            if stable >= self.final_stable_frames:
                self.publish_zero(30)
                self.publish_state(
                    "PARKING_SUCCESS", wall_distance=wall,
                    nose_gap_estimate=wall - 0.061,
                    heading_error_deg=math.degrees(heading_error),
                    forward_motion=travelled)
                rospy.logwarn(
                    "SIGN_PARK_SUCCESS wall=%.3fm estimated_nose_gap=%.3fm "
                    "heading_error=%.2fdeg",
                    wall, wall - 0.061, math.degrees(heading_error))
                return True
            rate.sleep()
        self.publish_zero(30)
        return False

    def run(self):
        self.parking_pub.publish(Bool(data=True))
        self.publish_zero(20)
        if not self.wait_for_inputs():
            self.publish_state("FAILED", reason="inputs_unavailable")
            return False
        self.countdown()
        if not self.align_wall_heading("ALIGNING_TO_WALL"):
            self.publish_state("FAILED", reason="wall_alignment_failed")
            return False
        if not self.center_under_sign():
            self.publish_state("FAILED", reason="sign_centering_failed")
            return False
        if not self.align_wall_heading("FINAL_HEADING_ALIGNMENT"):
            self.publish_state("FAILED", reason="final_alignment_failed")
            return False
        if not self.approach_wall():
            self.publish_state("FAILED", reason="wall_approach_failed")
            return False
        return True

    def shutdown(self):
        self.publish_zero(40)
        self.parking_pub.publish(Bool(data=False))


if __name__ == "__main__":
    node = SignCenterParkingTest()
    try:
        rospy.sleep(1.0)
        node.success = node.run()
        node.finished = True
        node.publish_zero(30)
        if not node.success:
            rospy.logerr("SIGN_PARK_TEST_FAILED; vehicle remains stopped")
        rospy.logwarn("SIGN_PARK_TEST_FINISHED; node remains alive and holds zero")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            node.publish_zero()
            node.parking_pub.publish(Bool(data=True))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

