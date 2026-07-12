#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Full factory-room delivery with lidar-wall and OCR-centerline parking."""

import math
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from factory_room_delivery_manager import FactoryRoomDeliveryManager


def clamp(value, low, high):
    return max(low, min(high, value))


class CenterlineFactoryDeliveryManager(FactoryRoomDeliveryManager):
    def __init__(self):
        # Defaults are needed before the base constructor's scan subscriber can
        # dispatch into this subclass.
        self.center_wall_model = None
        self.center_scan_time = 0.0
        self.left_swept_clear = float("inf")
        self.right_swept_clear = float("inf")
        self.robot_half_length = 0.171
        self.robot_half_width = 0.128
        self.laser_offset_x = -0.11
        self.swept_front_margin = 0.10
        self.swept_rear_margin = 0.08
        self.wall_fit_sector = math.radians(68.0)
        self.wall_min_range = 0.10
        self.wall_max_range = 2.00
        self.wall_inlier = 0.025
        self.wall_min_inliers = 18
        self.wall_min_span = 0.28
        self.wall_heading_abort = math.radians(45.0)
        super(CenterlineFactoryDeliveryManager, self).__init__()

        self.robot_half_length = float(rospy.get_param(
            "~center_robot_half_length_m", 0.171))
        self.robot_half_width = float(rospy.get_param(
            "~center_robot_half_width_m", 0.128))
        self.laser_offset_x = float(rospy.get_param(
            "~center_laser_offset_x_m", -0.11))
        self.swept_front_margin = float(rospy.get_param(
            "~center_swept_front_margin_m", 0.10))
        self.swept_rear_margin = float(rospy.get_param(
            "~center_swept_rear_margin_m", 0.08))
        self.wall_fit_sector = math.radians(float(rospy.get_param(
            "~center_wall_sector_deg", 68.0)))
        self.wall_min_range = float(rospy.get_param(
            "~center_wall_min_range_m", 0.10))
        self.wall_max_range = float(rospy.get_param(
            "~center_wall_max_range_m", 2.00))
        self.wall_inlier = float(rospy.get_param(
            "~center_wall_inlier_m", 0.025))
        self.wall_min_inliers = int(rospy.get_param(
            "~center_wall_min_inliers", 18))
        self.wall_min_span = float(rospy.get_param(
            "~center_wall_min_span_m", 0.28))
        self.wall_heading_abort = math.radians(float(rospy.get_param(
            "~center_wall_heading_abort_deg", 45.0)))

        self.heading_tolerance = math.radians(float(rospy.get_param(
            "~center_heading_tolerance_deg", 2.8)))
        self.lateral_realign = math.radians(float(rospy.get_param(
            "~center_lateral_realign_deg", 7.0)))
        self.approach_heading_limit = math.radians(float(rospy.get_param(
            "~center_approach_heading_limit_deg", 4.5)))
        self.heading_kp = float(rospy.get_param(
            "~center_heading_kp", 1.35))
        self.heading_min_wz = float(rospy.get_param(
            "~center_heading_min_wz", 0.115))
        self.heading_stuck_wz = float(rospy.get_param(
            "~center_heading_stuck_wz", 0.155))
        self.heading_max_wz = float(rospy.get_param(
            "~center_heading_max_wz", 0.22))
        self.heading_timeout = float(rospy.get_param(
            "~center_heading_timeout_s", 16.0))
        self.heading_stable_frames = int(rospy.get_param(
            "~center_heading_stable_frames", 3))

        self.sign_tolerance_px = float(rospy.get_param(
            "~center_sign_tolerance_px", 20.0))
        self.sign_stable_frames = int(rospy.get_param(
            "~center_sign_stable_frames", 3))
        self.sign_ema_alpha = float(rospy.get_param(
            "~center_sign_ema_alpha", 0.35))
        self.optical_center_ratio = float(rospy.get_param(
            "~center_optical_x_ratio", 0.5))
        self.lateral_command_sign = float(rospy.get_param(
            "~center_lateral_command_sign", -1.0))
        self.lateral_kp = float(rospy.get_param(
            "~center_lateral_kp", 0.28))
        self.lateral_min_speed = float(rospy.get_param(
            "~center_lateral_min_speed", 0.025))
        self.lateral_near_min_speed = float(rospy.get_param(
            "~center_lateral_near_min_speed", 0.010))
        self.lateral_near_band_px = float(rospy.get_param(
            "~center_lateral_near_band_px", 44.0))
        self.lateral_max_speed = float(rospy.get_param(
            "~center_lateral_max_speed", 0.090))
        self.lateral_limit = float(rospy.get_param(
            "~center_lateral_limit_m", 0.45))
        self.lateral_timeout = float(rospy.get_param(
            "~center_lateral_timeout_s", 20.0))
        self.lateral_hard_clearance = float(rospy.get_param(
            "~center_lateral_hard_clearance_m", 0.10))
        self.lateral_slow_clearance = float(rospy.get_param(
            "~center_lateral_slow_clearance_m", 0.34))
        self.obstacle_wait_timeout = float(rospy.get_param(
            "~center_obstacle_wait_timeout_s", 3.0))

        self.final_wall_distance = float(rospy.get_param(
            "~center_final_wall_distance_m", 0.127))
        self.final_wall_tolerance = float(rospy.get_param(
            "~center_final_wall_tolerance_m", 0.006))
        self.final_fail_tolerance = float(rospy.get_param(
            "~center_final_fail_tolerance_m", 0.018))
        self.front_emergency = float(rospy.get_param(
            "~center_front_emergency_m", 0.105))
        self.approach_fast_speed = float(rospy.get_param(
            "~center_approach_fast_speed", 0.100))
        self.approach_slow_speed = float(rospy.get_param(
            "~center_approach_slow_speed", 0.024))
        self.approach_slow_error = float(rospy.get_param(
            "~center_approach_slow_error_m", 0.15))
        self.approach_max_travel = float(rospy.get_param(
            "~center_approach_max_travel_m", 0.95))
        self.approach_timeout = float(rospy.get_param(
            "~center_approach_timeout_s", 28.0))
        self.final_stable_frames = int(rospy.get_param(
            "~center_final_stable_frames", 6))

        self.center_ocr_value = None
        self.center_ocr_stamp = None
        self.parking_mode = False
        self.parking_mode_pub = rospy.Publisher(
            rospy.get_param("~parking_close_mode_topic",
                            "/factory/parking_close_mode"),
            Bool, queue_size=1, latch=True)
        rospy.logwarn(
            "CENTERLINE_FULL_MANAGER_READY wall_target=%.3fm nose_gap=%.3fm",
            self.final_wall_distance, self.final_wall_distance - 0.061)

    def scan_callback(self, msg):
        super(CenterlineFactoryDeliveryManager, self).scan_callback(msg)
        points = []
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
            y_base = distance * math.sin(angle)
            x_base = x_laser + self.laser_offset_x
            if (abs(angle) <= self.wall_fit_sector and
                    self.wall_min_range <= distance <= self.wall_max_range):
                # Keep the wall model in the laser frame because the configured
                # 0.127 m target is a laser-to-wall measurement. Translation
                # does not affect the fitted wall angle.
                points.append((x_laser, y_base))
            if min_x <= x_base <= max_x:
                if y_base > self.robot_half_width:
                    left_clear = min(
                        left_clear, y_base - self.robot_half_width)
                elif y_base < -self.robot_half_width:
                    right_clear = min(
                        right_clear, -y_base - self.robot_half_width)
        model = self.fit_front_wall(points)
        now = time.time()
        with self.lock:
            if model is not None:
                previous = self.center_wall_model
                if previous is not None:
                    alpha = 0.32
                    angle_delta = math.atan2(
                        math.sin(model["heading_error"] -
                                 previous["heading_error"]),
                        math.cos(model["heading_error"] -
                                 previous["heading_error"]))
                    model["heading_error"] = (
                        previous["heading_error"] + alpha * angle_delta)
                    model["distance"] = (
                        previous["distance"] + alpha *
                        (model["distance"] - previous["distance"]))
                self.center_wall_model = model
                self.center_scan_time = now
            self.left_swept_clear = left_clear
            self.right_swept_clear = right_clear

    def fit_front_wall(self, points):
        if len(points) < self.wall_min_inliers:
            return None
        data = np.asarray(points, dtype=np.float64)
        count = len(data)
        stride = max(1, count // 36)
        candidates = range(0, count, stride)
        best_indices = None
        best_score = -1.0
        for first in candidates:
            for second in candidates:
                if second <= first + max(2, stride):
                    continue
                delta = data[second] - data[first]
                length = float(np.linalg.norm(delta))
                if length < 0.55 * self.wall_min_span:
                    continue
                normal = np.array([-delta[1], delta[0]]) / length
                residual = np.abs((data - data[first]).dot(normal))
                indices = np.flatnonzero(residual <= self.wall_inlier)
                if len(indices) < self.wall_min_inliers:
                    continue
                tangent = delta / length
                projection = data[indices].dot(tangent)
                span = float(projection.max() - projection.min())
                if span < self.wall_min_span:
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
        heading = math.atan2(float(normal[1]), float(normal[0]))
        distance = float(normal.dot(centroid))
        tangent = np.array([-normal[1], normal[0]])
        projection = inliers.dot(tangent)
        span = float(projection.max() - projection.min())
        residual = np.abs((inliers - centroid).dot(normal))
        if (distance <= 0.0 or
                abs(heading) > self.wall_heading_abort):
            return None
        return {
            "distance": distance,
            "heading_error": heading,
            "inliers": int(len(inliers)),
            "span": span,
            "rms": float(math.sqrt(np.mean(residual * residual))),
        }

    def set_parking_mode(self, enabled):
        self.parking_mode = bool(enabled)
        self.parking_mode_pub.publish(Bool(data=self.parking_mode))

    def publish_center_command(self, x=0.0, y=0.0, wz=0.0):
        command = Twist()
        command.linear.x = float(x)
        command.linear.y = float(y)
        command.angular.z = float(wz)
        self.set_parking_mode(True)
        self.cmd_pub.publish(command)

    def snapshot_center(self):
        with self.lock:
            wall = (None if self.center_wall_model is None else
                    dict(self.center_wall_model))
            return {
                "wall": wall,
                "scan_time": self.center_scan_time,
                "front_min": self.front_min,
                "left_clear": self.left_swept_clear,
                "right_clear": self.right_swept_clear,
                "odom_xy": self.odom_pose,
                "odom_yaw": self.odom_yaw,
                "ocr": (None if self.latest_ocr is None else
                        dict(self.latest_ocr)),
            }

    def target_sign_error(self):
        snapshot = self.snapshot_center()
        payload = snapshot["ocr"]
        if (not payload or not payload.get("stable") or
                str(payload.get("label", "")) != self.target_warehouse or
                not payload.get("bbox")):
            return None
        stamp = float(payload.get("stamp", 0.0))
        if time.time() - stamp > 1.2:
            return None
        bbox = payload["bbox"]
        width = float(payload.get("image_width", 0.0))
        if len(bbox) != 4 or width <= 1.0:
            return None
        raw_center = 0.5 * (float(bbox[0]) + float(bbox[2]))
        if self.center_ocr_stamp != stamp:
            if self.center_ocr_value is None:
                self.center_ocr_value = raw_center
            else:
                alpha = clamp(self.sign_ema_alpha, 0.05, 1.0)
                self.center_ocr_value += alpha * (
                    raw_center - self.center_ocr_value)
            self.center_ocr_stamp = stamp
        target = width * self.optical_center_ratio
        return {
            "error_px": self.center_ocr_value - target,
            "center_px": self.center_ocr_value,
            "target_px": target,
            "width": width,
            "label": payload.get("label"),
        }

    @staticmethod
    def distance_between(first, second):
        if first is None or second is None:
            return float("inf")
        return math.hypot(second[0] - first[0], second[1] - first[1])

    def align_real_wall(self, state, tolerance=None):
        tolerance = self.heading_tolerance if tolerance is None else tolerance
        self.publish_state(state, tolerance_deg=math.degrees(tolerance))
        stable = 0
        best = float("inf")
        last_progress = time.time()
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < self.heading_timeout:
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if (wall is None or
                    time.time() - snapshot["scan_time"] > 1.0):
                stable = 0
                self.publish_center_command()
                rospy.logwarn_throttle(
                    0.8, "CENTERLINE_WALL_ALIGN waiting for wall fit")
                rate.sleep()
                continue
            error = wall["heading_error"]
            abs_error = abs(error)
            if abs_error + math.radians(0.35) < best:
                best = abs_error
                last_progress = time.time()
            if abs_error <= tolerance:
                stable += 1
                command_wz = 0.0
            else:
                stable = 0
                minimum = (self.heading_stuck_wz
                           if time.time() - last_progress > 1.6
                           else self.heading_min_wz)
                command_wz = self.heading_kp * error
                if abs(command_wz) < minimum:
                    command_wz = math.copysign(minimum, error)
                command_wz = clamp(
                    command_wz, -self.heading_max_wz,
                    self.heading_max_wz)
            self.publish_center_command(wz=command_wz)
            rospy.logwarn_throttle(
                0.30,
                "CENTERLINE_WALL_ALIGN distance=%.3f error=%.2fdeg "
                "stable=%d/%d cmd_w=%.3f span=%.3f",
                wall["distance"], math.degrees(error), stable,
                self.heading_stable_frames, command_wz, wall["span"])
            if stable >= self.heading_stable_frames:
                self.publish_zero(12)
                rospy.logwarn(
                    "CENTERLINE_WALL_ALIGNED error=%.2fdeg elapsed=%.2fs",
                    math.degrees(error), time.time() - start)
                return True
            rate.sleep()
        self.publish_zero(25)
        rospy.logerr(
            "CENTERLINE_WALL_ALIGN_FAILED best=%.2fdeg",
            math.degrees(best) if math.isfinite(best) else 999.0)
        return False

    def guarded_lateral(self, requested, snapshot):
        clearance = (snapshot["left_clear"] if requested > 0.0
                     else snapshot["right_clear"])
        if clearance <= self.lateral_hard_clearance:
            return 0.0, clearance, "blocked"
        if (not math.isfinite(clearance) or
                clearance >= self.lateral_slow_clearance):
            return requested, clearance, "clear"
        ratio = clamp(
            (clearance - self.lateral_hard_clearance) /
            max(0.01, self.lateral_slow_clearance -
                self.lateral_hard_clearance), 0.22, 1.0)
        return requested * ratio, clearance, "slow"

    def center_on_target_sign(self):
        self.publish_state("CENTERLINE_LATERAL_ALIGNMENT",
                           warehouse=self.target_warehouse)
        start_xy = self.snapshot_center()["odom_xy"]
        stable = 0
        obstacle_since = None
        stale_since = None
        start = time.time()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() - start < self.lateral_timeout:
            snapshot = self.snapshot_center()
            moved = self.distance_between(start_xy, snapshot["odom_xy"])
            if moved > self.lateral_limit:
                rospy.logerr(
                    "CENTERLINE_LATERAL_LIMIT moved=%.3fm limit=%.3fm",
                    moved, self.lateral_limit)
                self.publish_zero(25)
                return False
            wall = snapshot["wall"]
            if (wall is None or
                    time.time() - snapshot["scan_time"] > 1.0):
                self.publish_center_command()
                stable = 0
                rate.sleep()
                continue
            if abs(wall["heading_error"]) > self.lateral_realign:
                self.publish_zero(8)
                if not self.align_real_wall(
                        "CENTERLINE_LATERAL_HEADING_RECOVERY",
                        tolerance=math.radians(3.5)):
                    return False
                stable = 0
                continue
            sign = self.target_sign_error()
            if sign is None:
                self.publish_center_command()
                stable = 0
                if stale_since is None:
                    stale_since = time.time()
                if time.time() - stale_since > 3.0:
                    rospy.logerr("CENTERLINE_TARGET_OCR_LOST")
                    return False
                rate.sleep()
                continue
            stale_since = None
            error = sign["error_px"]
            if abs(error) <= self.sign_tolerance_px:
                stable += 1
                obstacle_since = None
                command_y = 0.0
                clearance = float("inf")
                guard = "centered"
            else:
                stable = 0
                normalized = error / max(1.0, 0.5 * sign["width"])
                requested = self.lateral_command_sign * self.lateral_kp * normalized
                minimum = (self.lateral_near_min_speed
                           if abs(error) <= self.lateral_near_band_px
                           else self.lateral_min_speed)
                if abs(requested) < minimum:
                    requested = math.copysign(minimum, requested)
                requested = clamp(
                    requested, -self.lateral_max_speed,
                    self.lateral_max_speed)
                command_y, clearance, guard = self.guarded_lateral(
                    requested, snapshot)
                if guard == "blocked":
                    if obstacle_since is None:
                        obstacle_since = time.time()
                    if time.time() - obstacle_since > self.obstacle_wait_timeout:
                        self.publish_zero(25)
                        rospy.logerr(
                            "CENTERLINE_LATERAL_OBSTACLE clear=%.3fm",
                            clearance)
                        return False
                else:
                    obstacle_since = None
            self.publish_center_command(y=command_y)
            rospy.logwarn_throttle(
                0.30,
                "CENTERLINE_SIGN_ALIGN u=%.1f target=%.1f error=%.1fpx "
                "stable=%d/%d clear=%.3f guard=%s cmd_y=%.3f",
                sign["center_px"], sign["target_px"], error, stable,
                self.sign_stable_frames, clearance, guard, command_y)
            if stable >= self.sign_stable_frames:
                self.publish_zero(15)
                self.publish_state(
                    "CENTERLINE_SIGN_ALIGNED", error_px=error,
                    lateral_motion=moved)
                return True
            rate.sleep()
        self.publish_zero(25)
        rospy.logerr("CENTERLINE_SIGN_ALIGN_TIMEOUT")
        return False

    def approach_centered_wall(self):
        self.publish_state(
            "CENTERLINE_WALL_APPROACH",
            wall_target=self.final_wall_distance,
            nose_gap_target=self.final_wall_distance - 0.061)
        start_xy = self.snapshot_center()["odom_xy"]
        progress_xy = start_xy
        last_progress = time.time()
        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < self.approach_timeout:
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if (wall is None or
                    time.time() - snapshot["scan_time"] > 1.0):
                self.publish_center_command()
                stable = 0
                rate.sleep()
                continue
            travelled = self.distance_between(start_xy, snapshot["odom_xy"])
            if self.distance_between(progress_xy, snapshot["odom_xy"]) >= 0.012:
                progress_xy = snapshot["odom_xy"]
                last_progress = time.time()
            distance_error = wall["distance"] - self.final_wall_distance
            heading_error = wall["heading_error"]
            if (snapshot["front_min"] < self.front_emergency or
                    wall["distance"] < self.final_wall_distance -
                    self.final_fail_tolerance):
                self.publish_zero(30)
                rospy.logerr(
                    "CENTERLINE_FRONT_EMERGENCY front=%.3f wall=%.3f",
                    snapshot["front_min"], wall["distance"])
                return False
            if travelled > self.approach_max_travel:
                self.publish_zero(30)
                rospy.logerr(
                    "CENTERLINE_APPROACH_TRAVEL_LIMIT %.3fm", travelled)
                return False
            if abs(heading_error) > self.approach_heading_limit:
                self.publish_zero(8)
                if wall["distance"] <= 0.30:
                    rospy.logerr(
                        "CENTERLINE_CLOSE_HEADING_REJECTED wall=%.3f error=%.2fdeg",
                        wall["distance"], math.degrees(heading_error))
                    return False
                if not self.align_real_wall(
                        "CENTERLINE_APPROACH_HEADING_RECOVERY"):
                    return False
                stable = 0
                progress_xy = self.snapshot_center()["odom_xy"]
                last_progress = time.time()
                continue
            if abs(distance_error) <= self.final_wall_tolerance:
                stable += 1
                command_x = 0.0
                command_wz = 0.0
            elif distance_error < 0.0:
                stable = 0
                command_x = 0.0
                command_wz = 0.0
            else:
                stable = 0
                ratio = clamp(
                    distance_error / self.approach_slow_error,
                    0.0, 1.0)
                command_x = (
                    self.approach_slow_speed + ratio *
                    (self.approach_fast_speed -
                     self.approach_slow_speed))
                command_wz = clamp(
                    self.heading_kp * heading_error, -0.045, 0.045)
            self.publish_center_command(x=command_x, wz=command_wz)
            rospy.logwarn_throttle(
                0.30,
                "CENTERLINE_APPROACH wall=%.3f target=%.3f error=%.3f "
                "heading=%.2fdeg travel=%.3f stable=%d/%d cmd=(%.3f,%.3f)",
                wall["distance"], self.final_wall_distance,
                distance_error, math.degrees(heading_error), travelled,
                stable, self.final_stable_frames, command_x, command_wz)
            if stable >= self.final_stable_frames:
                self.publish_zero(30)
                self.publish_state(
                    "CENTERLINE_PARKING_SUCCESS",
                    wall_distance=wall["distance"],
                    nose_gap_estimate=wall["distance"] - 0.061,
                    heading_error_deg=math.degrees(heading_error),
                    forward_motion=travelled)
                rospy.logwarn(
                    "CENTERLINE_PARKED wall=%.3fm nose_gap=%.3fm heading=%.2fdeg",
                    wall["distance"], wall["distance"] - 0.061,
                    math.degrees(heading_error))
                return True
            if (command_x > 0.0 and
                    time.time() - last_progress > 3.0):
                self.publish_zero(30)
                rospy.logerr("CENTERLINE_APPROACH_NO_PROGRESS")
                return False
            rate.sleep()
        self.publish_zero(30)
        rospy.logerr("CENTERLINE_APPROACH_TIMEOUT")
        return False

    def approach_target_wall_with_navigation(self):
        # Keep the target sign in front, but do not generate a map-coordinate
        # wall goal. The local wall-normal controller owns the final approach.
        self.move_client.cancel_all_goals()
        self.publish_zero(8)
        self.publish_state(
            "CENTERLINE_PARKING_HANDOFF",
            warehouse=self.target_warehouse)
        self.ocr_control("reset")
        self.ocr_control("enable")
        rospy.sleep(0.8)
        self.align_to_target_sign(timeout_s=6.0)
        self.publish_zero(8)
        return True

    def acquire_square(self, timeout_s=8.0):
        # Compatibility handoff for the inherited mission state machine. No
        # white-frame detector is called in this manager.
        self.publish_state("WHITE_BOX_DETECTION_BYPASSED_CENTERLINE_MODE")
        return {"found": True, "mode": "ocr_centerline"}

    def park_inside_square(self):
        self.move_client.cancel_all_goals()
        self.publish_zero(10)
        self.center_ocr_value = None
        self.center_ocr_stamp = None
        self.set_parking_mode(True)
        self.publish_state(
            "CENTERLINE_PARKING_START",
            warehouse=self.target_warehouse)
        success = False
        try:
            if not self.align_real_wall("CENTERLINE_INITIAL_WALL_ALIGNMENT"):
                return False
            if not self.center_on_target_sign():
                return False
            if not self.align_real_wall("CENTERLINE_FINAL_WALL_ALIGNMENT"):
                return False
            if not self.approach_centered_wall():
                return False
            success = True
            return True
        finally:
            self.publish_zero(30)
            self.set_parking_mode(False)
            rospy.logwarn(
                "CENTERLINE_PARKING_END success=%s", str(success))

    def shutdown(self):
        try:
            self.set_parking_mode(False)
        except Exception:
            pass
        super(CenterlineFactoryDeliveryManager, self).shutdown()


if __name__ == "__main__":
    CenterlineFactoryDeliveryManager().run()
