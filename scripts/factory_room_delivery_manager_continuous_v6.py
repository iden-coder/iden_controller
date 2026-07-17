#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parking handoff with oriented wall selection and OCR heading bootstrap."""

import math
import time

import rospy

from factory_room_delivery_manager_continuous_v5 import (
    DeliveryOnlyVoiceContinuousParkingManager,
)
from factory_room_wall_handoff_core_v6 import (
    estimated_sign_heading,
    fit_oriented_front_wall,
    nearest_orthogonal_heading,
    norm_angle,
    ocr_heading_command,
)


class OrientedWallHandoffParkingManager(
        DeliveryOnlyVoiceContinuousParkingManager):
    def __init__(self):
        # The base constructor installs /scan before returning. Defaults must
        # exist because the callback dispatches to this subclass immediately.
        self.oriented_wall_inlier = 0.035
        self.oriented_wall_min_inliers = 12
        self.oriented_wall_min_span = 0.20
        self.oriented_wall_max_heading = math.radians(48.0)
        self.bootstrap_timeout = 2.8
        self.bootstrap_tolerance_px = 45.0
        self.bootstrap_kp = 0.40
        self.bootstrap_min_wz = 0.09
        self.bootstrap_max_wz = 0.24
        self.bootstrap_front_min = 0.28
        self.bootstrap_side_min = 0.08
        self.orthogonal_camera_hfov = math.radians(70.0)
        self.orthogonal_timeout = 6.0
        self.orthogonal_tolerance = math.radians(3.0)
        self.orthogonal_kp = 1.35
        self.orthogonal_min_wz = 0.10
        self.orthogonal_max_wz = 0.28
        self.orthogonal_wall_wait = 1.2
        super(OrientedWallHandoffParkingManager, self).__init__()
        self.oriented_wall_inlier = float(rospy.get_param(
            "~parking_oriented_wall_inlier_m", max(0.032, self.wall_inlier)))
        self.oriented_wall_min_inliers = int(rospy.get_param(
            "~parking_oriented_wall_min_inliers",
            max(10, int(round(0.67 * self.wall_min_inliers)))))
        self.oriented_wall_min_span = float(rospy.get_param(
            "~parking_oriented_wall_min_span_m",
            max(0.20, 0.72 * self.wall_min_span)))
        self.oriented_wall_max_heading = math.radians(float(rospy.get_param(
            "~parking_oriented_wall_max_heading_deg", 48.0)))
        self.bootstrap_timeout = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_timeout_s", 2.8))
        self.bootstrap_tolerance_px = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_tolerance_px", 45.0))
        self.bootstrap_kp = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_kp", 0.40))
        self.bootstrap_min_wz = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_min_rps", 0.09))
        self.bootstrap_max_wz = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_max_rps", 0.24))
        self.bootstrap_front_min = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_front_min_m", 0.28))
        self.bootstrap_side_min = float(rospy.get_param(
            "~parking_ocr_heading_bootstrap_side_min_m", 0.08))
        self.orthogonal_camera_hfov = math.radians(float(rospy.get_param(
            "~parking_orthogonal_camera_hfov_deg", 70.0)))
        self.orthogonal_timeout = float(rospy.get_param(
            "~parking_orthogonal_timeout_s", 6.0))
        self.orthogonal_tolerance = math.radians(float(rospy.get_param(
            "~parking_orthogonal_tolerance_deg", 3.0)))
        self.orthogonal_kp = float(rospy.get_param(
            "~parking_orthogonal_kp", 1.35))
        self.orthogonal_min_wz = float(rospy.get_param(
            "~parking_orthogonal_min_rps", 0.10))
        self.orthogonal_max_wz = float(rospy.get_param(
            "~parking_orthogonal_max_rps", 0.28))
        self.orthogonal_wall_wait = float(rospy.get_param(
            "~parking_orthogonal_wall_wait_s", 1.2))
        rospy.logwarn(
            "ROOM_ORIENTED_WALL_HANDOFF_V6_READY inliers=%d span=%.2f "
            "heading=%.0fdeg bootstrap=%.1fs",
            self.oriented_wall_min_inliers,
            self.oriented_wall_min_span,
            math.degrees(self.oriented_wall_max_heading),
            self.bootstrap_timeout)

    def fit_front_wall(self, points):
        strict = super(OrientedWallHandoffParkingManager,
                       self).fit_front_wall(points)
        if strict is not None:
            strict["method"] = "strict"
            return strict
        model = fit_oriented_front_wall(
            points,
            self.oriented_wall_inlier,
            self.oriented_wall_min_inliers,
            self.oriented_wall_min_span,
            self.oriented_wall_max_heading)
        if model is not None:
            rospy.logwarn_throttle(
                0.8,
                "CENTERLINE_V6_ORIENTED_WALL distance=%.3f "
                "heading=%.1fdeg inliers=%d span=%.3f",
                model["distance"],
                math.degrees(model["heading_error"]),
                model["inliers"], model["span"])
        return model

    @staticmethod
    def _fresh_wall(snapshot):
        return (snapshot["wall"] is not None and
                time.time() - snapshot["scan_time"] <= 0.8)

    @staticmethod
    def _finite_clearance_ok(value, minimum):
        return not math.isfinite(value) or value >= minimum

    def _bootstrap_wall_heading_from_ocr(self, state):
        self.publish_state(
            "CENTERLINE_V6_OCR_HEADING_BOOTSTRAP", source_state=state)
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.bootstrap_timeout):
            snapshot = self.snapshot_center()
            if self._fresh_wall(snapshot):
                self.publish_zero(10)
                wall = snapshot["wall"]
                rospy.logwarn(
                    "CENTERLINE_V6_WALL_ACQUIRED elapsed=%.2fs "
                    "distance=%.3f heading=%.1fdeg method=%s",
                    time.time() - start, wall["distance"],
                    math.degrees(wall["heading_error"]),
                    wall.get("method", "unknown"))
                return "acquired"

            sign = self.target_sign_error()
            if sign is None:
                self.publish_center_command()
                rospy.logwarn_throttle(
                    0.5, "CENTERLINE_V6_BOOTSTRAP waiting fresh target OCR")
                rate.sleep()
                continue

            command_wz = ocr_heading_command(
                sign["error_px"], sign["width"],
                self.bootstrap_tolerance_px, self.bootstrap_kp,
                self.bootstrap_min_wz, self.bootstrap_max_wz)
            front_ok = snapshot["front_min"] >= self.bootstrap_front_min
            sides_ok = (
                self._finite_clearance_ok(
                    snapshot["left_clear"], self.bootstrap_side_min) and
                self._finite_clearance_ok(
                    snapshot["right_clear"], self.bootstrap_side_min))
            if not front_ok or not sides_ok:
                self.publish_zero(8)
                rospy.logerr(
                    "CENTERLINE_V6_BOOTSTRAP_BLOCKED front=%.3f "
                    "left=%.3f right=%.3f",
                    snapshot["front_min"], snapshot["left_clear"],
                    snapshot["right_clear"])
                return "blocked"
            if abs(command_wz) < 1e-6:
                self.publish_center_command()
                rospy.logwarn_throttle(
                    0.5,
                    "CENTERLINE_V6_BOOTSTRAP_OCR_CENTERED error=%.1fpx "
                    "waiting wall model",
                    sign["error_px"])
            else:
                self.publish_center_command(wz=command_wz)
                rospy.logwarn_throttle(
                    0.25,
                    "CENTERLINE_V6_BOOTSTRAP_ROTATE error=%.1fpx "
                    "cmd_wz=%.3f front=%.3f",
                    sign["error_px"], command_wz, snapshot["front_min"])
            rate.sleep()
        self.publish_zero(12)
        rospy.logwarn(
            "CENTERLINE_V6_BOOTSTRAP_EXHAUSTED state=%s; "
            "trying orthogonal fallback", state)
        return "exhausted"

    def _rotation_clearance_ok(self, snapshot):
        return (
            snapshot["front_min"] >= self.bootstrap_front_min and
            self._finite_clearance_ok(
                snapshot["left_clear"], self.bootstrap_side_min) and
            self._finite_clearance_ok(
                snapshot["right_clear"], self.bootstrap_side_min))

    def _wait_for_fresh_wall(self, timeout_s):
        deadline = time.time() + max(0.1, timeout_s)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            snapshot = self.snapshot_center()
            if self._fresh_wall(snapshot):
                return True
            self.publish_center_command()
            rate.sleep()
        return False

    def _rotate_to_orthogonal_fallback(self, state):
        snapshot = self.snapshot_center()
        current_odom_yaw = snapshot["odom_yaw"]
        map_pose = self.snapshot_pose()
        if current_odom_yaw is None:
            rospy.logerr("CENTERLINE_V6_ORTHOGONAL_NO_ODOM state=%s", state)
            return False
        current_map_yaw = (current_odom_yaw if map_pose is None
                           else map_pose[2])
        sign = self.target_sign_error()
        if sign is None:
            estimated_map_heading = current_map_yaw
            source = "odom_only"
        else:
            estimated_map_heading = estimated_sign_heading(
                current_map_yaw, sign["error_px"], sign["width"],
                self.orthogonal_camera_hfov)
            source = "ocr_bearing"
        target_map_yaw = nearest_orthogonal_heading(estimated_map_heading)
        rotation_delta = norm_angle(target_map_yaw - current_map_yaw)
        target_odom_yaw = norm_angle(current_odom_yaw + rotation_delta)
        self.publish_state(
            "CENTERLINE_V6_ORTHOGONAL_FALLBACK",
            source_state=state, source=source,
            current_map_yaw_deg=math.degrees(current_map_yaw),
            estimated_map_yaw_deg=math.degrees(estimated_map_heading),
            target_map_yaw_deg=math.degrees(target_map_yaw),
            target_odom_yaw_deg=math.degrees(target_odom_yaw))
        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.orthogonal_timeout):
            snapshot = self.snapshot_center()
            current_odom_yaw = snapshot["odom_yaw"]
            if current_odom_yaw is None:
                self.publish_zero(8)
                return False
            if not self._rotation_clearance_ok(snapshot):
                self.publish_zero(8)
                rospy.logerr(
                    "CENTERLINE_V6_ORTHOGONAL_BLOCKED front=%.3f "
                    "left=%.3f right=%.3f",
                    snapshot["front_min"], snapshot["left_clear"],
                    snapshot["right_clear"])
                return False
            error = norm_angle(target_odom_yaw - current_odom_yaw)
            if abs(error) <= self.orthogonal_tolerance:
                stable += 1
                command_wz = 0.0
            else:
                stable = 0
                magnitude = max(
                    self.orthogonal_min_wz,
                    min(self.orthogonal_max_wz,
                        self.orthogonal_kp * abs(error)))
                command_wz = math.copysign(magnitude, error)
            self.publish_center_command(wz=command_wz)
            rospy.logwarn_throttle(
                0.25,
                "CENTERLINE_V6_ORTHOGONAL_ROTATE odom=%.1fdeg "
                "target_odom=%.1fdeg error=%.1fdeg cmd_wz=%.3f stable=%d/3",
                math.degrees(current_odom_yaw),
                math.degrees(target_odom_yaw),
                math.degrees(error), command_wz, stable)
            if stable >= 3:
                self.publish_zero(10)
                rospy.logwarn(
                    "CENTERLINE_V6_ORTHOGONAL_REACHED map_target=%.1fdeg "
                    "elapsed=%.2fs",
                    math.degrees(target_map_yaw), time.time() - start)
                return self._wait_for_fresh_wall(self.orthogonal_wall_wait)
            rate.sleep()
        self.publish_zero(12)
        rospy.logerr("CENTERLINE_V6_ORTHOGONAL_TIMEOUT state=%s", state)
        return False

    def align_real_wall(self, state, tolerance=None):
        snapshot = self.snapshot_center()
        if not self._fresh_wall(snapshot):
            result = self._bootstrap_wall_heading_from_ocr(state)
            if result == "blocked":
                return False
            if (result != "acquired" and
                    not self._rotate_to_orthogonal_fallback(state)):
                return False
        return super(OrientedWallHandoffParkingManager, self).align_real_wall(
            state, tolerance=tolerance)


if __name__ == "__main__":
    OrientedWallHandoffParkingManager().run()
