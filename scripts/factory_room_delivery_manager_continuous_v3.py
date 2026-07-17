#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conflict-free room parking with fresh OCR after every heading change."""

import math
import time

import rospy

from factory_room_delivery_manager_continuous_v2 import (
    ReliableContinuousSearchParkingManager,
    clamp,
)
from factory_room_parking_handoff_core_v3 import (
    is_fresh_target_payload,
    precenter_action,
)


class ConflictFreeContinuousParkingManager(
        ReliableContinuousSearchParkingManager):
    def __init__(self):
        super(ConflictFreeContinuousParkingManager, self).__init__()
        self.fresh_ocr_timeout = float(rospy.get_param(
            "~parking_fresh_ocr_timeout_s", 2.2))
        self.fresh_ocr_frames = int(rospy.get_param(
            "~parking_fresh_ocr_frames", 3))
        self.center_attempts = int(rospy.get_param(
            "~parking_center_attempts", 2))
        self.final_verify_cycles = int(rospy.get_param(
            "~parking_final_verify_cycles", 2))
        self.clearance_advance_speed = float(rospy.get_param(
            "~parking_clearance_advance_speed_mps", 0.040))
        self.clearance_advance_distance = float(rospy.get_param(
            "~parking_clearance_advance_distance_m", 0.080))
        self.clearance_advance_timeout = float(rospy.get_param(
            "~parking_clearance_advance_timeout_s", 3.0))
        self.clearance_wall_min = float(rospy.get_param(
            "~parking_clearance_wall_min_m", 0.40))
        self.clearance_front_min = float(rospy.get_param(
            "~parking_clearance_front_min_m", 0.32))
        self.clearance_side_release = float(rospy.get_param(
            "~parking_clearance_side_release_m", 0.11))
        self.straight_side_min = float(rospy.get_param(
            "~parking_straight_side_min_m", 0.045))
        rospy.logwarn(
            "CONFLICT_FREE_PARKING_V3_READY fresh=%d/%.1fs attempts=%d "
            "legacy_precenter_retreat=false single_cmd_owner=true",
            self.fresh_ocr_frames, self.fresh_ocr_timeout,
            self.center_attempts)

    def _clear_ocr_geometry(self):
        with self.lock:
            self.latest_ocr = None
            self.center_ocr_value = None
            self.center_ocr_stamp = None

    def wait_for_fresh_target_sign(self, state):
        started = time.time()
        self._clear_ocr_geometry()
        self.ocr_control("reset")
        self.ocr_control("enable")
        self.publish_state(
            state, timeout_s=self.fresh_ocr_timeout,
            required_frames=self.fresh_ocr_frames)
        seen_stamps = set()
        stable = 0
        latest_sign = None
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.fresh_ocr_timeout):
            with self.lock:
                payload = (None if self.latest_ocr is None
                           else dict(self.latest_ocr))
            if is_fresh_target_payload(
                    payload, self.target_warehouse, started):
                stamp = float(payload.get("stamp", 0.0))
                if stamp not in seen_stamps:
                    seen_stamps.add(stamp)
                    latest_sign = self.target_sign_error()
                    stable = stable + 1 if latest_sign is not None else 0
                    if stable >= max(1, self.fresh_ocr_frames):
                        rospy.logwarn(
                            "PARKING_FRESH_OCR_READY state=%s frames=%d "
                            "error=%.1fpx",
                            state, stable, latest_sign["error_px"])
                        return latest_sign
            self.publish_center_command()
            rospy.logwarn_throttle(
                0.40,
                "PARKING_FRESH_OCR_WAIT state=%s frames=%d/%d",
                state, stable, self.fresh_ocr_frames)
            rate.sleep()
        self.publish_zero(8)
        rospy.logwarn(
            "PARKING_FRESH_OCR_TIMEOUT state=%s frames=%d/%d",
            state, stable, self.fresh_ocr_frames)
        return None

    def _advance_for_lateral_clearance(self, direction):
        snapshot = self.snapshot_center()
        wall = snapshot["wall"]
        if wall is None:
            return False
        clearance = self._directional_clearance(snapshot, direction)
        can_advance = (
            wall["distance"] > self.clearance_wall_min and
            snapshot["front_min"] > self.clearance_front_min and
            clearance >= self.straight_side_min)
        action = precenter_action(
            self.sign_tolerance_px + 1.0, self.sign_tolerance_px,
            clearance, self.lateral_hard_clearance,
            self.straight_side_min, can_advance)
        if action != "advance":
            return False

        self.publish_state(
            "CENTERLINE_FORWARD_CLEARANCE",
            side="left" if direction > 0.0 else "right",
            clearance=clearance,
            max_distance=self.clearance_advance_distance)
        start_xy = snapshot["odom_xy"]
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.clearance_advance_timeout):
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if wall is None:
                self.publish_center_command()
                rate.sleep()
                continue
            clearance = self._directional_clearance(snapshot, direction)
            moved = self.distance_between(start_xy, snapshot["odom_xy"])
            if clearance >= self.clearance_side_release:
                self.publish_zero(8)
                rospy.logwarn(
                    "CENTERLINE_FORWARD_CLEARANCE_READY clear=%.3f moved=%.3f",
                    clearance, moved)
                return True
            if (moved >= self.clearance_advance_distance or
                    wall["distance"] <= self.clearance_wall_min or
                    snapshot["front_min"] <= self.clearance_front_min or
                    clearance < self.straight_side_min or
                    abs(wall["heading_error"]) > self.final_heading_tolerance):
                self.publish_zero(12)
                rospy.logwarn(
                    "CENTERLINE_FORWARD_CLEARANCE_STOP clear=%.3f moved=%.3f "
                    "wall=%.3f front=%.3f heading=%.2fdeg",
                    clearance, moved, wall["distance"],
                    snapshot["front_min"],
                    math.degrees(wall["heading_error"]))
                return clearance > self.lateral_hard_clearance
            command_wz = clamp(
                self.heading_kp * wall["heading_error"], -0.018, 0.018)
            self.publish_center_command(
                x=self.clearance_advance_speed, wz=command_wz)
            rospy.logwarn_throttle(
                0.25,
                "CENTERLINE_FORWARD_CLEARANCE moving=%.3f/%.3f "
                "clear=%.3f wall=%.3f cmd=(%.3f,%.3f)",
                moved, self.clearance_advance_distance, clearance,
                wall["distance"], self.clearance_advance_speed, command_wz)
            rate.sleep()
        self.publish_zero(12)
        return False

    def center_on_target_sign_conflict_free(self, initial_sign=None):
        sign = initial_sign
        if sign is None:
            sign = self.wait_for_fresh_target_sign(
                "CENTERLINE_FRESH_OCR_BEFORE_CENTER")
        if sign is None:
            return False

        start_xy = self.snapshot_center()["odom_xy"]
        stable = 0
        stale_since = None
        obstacle_since = None
        clearance_recoveries = 0
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.lateral_timeout):
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if wall is None:
                self.publish_center_command()
                stable = 0
                rate.sleep()
                continue
            if abs(wall["heading_error"]) > self.lateral_realign:
                self.publish_zero(8)
                if not self.align_real_wall(
                        "CENTERLINE_V3_LATERAL_REALIGN",
                        tolerance=math.radians(3.0)):
                    return False
                sign = self.wait_for_fresh_target_sign(
                    "CENTERLINE_V3_FRESH_OCR_AFTER_REALIGN")
                if sign is None:
                    return False
                stable = 0
                continue

            sign = self.target_sign_error()
            if sign is None:
                self.publish_center_command()
                stable = 0
                if stale_since is None:
                    stale_since = time.time()
                if time.time() - stale_since > 2.2:
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
                    if (time.time() - obstacle_since >
                            self.obstacle_wait_timeout):
                        direction = math.copysign(1.0, requested)
                        if (clearance_recoveries >= 1 or
                                not self._advance_for_lateral_clearance(
                                    direction)):
                            return False
                        clearance_recoveries += 1
                        if not self.align_real_wall(
                                "CENTERLINE_V3_AFTER_FORWARD_CLEARANCE",
                                tolerance=math.radians(3.0)):
                            return False
                        sign = self.wait_for_fresh_target_sign(
                            "CENTERLINE_V3_FRESH_OCR_AFTER_CLEARANCE")
                        if sign is None:
                            return False
                        obstacle_since = None
                        stable = 0
                        continue
                else:
                    obstacle_since = None

            self.publish_center_command(y=command_y)
            rospy.logwarn_throttle(
                0.25,
                "CENTERLINE_V3_SIGN_ALIGN error=%.1fpx stable=%d/%d "
                "clear=%.3f guard=%s cmd_y=%.3f",
                error, stable, self.sign_stable_frames,
                clearance, guard, command_y)
            if stable >= self.sign_stable_frames:
                self.publish_zero(12)
                self.publish_state(
                    "CENTERLINE_SIGN_ALIGNED",
                    error_px=error,
                    lateral_motion=self.distance_between(
                        start_xy, self.snapshot_center()["odom_xy"]))
                return True
            rate.sleep()
        self.publish_zero(20)
        return False

    def park_inside_square(self):
        self.move_client.cancel_all_goals()
        self.publish_zero(5)
        self.set_parking_mode(True)
        self.publish_state(
            "CONFLICT_FREE_PARKING_START",
            warehouse=self.target_warehouse,
            attempts=self.center_attempts)
        success = False
        try:
            centered = False
            for attempt in range(1, max(1, self.center_attempts) + 1):
                self.publish_state(
                    "CONFLICT_FREE_CENTER_ATTEMPT", attempt=attempt)
                if not self.align_real_wall(
                        "CENTERLINE_V3_INITIAL_WALL_ALIGNMENT",
                        tolerance=math.radians(2.8)):
                    continue
                sign = self.wait_for_fresh_target_sign(
                    "CENTERLINE_V3_FRESH_OCR_AFTER_INITIAL_ALIGN")
                if sign is None:
                    continue
                if self.center_on_target_sign_conflict_free(sign):
                    centered = True
                    break
                self.publish_zero(10)
                rospy.logwarn(
                    "CONFLICT_FREE_CENTER_RETRY attempt=%d/%d",
                    attempt, self.center_attempts)
            if not centered:
                return False

            final_verified = False
            for verify in range(1, max(1, self.final_verify_cycles) + 1):
                if not self.align_real_wall(
                        "CENTERLINE_V3_FINAL_WALL_ALIGNMENT",
                        tolerance=math.radians(2.6)):
                    continue
                final_sign = self.wait_for_fresh_target_sign(
                    "CENTERLINE_V3_FINAL_FRESH_OCR")
                if (final_sign is not None and
                        abs(final_sign["error_px"]) <=
                        self.sign_tolerance_px):
                    final_verified = True
                    break
                self.publish_state(
                    "CENTERLINE_V3_FINAL_RECENTER",
                    verify=verify,
                    error_px=(None if final_sign is None
                              else final_sign["error_px"]))
                if not self.center_on_target_sign_conflict_free(final_sign):
                    continue
            if not final_verified:
                return False
            if not self.approach_centered_wall():
                return False
            success = True
            return True
        finally:
            self.publish_zero(15)
            self.set_parking_mode(False)
            rospy.logwarn(
                "CONFLICT_FREE_PARKING_END success=%s", str(success))


if __name__ == "__main__":
    ConflictFreeContinuousParkingManager().run()
