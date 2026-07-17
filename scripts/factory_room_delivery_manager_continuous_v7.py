#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fast single-pass parking based on the proven center-only test sequence."""

import math
import time

import rospy

from factory_room_delivery_manager_continuous_v2 import clamp
from factory_room_delivery_manager_continuous_v6 import (
    OrientedWallHandoffParkingManager,
)
from factory_room_direct_center_core_v7 import (
    direct_lateral_decision,
    progress_state,
)


class DirectCenterlineParkingManager(OrientedWallHandoffParkingManager):
    def __init__(self):
        # ROS subscribers are created by the parent constructor.  Seed every
        # value used by parking callbacks before those subscribers can fire.
        self.direct_tolerance_px = 20.0
        self.direct_stable_frames = 3
        self.direct_timeout = 10.0
        self.direct_min_effective_speed = 0.026
        self.direct_narrow_speed_cap = 0.032
        self.direct_progress_px = 2.0
        self.direct_boost_after = 1.2
        self.direct_fail_after = 3.6
        self.direct_boost_speed = 0.036
        self.direct_heading_abort = math.radians(12.0)
        self.direct_ocr_stale_timeout = 1.6
        self.direct_blocked_accept_lateral = 0.075
        self.direct_blocked_accept_max_px = 80.0
        self.direct_clearance_recoveries = 1
        super(DirectCenterlineParkingManager, self).__init__()
        self.direct_tolerance_px = float(rospy.get_param(
            "~parking_direct_tolerance_px", 20.0))
        self.direct_stable_frames = int(rospy.get_param(
            "~parking_direct_stable_frames", 3))
        self.direct_timeout = float(rospy.get_param(
            "~parking_direct_timeout_s", 10.0))
        self.direct_min_effective_speed = float(rospy.get_param(
            "~parking_direct_min_effective_speed_mps", 0.026))
        self.direct_narrow_speed_cap = float(rospy.get_param(
            "~parking_direct_narrow_speed_cap_mps", 0.032))
        self.direct_progress_px = float(rospy.get_param(
            "~parking_direct_progress_px", 2.0))
        self.direct_boost_after = float(rospy.get_param(
            "~parking_direct_boost_after_s", 1.2))
        self.direct_fail_after = float(rospy.get_param(
            "~parking_direct_fail_after_s", 3.6))
        self.direct_boost_speed = float(rospy.get_param(
            "~parking_direct_boost_speed_mps", 0.036))
        self.direct_heading_abort = math.radians(float(rospy.get_param(
            "~parking_direct_heading_abort_deg", 12.0)))
        self.direct_ocr_stale_timeout = float(rospy.get_param(
            "~parking_direct_ocr_stale_timeout_s", 1.6))
        self.direct_blocked_accept_lateral = float(rospy.get_param(
            "~parking_direct_blocked_accept_lateral_m", 0.075))
        self.direct_blocked_accept_max_px = float(rospy.get_param(
            "~parking_direct_blocked_accept_max_px", 80.0))
        self.direct_clearance_recoveries = int(rospy.get_param(
            "~parking_direct_clearance_recoveries", 1))
        rospy.logwarn(
            "ROOM_DIRECT_CENTERLINE_V7_READY tolerance=%.1fpx timeout=%.1fs "
            "effective_min=%.3fmps clearance_recoveries=%d",
            self.direct_tolerance_px, self.direct_timeout,
            self.direct_min_effective_speed,
            self.direct_clearance_recoveries)

    def _blocked_center_offset(self, error_px, image_width, wall_distance):
        half_width = max(1.0, 0.5 * float(image_width))
        normalized = min(1.0, abs(float(error_px)) / half_width)
        bearing = normalized * 0.5 * self.orthogonal_camera_hfov
        return max(0.0, float(wall_distance)) * math.tan(bearing)

    def _center_on_target_sign_direct(self, initial_sign):
        if initial_sign is None:
            return False
        start_xy = self.snapshot_center()["odom_xy"]
        stable = 0
        stale_since = None
        blocked_since = None
        clearance_recoveries = 0
        start = time.time()
        best_error = abs(initial_sign["error_px"])
        last_progress = start
        rate = rospy.Rate(20)

        while (not rospy.is_shutdown() and
               time.time() - start < self.direct_timeout):
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if (wall is None or
                    time.time() - snapshot["scan_time"] > 1.0):
                self.publish_center_command()
                rate.sleep()
                continue
            if abs(wall["heading_error"]) > self.direct_heading_abort:
                self.publish_zero(8)
                rospy.logerr(
                    "CENTERLINE_V7_HEADING_ABORT error=%.2fdeg",
                    math.degrees(wall["heading_error"]))
                return False

            sign = self.target_sign_error()
            if sign is None:
                self.publish_center_command()
                stable = 0
                if stale_since is None:
                    stale_since = time.time()
                if time.time() - stale_since > self.direct_ocr_stale_timeout:
                    rospy.logerr("CENTERLINE_V7_OCR_STALE")
                    return False
                rate.sleep()
                continue
            stale_since = None

            error = sign["error_px"]
            normalized = error / max(1.0, 0.5 * sign["width"])
            requested = self.lateral_command_sign * self.lateral_kp * normalized
            requested = clamp(
                requested, -self.lateral_max_speed, self.lateral_max_speed)
            direction = 1.0 if requested >= 0.0 else -1.0
            clearance = self._directional_clearance(snapshot, direction)
            decision = direct_lateral_decision(
                error, self.direct_tolerance_px, requested, clearance,
                self.lateral_hard_clearance, self.lateral_slow_clearance,
                self.direct_min_effective_speed,
                self.direct_narrow_speed_cap)

            now = time.time()
            progress = progress_state(
                best_error, error, last_progress, now,
                self.direct_progress_px, self.direct_boost_after,
                self.direct_fail_after)
            best_error = progress["best_error"]
            last_progress = progress["last_progress_time"]
            if progress["failed"] and decision["action"] == "move":
                self.publish_zero(12)
                rospy.logerr(
                    "CENTERLINE_V7_NO_PIXEL_PROGRESS error=%.1fpx "
                    "best=%.1fpx",
                    error, best_error)
                return False

            if decision["action"] == "centered":
                stable += 1
                blocked_since = None
                command_y = 0.0
            elif decision["action"] == "blocked":
                stable = 0
                command_y = 0.0
                if blocked_since is None:
                    blocked_since = now
                if now - blocked_since > 0.8:
                    estimated_offset = self._blocked_center_offset(
                        error, sign["width"], wall["distance"])
                    if (clearance >= self.straight_side_min and
                            abs(error) <= self.direct_blocked_accept_max_px and
                            estimated_offset <=
                            self.direct_blocked_accept_lateral):
                        self.publish_zero(8)
                        self.publish_state(
                            "CENTERLINE_V7_BLOCKED_SAFE_ACCEPT",
                            error_px=error, clearance=clearance,
                            estimated_offset_m=estimated_offset)
                        rospy.logwarn(
                            "CENTERLINE_V7_BLOCKED_SAFE_ACCEPT error=%.1fpx "
                            "estimated_offset=%.3fm clear=%.3fm "
                            "forced_lateral_avoided=true",
                            error, estimated_offset, clearance)
                        return True
                    direction = 1.0 if requested >= 0.0 else -1.0
                    if (clearance_recoveries <
                            max(0, self.direct_clearance_recoveries) and
                            self._advance_for_lateral_clearance(direction)):
                        clearance_recoveries += 1
                        if not self.align_real_wall(
                                "CENTERLINE_V7_AFTER_FORWARD_CLEARANCE",
                                tolerance=math.radians(3.0)):
                            return False
                        fresh = self.wait_for_fresh_target_sign(
                            "CENTERLINE_V7_FRESH_OCR_AFTER_CLEARANCE")
                        if fresh is None:
                            return False
                        start_xy = self.snapshot_center()["odom_xy"]
                        start = time.time()
                        best_error = abs(fresh["error_px"])
                        last_progress = start
                        blocked_since = None
                        stable = 0
                        rospy.logwarn(
                            "CENTERLINE_V7_CLEARANCE_RECOVERY_COMPLETE "
                            "attempt=%d",
                            clearance_recoveries)
                        continue
                    self.publish_zero(12)
                    rospy.logerr(
                        "CENTERLINE_V7_LATERAL_BLOCKED clear=%.3f "
                        "estimated_offset=%.3f",
                        clearance, estimated_offset)
                    return False
            else:
                stable = 0
                blocked_since = None
                command_y = decision["command"]
                if progress["boost"]:
                    command_y = math.copysign(
                        max(abs(command_y), self.direct_boost_speed),
                        command_y)
                    if decision["guard"] == "narrow":
                        command_y = math.copysign(
                            min(abs(command_y),
                                max(self.direct_narrow_speed_cap,
                                    self.direct_min_effective_speed)),
                            command_y)

            moved = self.distance_between(
                start_xy, snapshot["odom_xy"])
            if moved > self.lateral_limit:
                self.publish_zero(12)
                rospy.logerr(
                    "CENTERLINE_V7_TRAVEL_LIMIT moved=%.3f", moved)
                return False
            self.publish_center_command(y=command_y)
            rospy.logwarn_throttle(
                0.20,
                "CENTERLINE_V7_DIRECT error=%.1fpx stable=%d/%d "
                "clear=%.3f guard=%s cmd_y=%.3f moved=%.3f "
                "progress_age=%.2f",
                error, stable, self.direct_stable_frames,
                clearance, decision["guard"], command_y, moved,
                now - last_progress)
            if stable >= self.direct_stable_frames:
                self.publish_zero(12)
                self.publish_state(
                    "CENTERLINE_V7_DIRECT_ALIGNED",
                    error_px=error, lateral_motion=moved)
                return True
            rate.sleep()

        self.publish_zero(15)
        rospy.logerr("CENTERLINE_V7_DIRECT_TIMEOUT")
        return False

    def park_inside_square(self):
        self.move_client.cancel_all_goals()
        self.publish_zero(5)
        self.set_parking_mode(True)
        self.publish_state(
            "CENTERLINE_V7_DIRECT_PARKING_START",
            warehouse=self.target_warehouse,
            sequence="align-center-final_align-approach")
        success = False
        try:
            if not self.align_real_wall(
                    "CENTERLINE_V7_INITIAL_WALL_ALIGNMENT",
                    tolerance=math.radians(2.8)):
                return False
            sign = self.wait_for_fresh_target_sign(
                "CENTERLINE_V7_INITIAL_FRESH_OCR")
            if not self._center_on_target_sign_direct(sign):
                return False
            if not self.align_real_wall(
                    "CENTERLINE_V7_FINAL_WALL_ALIGNMENT",
                    tolerance=math.radians(2.8)):
                return False
            if not self.approach_centered_wall():
                return False
            success = True
            return True
        finally:
            self.publish_zero(15)
            self.set_parking_mode(False)
            rospy.logwarn(
                "CENTERLINE_V7_DIRECT_PARKING_END success=%s",
                str(success))


if __name__ == "__main__":
    DirectCenterlineParkingManager().run()
