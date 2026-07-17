#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Continuous-search delivery with reliable heartbeat and precise parking."""

import math
import threading
import time

import rospy
from std_msgs.msg import Bool

from factory_room_delivery_manager_continuous_v1 import (
    ContinuousSearchParkingManager,
)
from factory_room_precision_parking_core_v2 import longitudinal_decision


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


class ReliableContinuousSearchParkingManager(ContinuousSearchParkingManager):
    def __init__(self):
        self._parking_heartbeat_lock = threading.RLock()
        self._parking_heartbeat_timer = None
        super(ReliableContinuousSearchParkingManager, self).__init__()

        self.parking_heartbeat_period = float(rospy.get_param(
            "~parking_heartbeat_period_s", 0.20))
        self.precision_tolerance = float(rospy.get_param(
            "~center_precision_tolerance_m", 0.008))
        self.stop_lead = float(rospy.get_param(
            "~center_stop_lead_m", 0.025))
        self.settle_duration = float(rospy.get_param(
            "~center_settle_duration_s", 0.35))
        self.success_hold = float(rospy.get_param(
            "~center_success_hold_s", 0.40))
        self.near_band = float(rospy.get_param(
            "~center_near_band_m", 0.080))
        self.near_min_speed = float(rospy.get_param(
            "~center_near_min_speed_mps", 0.018))
        self.near_max_speed = float(rospy.get_param(
            "~center_near_max_speed_mps", 0.035))
        self.far_full_speed_error = float(rospy.get_param(
            "~center_far_full_speed_error_m", 0.120))
        self.reverse_speed = float(rospy.get_param(
            "~center_reverse_correction_speed_mps", 0.020))
        self.reverse_rear_clear = float(rospy.get_param(
            "~center_reverse_rear_clearance_m", 0.14))
        self.max_reverse_correction = float(rospy.get_param(
            "~center_max_reverse_correction_m", 0.050))
        self.reverse_recover_limit = float(rospy.get_param(
            "~center_reverse_recover_limit_m", 0.030))
        self.final_heading_tolerance = math.radians(float(rospy.get_param(
            "~center_final_heading_tolerance_deg", 2.8)))
        self.near_heading_max_speed = float(rospy.get_param(
            "~center_near_heading_max_rps", 0.018))
        self.heading_retreat_wall = float(rospy.get_param(
            "~center_heading_retreat_wall_m", 0.270))
        self.heading_retreat_speed = float(rospy.get_param(
            "~center_heading_retreat_speed_mps", 0.030))
        self.heading_retreat_max_distance = float(rospy.get_param(
            "~center_heading_retreat_max_distance_m", 0.140))
        self.heading_retreat_timeout = float(rospy.get_param(
            "~center_heading_retreat_timeout_s", 5.0))

        self.final_wall_tolerance = max(
            self.final_wall_tolerance, self.precision_tolerance)
        self._parking_heartbeat_timer = rospy.Timer(
            rospy.Duration(max(0.05, self.parking_heartbeat_period)),
            self._publish_parking_heartbeat)
        rospy.logwarn(
            "PRECISION_PARKING_V2_READY target=%.3fm tolerance=%.3fm "
            "stop_lead=%.3fm near_speed=%.3f..%.3f reverse=%.3f",
            self.final_wall_distance, self.final_wall_tolerance,
            self.stop_lead, self.near_min_speed, self.near_max_speed,
            self.reverse_speed)

    def set_parking_mode(self, enabled):
        with self._parking_heartbeat_lock:
            self.parking_mode = bool(enabled)
            self.parking_mode_pub.publish(Bool(data=self.parking_mode))

    def _publish_parking_heartbeat(self, _event=None):
        with self._parking_heartbeat_lock:
            if self.parking_mode:
                self.parking_mode_pub.publish(Bool(data=True))

    def _recover_close_heading(self, initial_snapshot):
        wall = initial_snapshot["wall"]
        self.publish_zero(8)
        self.publish_state(
            "CENTERLINE_V2_CLOSE_HEADING_RETREAT",
            wall_distance=wall["distance"],
            heading_error_deg=math.degrees(wall["heading_error"]),
            retreat_wall_target=self.heading_retreat_wall)
        start_xy = initial_snapshot["odom_xy"]
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.heading_retreat_timeout):
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if (wall is None or
                    time.time() - snapshot["scan_time"] > 1.0):
                self.publish_center_command()
                rate.sleep()
                continue
            moved = self.distance_between(start_xy, snapshot["odom_xy"])
            rear_clear = float(snapshot.get("rear_clear", float("inf")))
            if wall["distance"] >= self.heading_retreat_wall:
                self.publish_zero(10)
                rospy.logwarn(
                    "CENTERLINE_V2_HEADING_RETREAT_CLEAR wall=%.3f "
                    "moved=%.3f; realigning",
                    wall["distance"], moved)
                return self.align_real_wall(
                    "CENTERLINE_V2_CLOSE_HEADING_REALIGN",
                    tolerance=math.radians(2.4))
            if (moved >= self.heading_retreat_max_distance or
                    rear_clear < self.reverse_rear_clear):
                self.publish_zero(20)
                rospy.logerr(
                    "CENTERLINE_V2_HEADING_RETREAT_BLOCKED wall=%.3f "
                    "moved=%.3f rear=%.3f",
                    wall["distance"], moved, rear_clear)
                return False
            self.publish_center_command(x=-self.heading_retreat_speed)
            rospy.logwarn_throttle(
                0.25,
                "CENTERLINE_V2_HEADING_RETREAT wall=%.3f/%.3f moved=%.3f "
                "rear=%.3f cmd_x=-%.3f",
                wall["distance"], self.heading_retreat_wall, moved,
                rear_clear, self.heading_retreat_speed)
            rate.sleep()
        self.publish_zero(20)
        rospy.logerr("CENTERLINE_V2_HEADING_RETREAT_TIMEOUT")
        return False

    def approach_centered_wall(self):
        self.publish_state(
            "CENTERLINE_WALL_APPROACH_V2",
            wall_target=self.final_wall_distance,
            nose_gap_target=self.final_wall_distance - 0.061)
        start_xy = self.snapshot_center()["odom_xy"]
        progress_xy = start_xy
        last_progress = time.time()
        settle_until = None
        success_since = None
        reverse_start_xy = None
        start = time.time()
        rate = rospy.Rate(20)

        while (not rospy.is_shutdown() and
               time.time() - start < self.approach_timeout):
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            now = time.time()
            if (wall is None or now - snapshot["scan_time"] > 1.0):
                self.publish_center_command()
                success_since = None
                rate.sleep()
                continue

            travelled = self.distance_between(start_xy, snapshot["odom_xy"])
            if self.distance_between(progress_xy, snapshot["odom_xy"]) >= 0.008:
                progress_xy = snapshot["odom_xy"]
                last_progress = now
            distance_error = wall["distance"] - self.final_wall_distance
            heading_error = wall["heading_error"]

            recoverable_overshoot = max(
                self.final_fail_tolerance, self.reverse_recover_limit)
            if (snapshot["front_min"] < self.front_emergency or
                    distance_error < -recoverable_overshoot):
                self.publish_zero(30)
                rospy.logerr(
                    "CENTERLINE_V2_FRONT_EMERGENCY front=%.3f wall=%.3f "
                    "error=%.3f",
                    snapshot["front_min"], wall["distance"], distance_error)
                return False
            if travelled > self.approach_max_travel:
                self.publish_zero(30)
                rospy.logerr("CENTERLINE_V2_TRAVEL_LIMIT %.3fm", travelled)
                return False
            if (wall["distance"] <= 0.30 and
                    abs(heading_error) > self.final_heading_tolerance):
                if not self._recover_close_heading(snapshot):
                    return False
                settle_until = None
                success_since = None
                reverse_start_xy = None
                progress_xy = self.snapshot_center()["odom_xy"]
                last_progress = time.time()
                continue
            if abs(heading_error) > self.approach_heading_limit:
                self.publish_zero(8)
                if wall["distance"] <= 0.30:
                    if not self._recover_close_heading(snapshot):
                        return False
                else:
                    if not self.align_real_wall(
                            "CENTERLINE_V2_HEADING_RECOVERY"):
                        return False
                settle_until = None
                success_since = None
                progress_xy = self.snapshot_center()["odom_xy"]
                last_progress = time.time()
                continue

            if distance_error > self.stop_lead:
                settle_until = None
            settled = (settle_until is not None and now >= settle_until)
            phase, command_x = longitudinal_decision(
                distance_error,
                tolerance=self.final_wall_tolerance,
                stop_lead=self.stop_lead,
                settled=settled,
                near_band=self.near_band,
                near_min_speed=self.near_min_speed,
                near_max_speed=self.near_max_speed,
                far_full_speed_error=self.far_full_speed_error,
                far_max_speed=self.approach_fast_speed,
                reverse_speed=self.reverse_speed)

            if phase == "settle" and settle_until is None:
                settle_until = now + self.settle_duration
            if phase == "hold":
                if success_since is None:
                    success_since = now
            else:
                success_since = None

            rear_clear = float(snapshot.get("rear_clear", float("inf")))
            if phase == "reverse":
                if rear_clear < self.reverse_rear_clear:
                    self.publish_zero(30)
                    rospy.logerr(
                        "CENTERLINE_V2_REVERSE_BLOCKED rear=%.3f error=%.3f",
                        rear_clear, distance_error)
                    return False
                if reverse_start_xy is None:
                    reverse_start_xy = snapshot["odom_xy"]
                correction = self.distance_between(
                    reverse_start_xy, snapshot["odom_xy"])
                if correction > self.max_reverse_correction:
                    self.publish_zero(30)
                    rospy.logerr(
                        "CENTERLINE_V2_REVERSE_LIMIT correction=%.3f",
                        correction)
                    return False
            else:
                reverse_start_xy = None

            if phase in ("hold", "settle", "reverse"):
                command_wz = 0.0
            else:
                angular_limit = (self.near_heading_max_speed
                                 if phase in ("near", "crawl") else 0.045)
                command_wz = clamp(
                    self.heading_kp * heading_error,
                    -angular_limit, angular_limit)
            self.publish_center_command(x=command_x, wz=command_wz)

            stable_time = (0.0 if success_since is None
                           else now - success_since)
            rospy.logwarn_throttle(
                0.25,
                "CENTERLINE_V2_APPROACH phase=%s wall=%.3f target=%.3f "
                "error=%.3f heading=%.2fdeg travel=%.3f rear=%.3f "
                "stable=%.2f/%.2fs cmd=(%.3f,%.3f)",
                phase, wall["distance"], self.final_wall_distance,
                distance_error, math.degrees(heading_error), travelled,
                rear_clear, stable_time, self.success_hold,
                command_x, command_wz)

            if success_since is not None and stable_time >= self.success_hold:
                self.publish_zero(30)
                self.publish_state(
                    "CENTERLINE_PARKING_SUCCESS",
                    wall_distance=wall["distance"],
                    nose_gap_estimate=wall["distance"] - 0.061,
                    heading_error_deg=math.degrees(heading_error),
                    forward_motion=travelled)
                rospy.logwarn(
                    "CENTERLINE_V2_PARKED wall=%.3fm nose_gap=%.3fm "
                    "heading=%.2fdeg",
                    wall["distance"], wall["distance"] - 0.061,
                    math.degrees(heading_error))
                return True
            if abs(command_x) > 0.0 and now - last_progress > 3.0:
                self.publish_zero(30)
                rospy.logerr("CENTERLINE_V2_NO_PROGRESS phase=%s", phase)
                return False
            rate.sleep()

        self.publish_zero(30)
        rospy.logerr("CENTERLINE_V2_APPROACH_TIMEOUT")
        return False


if __name__ == "__main__":
    ReliableContinuousSearchParkingManager().run()
