#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Factory-room manager with recoverable two-stage white-box parking."""

import math
import time

import rospy
from geometry_msgs.msg import Twist

from factory_room_delivery_manager import FactoryRoomDeliveryManager
from factory_room_vision_core import clamp


class RobustParkingFactoryRoomManager(FactoryRoomDeliveryManager):
    def park_inside_square(self):
        align_tolerance = float(rospy.get_param(
            "~parking_align_tolerance_m", 0.055))
        align_frames = int(rospy.get_param(
            "~parking_align_stable_frames", 5))
        align_timeout = float(rospy.get_param(
            "~parking_align_timeout_s", 10.0))
        min_travel = float(rospy.get_param(
            "~parking_min_travel_m", 0.10))
        max_travel = float(rospy.get_param(
            "~parking_max_travel_m", 0.52))
        wall_tolerance = float(rospy.get_param(
            "~parking_wall_tolerance_m", 0.04))
        wall_fail_tolerance = float(rospy.get_param(
            "~parking_wall_fail_tolerance_m", 0.10))
        lost_steer_hold = float(rospy.get_param(
            "~parking_lost_steer_hold_m", 0.06))
        approach_timeout = float(rospy.get_param(
            "~parking_approach_timeout_s", 24.0))
        fast_speed = float(rospy.get_param(
            "~parking_speed_fast", 0.08))
        slow_speed = float(rospy.get_param(
            "~parking_speed_slow", 0.04))

        self.move_client.cancel_all_goals()
        self.publish_zero(8)
        self.publish_state("PARKING_ALIGN_WHITE_BOX")

        # Stage 1: rotate in place until the frame center is stable. No forward
        # motion is allowed before this gate passes.
        stable = 0
        last_result = None
        align_start = time.time()
        rate = rospy.Rate(10)
        while (not rospy.is_shutdown() and
               time.time() - align_start < align_timeout):
            result = self.detect_square_once()
            if not result or not result.get("found"):
                stable = 0
                self.publish_zero()
                rospy.logwarn_throttle(
                    1.0, "WHITE_BOX_ALIGN_WAITING_FOR_FRAME")
                rate.sleep()
                continue

            last_result = result
            lateral = float(result["lateral_error_m"])
            if abs(lateral) <= align_tolerance:
                stable += 1
                self.publish_zero()
                rospy.loginfo_throttle(
                    0.5,
                    "WHITE_BOX_ALIGN_STABLE frames=%d/%d off=%.3fm conf=%.2f",
                    stable, align_frames, lateral,
                    float(result.get("confidence", 0.0)))
                if stable >= max(3, align_frames):
                    break
            else:
                stable = 0
                direction = 1.0 if self.image_mirrored else -1.0
                wz = clamp(direction * lateral * 2.0, -0.18, 0.18)
                if abs(wz) < 0.09:
                    wz = math.copysign(0.09, wz)
                command = Twist()
                command.angular.z = wz
                self.cmd_pub.publish(command)
                rospy.loginfo_throttle(
                    0.5, "WHITE_BOX_ALIGN off=%.3fm cmd_wz=%.3f",
                    lateral, wz)
            rate.sleep()

        self.publish_zero(6)
        if stable < max(3, align_frames) or last_result is None:
            rospy.logerr("WHITE_BOX_ALIGN_FAILED; no forward motion performed")
            return False

        with self.lock:
            initial_front = self.front_min
            initial_wall = self.front_wall
        entry_odom, _ = self.snapshot_odom()
        if entry_odom is None or math.isinf(initial_wall):
            rospy.logerr(
                "WHITE_BOX_ENTRY_REJECTED odom=%s wall=%s",
                str(entry_odom), str(initial_wall))
            return False

        wall_target = self.wall_final_distance_m
        planned_travel = clamp(
            initial_wall - wall_target, min_travel, max_travel)
        entry_lateral = float(last_result["lateral_error_m"])
        last_lateral = entry_lateral
        last_seen_travel = 0.0
        progress_anchor = entry_odom
        last_progress_time = time.time()
        final_wall_frames = 0
        approach_start = time.time()
        self.publish_state(
            "PARKING_IN_WHITE_BOX",
            initial_wall=initial_wall,
            planned_travel=planned_travel,
            entry_offset=entry_lateral)
        rospy.logwarn(
            "WHITE_BOX_ENTRY_COMMITTED wall=%.3fm travel=%.3fm off=%.3fm",
            initial_wall, planned_travel, entry_lateral)

        # Stage 2: after alignment, use odometry and the lidar wall estimate.
        # Losing the floor frame near the camera is expected and is not a stop
        # condition by itself.
        while (not rospy.is_shutdown() and
               time.time() - approach_start < approach_timeout):
            current_odom, _ = self.snapshot_odom()
            travelled = self.odom_distance(entry_odom, current_odom)
            if (current_odom is not None and progress_anchor is not None and
                    self.odom_distance(progress_anchor, current_odom) >= 0.015):
                progress_anchor = current_odom
                last_progress_time = time.time()

            result = self.detect_square_once()
            visible = bool(result and result.get("found"))
            if visible:
                last_result = result
                last_lateral = float(result["lateral_error_m"])
                last_seen_travel = travelled

            with self.lock:
                front = self.front_min
                wall = self.front_wall

            wall_reached = (
                not math.isinf(wall) and
                wall <= wall_target + wall_tolerance)
            if wall_reached and travelled >= min_travel:
                final_wall_frames += 1
            else:
                final_wall_frames = 0
            if final_wall_frames >= 3:
                self.publish_zero(15)
                rospy.logwarn(
                    "WHITE_BOX_PARKED wall=%.3fm travel=%.3fm off_entry=%.3fm",
                    wall, travelled, entry_lateral)
                return True

            if front <= self.square_front_hard_stop_m:
                self.publish_zero(15)
                valid = (travelled >= min_travel and not math.isinf(wall) and
                         wall <= wall_target + wall_fail_tolerance)
                rospy.logerr(
                    "WHITE_BOX_FRONT_STOP front=%.3fm wall=%.3fm travel=%.3fm valid=%s",
                    front, wall, travelled, str(valid))
                return valid

            if travelled >= max_travel:
                self.publish_zero(15)
                valid = (not math.isinf(wall) and
                         wall <= wall_target + wall_fail_tolerance)
                rospy.logerr(
                    "WHITE_BOX_MAX_TRAVEL wall=%.3fm travel=%.3fm valid=%s",
                    wall, travelled, str(valid))
                return valid

            if travelled >= planned_travel:
                self.publish_zero(15)
                valid = (not math.isinf(wall) and
                         wall <= wall_target + wall_fail_tolerance)
                rospy.logwarn(
                    "WHITE_BOX_ODOM_TARGET wall=%.3fm travel=%.3fm valid=%s",
                    wall, travelled, str(valid))
                return valid

            if time.time() - last_progress_time > 4.0:
                self.publish_zero(15)
                rospy.logerr(
                    "WHITE_BOX_NO_PROGRESS wall=%.3fm travel=%.3fm",
                    wall, travelled)
                return False

            remaining = max(0.0, planned_travel - travelled)
            command = Twist()
            if visible or travelled - last_seen_travel <= lost_steer_hold:
                direction = 1.0 if self.image_mirrored else -1.0
                command.angular.z = clamp(
                    direction * last_lateral * 1.7, -0.14, 0.14)
            else:
                command.angular.z = 0.0

            if visible and abs(last_lateral) > 0.12:
                command.linear.x = 0.025
            elif remaining <= 0.16 or front < 0.42:
                command.linear.x = slow_speed
            else:
                command.linear.x = fast_speed
            self.cmd_pub.publish(command)
            rospy.logwarn_throttle(
                0.8,
                "WHITE_BOX_APPROACH visible=%s front=%.3f wall=%.3f travel=%.3f/%.3f off=%.3f cmd=(%.3f,%.3f)",
                str(visible), front, wall, travelled, planned_travel,
                last_lateral, command.linear.x, command.angular.z)
            rate.sleep()

        self.publish_zero(15)
        rospy.logerr("WHITE_BOX_APPROACH_TIMEOUT; robot stopped")
        return False


if __name__ == "__main__":
    node = RobustParkingFactoryRoomManager()
    rospy.spin()

