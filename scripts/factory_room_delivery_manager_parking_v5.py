#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Holonomic garage entry adapted from flow_end/src/follow_line.cpp."""

import math
import time

import rospy
from geometry_msgs.msg import Twist

from factory_room_delivery_manager_parking_v4 import CenterlineParkingFactoryRoomManager
from factory_room_vision_core import clamp


class FlowEndStyleParkingManager(CenterlineParkingFactoryRoomManager):
    def parking_errors(self, result):
        direction = 1.0 if self.image_mirrored else -1.0
        lateral = float(result.get(
            "raw_lateral_error_m", result.get("lateral_error_m", 0.0)))
        heading = float(result.get("heading_error_rad", 0.0))
        return direction * lateral, direction * heading

    @staticmethod
    def projected_motion(start_xy, start_yaw, current_xy):
        if start_xy is None or current_xy is None or start_yaw is None:
            return 0.0, 0.0
        dx = current_xy[0] - start_xy[0]
        dy = current_xy[1] - start_xy[1]
        forward = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
        lateral = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
        return forward, lateral

    def park_inside_square(self):
        center_tolerance = float(rospy.get_param(
            "~parking_center_tolerance_m", 0.025))
        heading_tolerance = math.radians(float(rospy.get_param(
            "~parking_heading_tolerance_deg", 4.0)))
        stable_need = int(rospy.get_param(
            "~parking_center_stable_frames", 5))
        lateral_gain = float(rospy.get_param(
            "~parking_lateral_gain", 1.6))
        heading_gain = float(rospy.get_param(
            "~parking_heading_gain", 0.9))
        max_lateral = float(rospy.get_param(
            "~parking_max_lateral_speed", 0.08))
        max_heading = float(rospy.get_param(
            "~parking_max_heading_speed", 0.16))
        align_timeout = float(rospy.get_param(
            "~parking_align_timeout_s", 10.0))
        approach_timeout = float(rospy.get_param(
            "~parking_approach_timeout_s", 24.0))
        max_forward = float(rospy.get_param(
            "~parking_max_travel_m", 0.62))
        wall_target = self.wall_final_distance_m
        wall_tolerance = float(rospy.get_param(
            "~parking_wall_tolerance_m", 0.008))
        wall_fail_tolerance = float(rospy.get_param(
            "~parking_wall_fail_tolerance_m", 0.020))

        self.parking_close_mode = True
        self.publish_parking_mode()
        self.move_client.cancel_all_goals()
        self.publish_zero(10)
        self.publish_state("FLOW_END_CENTERING_IN_GARAGE")
        rospy.logwarn("FLOW_END_PARKING_MODE enabled")

        try:
            # flow_end first removes lateral error with linear.y while keeping
            # the vehicle facing the garage. Require live two-rail geometry.
            stable = 0
            start = time.time()
            rate = rospy.Rate(12)
            while (not rospy.is_shutdown() and
                   time.time() - start < align_timeout):
                result = self.detect_square_once()
                if not self.square_is_geometrically_valid(result):
                    stable = 0
                    self.publish_zero()
                    rospy.logwarn_throttle(
                        0.8, "FLOW_END_CENTER_WAIT two rails unavailable")
                    rate.sleep()
                    continue
                lateral, heading = self.parking_errors(result)
                centered = (abs(lateral) <= center_tolerance and
                            abs(heading) <= heading_tolerance)
                if centered:
                    stable += 1
                    self.publish_zero()
                else:
                    stable = 0
                    command = Twist()
                    command.linear.y = clamp(
                        lateral_gain * lateral, -max_lateral, max_lateral)
                    command.angular.z = clamp(
                        heading_gain * heading, -max_heading, max_heading)
                    self.cmd_pub.publish(command)
                rospy.logwarn_throttle(
                    0.6,
                    "FLOW_END_CENTER lateral=%.3fm heading=%.1fdeg stable=%d/%d cmd_y=%.3f",
                    lateral, math.degrees(heading), stable, stable_need,
                    0.0 if centered else command.linear.y)
                if stable >= max(3, stable_need):
                    break
                rate.sleep()
            self.publish_zero(8)
            if stable < max(3, stable_need):
                rospy.logerr("FLOW_END_CENTER_FAILED; vehicle did not enter")
                return False

            with self.lock:
                initial_wall = self.front_wall
            entry_xy, entry_yaw = self.snapshot_odom()
            if entry_xy is None or entry_yaw is None or math.isinf(initial_wall):
                rospy.logerr("FLOW_END_ENTRY_REJECTED wall/odom unavailable")
                return False
            planned_forward = clamp(
                initial_wall - wall_target, 0.10, max_forward)
            progress_anchor = entry_xy
            last_progress = time.time()
            last_centered = True
            final_frames = 0
            start = time.time()
            self.publish_state(
                "FLOW_END_ENTERING_GARAGE", initial_wall=initial_wall,
                planned_forward=planned_forward, wall_target=wall_target)

            while (not rospy.is_shutdown() and
                   time.time() - start < approach_timeout):
                current_xy, _ = self.snapshot_odom()
                forward, lateral_motion = self.projected_motion(
                    entry_xy, entry_yaw, current_xy)
                forward = max(0.0, forward)
                if (progress_anchor is not None and current_xy is not None and
                        self.odom_distance(progress_anchor, current_xy) >= 0.012):
                    progress_anchor = current_xy
                    last_progress = time.time()

                result = self.detect_square_once()
                visible = self.square_is_geometrically_valid(result)
                command = Twist()
                if visible:
                    lateral, heading = self.parking_errors(result)
                    last_centered = (abs(lateral) <= center_tolerance and
                                     abs(heading) <= heading_tolerance)
                    command.linear.y = clamp(
                        lateral_gain * lateral, -max_lateral, max_lateral)
                    command.angular.z = clamp(
                        heading_gain * heading, -max_heading, max_heading)
                else:
                    lateral, heading = 0.0, 0.0
                    command.linear.y = 0.0
                    command.angular.z = 0.0

                with self.lock:
                    front = self.front_min
                    wall = self.front_wall
                wall_reached = (not math.isinf(wall) and
                                wall <= wall_target + wall_tolerance)
                if wall_reached:
                    command.linear.x = 0.0
                    if not visible or last_centered:
                        final_frames += 1
                    else:
                        final_frames = 0
                else:
                    final_frames = 0
                    if wall < 0.15:
                        command.linear.x = 0.015
                    elif wall < 0.25:
                        command.linear.x = 0.025
                    else:
                        command.linear.x = 0.06

                if final_frames >= 3:
                    self.publish_zero(20)
                    rospy.logwarn(
                        "FLOW_END_PARKED wall=%.3fm forward=%.3fm lateral_motion=%.3fm",
                        wall, forward, lateral_motion)
                    return True
                if front <= self.square_front_hard_stop_m:
                    self.publish_zero(20)
                    valid = (not math.isinf(wall) and
                             wall <= wall_target + wall_fail_tolerance and
                             last_centered)
                    rospy.logerr(
                        "FLOW_END_FRONT_STOP front=%.3fm wall=%.3fm valid=%s",
                        front, wall, str(valid))
                    return valid
                if forward >= max_forward or forward >= planned_forward + 0.03:
                    self.publish_zero(20)
                    valid = (not math.isinf(wall) and
                             wall <= wall_target + wall_fail_tolerance and
                             last_centered)
                    rospy.logerr(
                        "FLOW_END_TRAVEL_LIMIT wall=%.3fm forward=%.3fm valid=%s",
                        wall, forward, str(valid))
                    return valid
                if time.time() - last_progress > 4.0:
                    self.publish_zero(20)
                    rospy.logerr("FLOW_END_NO_PROGRESS; stopped")
                    return False

                self.cmd_pub.publish(command)
                rospy.logwarn_throttle(
                    0.6,
                    "FLOW_END_ENTRY visible=%s wall=%.3f forward=%.3f/%.3f lateral=%.3f heading=%.1f cmd=(%.3f,%.3f,%.3f)",
                    str(visible), wall, forward, planned_forward, lateral,
                    math.degrees(heading), command.linear.x,
                    command.linear.y, command.angular.z)
                rate.sleep()

            self.publish_zero(20)
            rospy.logerr("FLOW_END_PARKING_TIMEOUT; stopped")
            return False
        finally:
            self.parking_close_mode = False
            self.publish_parking_mode()
            self.publish_zero(20)
            rospy.logwarn("FLOW_END_PARKING_MODE disabled")


if __name__ == "__main__":
    node = FlowEndStyleParkingManager()
    rospy.spin()

