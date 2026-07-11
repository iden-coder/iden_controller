#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fast precise parking with odometry-quaternion heading lock."""

import math
import time

import rospy
from geometry_msgs.msg import Twist

from factory_room_delivery_manager_parking_v6 import AspectSafeParkingManager
from factory_room_vision_parking_v5 import clamp


def norm_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class QuaternionLockedParkingManager(AspectSafeParkingManager):
    def park_inside_square(self):
        center_tolerance = float(rospy.get_param(
            "~parking_center_tolerance_m", 0.010))
        vision_heading_tolerance = math.radians(float(rospy.get_param(
            "~parking_heading_tolerance_deg", 4.0)))
        final_heading_tolerance = math.radians(float(rospy.get_param(
            "~parking_final_heading_tolerance_deg", 1.5)))
        stable_need = int(rospy.get_param(
            "~parking_center_stable_frames", 4))
        lateral_gain = float(rospy.get_param("~parking_lateral_gain", 1.8))
        vision_heading_gain = float(rospy.get_param(
            "~parking_heading_gain", 0.9))
        odom_heading_gain = float(rospy.get_param(
            "~parking_odom_heading_gain", 1.8))
        max_lateral = float(rospy.get_param(
            "~parking_max_lateral_speed", 0.09))
        max_heading = float(rospy.get_param(
            "~parking_max_heading_speed", 0.14))
        align_timeout = float(rospy.get_param(
            "~parking_align_timeout_s", 12.0))
        approach_timeout = float(rospy.get_param(
            "~parking_approach_timeout_s", 18.0))
        max_forward_limit = float(rospy.get_param(
            "~parking_max_travel_m", 0.78))
        wall_target = self.wall_final_distance_m
        wall_tolerance = float(rospy.get_param(
            "~parking_wall_tolerance_m", 0.008))
        wall_fail_tolerance = float(rospy.get_param(
            "~parking_wall_fail_tolerance_m", 0.020))

        self.parking_close_mode = True
        self.publish_parking_mode()
        self.move_client.cancel_all_goals()
        self.publish_zero(10)
        self.publish_state("QUATERNION_PARKING_CENTERING")
        rospy.logwarn("QUATERNION_PARKING_MODE enabled")

        success = False
        try:
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
                        0.8, "QUATERNION_CENTER_WAIT frame unavailable")
                    rate.sleep()
                    continue
                lateral, heading = self.parking_errors(result)
                centered = (abs(lateral) <= center_tolerance and
                            abs(heading) <= vision_heading_tolerance)
                command = Twist()
                if centered:
                    stable += 1
                else:
                    stable = 0
                    command.linear.y = clamp(
                        lateral_gain * lateral, -max_lateral, max_lateral)
                    command.angular.z = clamp(
                        vision_heading_gain * heading,
                        -max_heading, max_heading)
                self.cmd_pub.publish(command)
                rospy.logwarn_throttle(
                    0.5,
                    "QUATERNION_CENTER off=%.3fm heading=%.2fdeg stable=%d/%d cmd=(y%.3f,w%.3f)",
                    lateral, math.degrees(heading), stable, stable_need,
                    command.linear.y, command.angular.z)
                if stable >= max(3, stable_need):
                    break
                rate.sleep()
            self.publish_zero(6)
            if stable < max(3, stable_need):
                rospy.logerr("QUATERNION_CENTER_FAILED")
                return False

            entry_xy, entry_yaw = self.snapshot_odom()
            with self.lock:
                initial_wall = self.front_wall
            if entry_xy is None or entry_yaw is None or math.isinf(initial_wall):
                rospy.logerr("QUATERNION_ENTRY_REJECTED wall/odom unavailable")
                return False

            # Snap to a cardinal quaternion only when the vision-aligned yaw is
            # already close to one.  Standalone odometry starts at zero; in the
            # full mission this also accepts 90/180/-90 degree wall headings.
            cardinal = round(entry_yaw / (0.5 * math.pi)) * (0.5 * math.pi)
            if abs(norm_angle(cardinal - entry_yaw)) <= math.radians(12.0):
                target_yaw = norm_angle(cardinal)
                target_source = "CARDINAL"
            else:
                target_yaw = entry_yaw
                target_source = "VISION_LOCK"
            target_qz = math.sin(0.5 * target_yaw)
            target_qw = math.cos(0.5 * target_yaw)

            required_forward = max(0.0, initial_wall - wall_target)
            max_forward = min(max_forward_limit, required_forward + 0.07)
            if max_forward + 1.0e-3 < required_forward:
                rospy.logerr(
                    "QUATERNION_TRAVEL_REJECTED required=%.3f limit=%.3f",
                    required_forward, max_forward_limit)
                return False
            rospy.logwarn(
                "QUATERNION_TARGET source=%s yaw=%.2fdeg qz=%.5f qw=%.5f wall=%.3f travel=%.3f",
                target_source, math.degrees(target_yaw), target_qz, target_qw,
                initial_wall, required_forward)

            progress_anchor = entry_xy
            last_progress = time.time()
            final_frames = 0
            start = time.time()
            self.publish_state(
                "QUATERNION_PARKING_ENTERING",
                target_yaw_deg=math.degrees(target_yaw),
                target_qz=target_qz, target_qw=target_qw,
                wall_target=wall_target)

            while (not rospy.is_shutdown() and
                   time.time() - start < approach_timeout):
                current_xy, current_yaw = self.snapshot_odom()
                forward, lateral_motion = self.projected_motion(
                    entry_xy, entry_yaw, current_xy)
                forward = max(0.0, forward)
                if (progress_anchor is not None and current_xy is not None and
                        self.odom_distance(progress_anchor, current_xy) >= 0.012):
                    progress_anchor = current_xy
                    last_progress = time.time()
                if current_yaw is None:
                    self.publish_zero(20)
                    rospy.logerr("QUATERNION_ODOM_LOST")
                    return False

                yaw_error = norm_angle(target_yaw - current_yaw)
                result = self.detect_square_once()
                visible = self.square_is_geometrically_valid(result)
                command = Twist()
                if visible:
                    lateral, _ = self.parking_errors(result)
                    command.linear.y = clamp(
                        lateral_gain * lateral, -max_lateral, max_lateral)
                else:
                    lateral = 0.0
                command.angular.z = clamp(
                    odom_heading_gain * yaw_error,
                    -max_heading, max_heading)

                with self.lock:
                    front = self.front_min
                    wall = self.front_wall
                wall_reached = (not math.isinf(wall) and
                                wall <= wall_target + wall_tolerance)
                angle_reached = abs(yaw_error) <= final_heading_tolerance
                if wall_reached:
                    command.linear.x = 0.0
                    command.linear.y = 0.0
                    final_frames = final_frames + 1 if angle_reached else 0
                else:
                    final_frames = 0
                    if wall > 0.35:
                        command.linear.x = 0.10
                    elif wall > 0.20:
                        command.linear.x = 0.07
                    elif wall > 0.145:
                        command.linear.x = 0.045
                    else:
                        command.linear.x = 0.022

                if final_frames >= 4:
                    self.publish_zero(25)
                    success = True
                    rospy.logwarn(
                        "QUATERNION_PARKED wall=%.3fm yaw=%.2fdeg error=%.2fdeg forward=%.3fm lateral=%.3fm",
                        wall, math.degrees(current_yaw),
                        math.degrees(yaw_error), forward, lateral_motion)
                    return True
                if front <= self.square_front_hard_stop_m:
                    self.publish_zero(25)
                    valid = (not math.isinf(wall) and
                             wall <= wall_target + wall_fail_tolerance and
                             angle_reached)
                    rospy.logerr(
                        "QUATERNION_FRONT_STOP front=%.3f wall=%.3f angle_ok=%s",
                        front, wall, str(angle_reached))
                    return valid
                if forward >= max_forward:
                    self.publish_zero(25)
                    rospy.logerr(
                        "QUATERNION_TRAVEL_LIMIT wall=%.3f forward=%.3f/%.3f yaw_error=%.2fdeg",
                        wall, forward, max_forward, math.degrees(yaw_error))
                    return False
                if time.time() - last_progress > 3.0 and not wall_reached:
                    self.publish_zero(25)
                    rospy.logerr("QUATERNION_NO_PROGRESS")
                    return False

                self.cmd_pub.publish(command)
                rospy.logwarn_throttle(
                    0.45,
                    "QUATERNION_ENTRY visible=%s wall=%.3f forward=%.3f/%.3f off=%.3f yaw=%.2f target=%.2f err=%.2f cmd=(%.3f,%.3f,%.3f)",
                    str(visible), wall, forward, max_forward, lateral,
                    math.degrees(current_yaw), math.degrees(target_yaw),
                    math.degrees(yaw_error), command.linear.x,
                    command.linear.y, command.angular.z)
                rate.sleep()

            self.publish_zero(25)
            rospy.logerr("QUATERNION_PARKING_TIMEOUT")
            return False
        finally:
            self.parking_close_mode = False
            self.publish_parking_mode()
            self.publish_zero(30)
            rospy.logwarn(
                "QUATERNION_PARKING_MODE disabled success=%s",
                str(success))


if __name__ == "__main__":
    node = QuaternionLockedParkingManager()
    rospy.spin()

