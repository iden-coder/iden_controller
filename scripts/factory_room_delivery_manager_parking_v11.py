#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adaptive parking with heading-preserving base-frame close approach."""

import math
import time

import rospy
from geometry_msgs.msg import Twist

from factory_room_delivery_manager_parking_v10 import AdaptiveRangeParkingManager
from factory_room_vision_parking_v5 import clamp


class StraightApproachParkingManager(AdaptiveRangeParkingManager):
    def acquire_square(self, timeout_s=10.0):
        # OCR has already selected the workshop.  Stop its heavy image pipeline
        # while parking geometry consumes the same camera stream.
        self.ocr_control("disable")
        return super(StraightApproachParkingManager, self).acquire_square(
            timeout_s=timeout_s)

    def move_closer_for_square(self):
        with self.lock:
            initial_wall = self.front_wall
        start_xy, start_yaw = self.snapshot_odom()
        if (start_xy is None or start_yaw is None or
                math.isinf(initial_wall) or initial_wall <= 0.42):
            rospy.logwarn(
                "WHITE_BOX_STRAIGHT_APPROACH_SKIPPED wall=%s odom=%s",
                str(initial_wall), str(start_xy is not None))
            return False

        travel_target = clamp(initial_wall - 0.48, 0.06, 0.16)
        self.move_client.cancel_all_goals()
        self.publish_zero(8)
        self.publish_state(
            "WHITE_BOX_STRAIGHT_APPROACH",
            wall=initial_wall, travel_target=travel_target,
            locked_yaw_deg=math.degrees(start_yaw))
        rospy.logwarn(
            "WHITE_BOX_STRAIGHT_APPROACH start wall=%.3f travel=%.3f yaw=%.1fdeg; map navigation disabled",
            initial_wall, travel_target, math.degrees(start_yaw))

        rate = rospy.Rate(12)
        start_time = time.time()
        frame_stable = 0
        moved = 0.0
        while not rospy.is_shutdown() and time.time() - start_time < 5.0:
            current_xy, current_yaw = self.snapshot_odom()
            moved = self.odom_distance(start_xy, current_xy)
            with self.lock:
                front = self.front_min
                wall = self.front_wall
            if moved >= travel_target:
                break
            if front <= 0.30 or (not math.isinf(wall) and wall <= 0.40):
                rospy.logwarn(
                    "WHITE_BOX_STRAIGHT_APPROACH clearance stop front=%.3f wall=%.3f",
                    front, wall)
                break

            result = self.detect_square_once()
            if self.square_is_geometrically_valid(result):
                frame_stable += 1
                if frame_stable >= 2:
                    rospy.logwarn(
                        "WHITE_BOX_STRAIGHT_APPROACH frame appeared after %.3fm",
                        moved)
                    break
            else:
                frame_stable = 0

            command = Twist()
            command.linear.x = 0.055
            self.cmd_pub.publish(command)
            rospy.logwarn_throttle(
                0.5,
                "WHITE_BOX_STRAIGHT_APPROACH moving=%.3f/%.3f wall=%.3f yaw_now=%.1fdeg",
                moved, travel_target, wall,
                math.degrees(current_yaw) if current_yaw is not None else 999.0)
            rate.sleep()

        self.publish_zero(20)
        rospy.logwarn(
            "WHITE_BOX_STRAIGHT_APPROACH done moved=%.3f frame_stable=%d",
            moved, frame_stable)
        return moved >= 0.025 or frame_stable >= 2


if __name__ == "__main__":
    node = StraightApproachParkingManager()
    rospy.spin()

