#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Cardinal-heading alignment followed by straight visual retreat recovery."""

import math
import time

import rospy
from geometry_msgs.msg import Twist

from factory_room_delivery_manager_parking_v11 import StraightApproachParkingManager
from factory_room_delivery_manager_parking_v7 import norm_angle
from factory_room_vision_parking_v5 import clamp


class CardinalRetreatParkingManager(StraightApproachParkingManager):
    def __init__(self):
        super(CardinalRetreatParkingManager, self).__init__()
        self.cardinal_aligned = False
        self.retreat_attempted = False
        self.cardinal_target_yaw = None

    def align_cardinal_heading(self):
        _, current_yaw = self.snapshot_odom()
        if current_yaw is None:
            rospy.logerr("CARDINAL_ALIGN_REJECTED odom yaw unavailable")
            return False
        quarter_turn = 0.5 * math.pi
        target_yaw = round(current_yaw / quarter_turn) * quarter_turn
        target_yaw = norm_angle(target_yaw)
        self.cardinal_target_yaw = target_yaw
        target_qz = math.sin(0.5 * target_yaw)
        target_qw = math.cos(0.5 * target_yaw)
        tolerance = math.radians(float(rospy.get_param(
            "~parking_cardinal_tolerance_deg", 2.0)))
        timeout_s = float(rospy.get_param(
            "~parking_cardinal_timeout_s", 9.0))
        max_wz = float(rospy.get_param(
            "~parking_cardinal_max_wz", 0.28))
        self.parking_close_mode = True
        self.publish_parking_mode()
        self.move_client.cancel_all_goals()
        self.publish_zero(8)
        self.publish_state(
            "ALIGNING_CARDINAL_HEADING",
            current_yaw_deg=math.degrees(current_yaw),
            target_yaw_deg=math.degrees(target_yaw),
            target_qz=target_qz, target_qw=target_qw)
        rospy.logwarn(
            "CARDINAL_ALIGN start yaw=%.2fdeg target=%.2fdeg qz=%.5f qw=%.5f",
            math.degrees(current_yaw), math.degrees(target_yaw),
            target_qz, target_qw)

        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            _, current_yaw = self.snapshot_odom()
            if current_yaw is None:
                self.publish_zero(20)
                return False
            error = norm_angle(target_yaw - current_yaw)
            command = Twist()
            if abs(error) <= tolerance:
                stable += 1
            else:
                stable = 0
                command.angular.z = clamp(1.25 * error, -max_wz, max_wz)
            self.cmd_pub.publish(command)
            rospy.logwarn_throttle(
                0.35,
                "CARDINAL_ALIGN yaw=%.2f target=%.2f error=%.2fdeg stable=%d/5 cmd_w=%.3f",
                math.degrees(current_yaw), math.degrees(target_yaw),
                math.degrees(error), stable, command.angular.z)
            if stable >= 5:
                self.publish_zero(15)
                self.cardinal_aligned = True
                rospy.logwarn(
                    "CARDINAL_ALIGN_OK yaw=%.2fdeg error=%.2fdeg",
                    math.degrees(current_yaw), math.degrees(error))
                return True
            rate.sleep()
        self.publish_zero(20)
        rospy.logerr("CARDINAL_ALIGN_FAILED")
        return False

    def acquire_square(self, timeout_s=10.0):
        self.ocr_control("disable")
        if not self.cardinal_aligned and not self.align_cardinal_heading():
            self.parking_close_mode = False
            self.publish_parking_mode()
            return None
        result = super(CardinalRetreatParkingManager, self).acquire_square(
            timeout_s=timeout_s)
        if result is None and self.retreat_attempted:
            self.parking_close_mode = False
            self.publish_parking_mode()
        return result

    def move_closer_for_square(self):
        # The old method moved forward.  At the observed 0.67 m wall distance
        # the frame is too low/cropped, so widen the view by a short straight
        # retreat while holding the cardinal quaternion heading.
        self.retreat_attempted = True
        if not self.cardinal_aligned or self.cardinal_target_yaw is None:
            self.parking_close_mode = False
            self.publish_parking_mode()
            return False
        start_xy, _ = self.snapshot_odom()
        if start_xy is None:
            return False
        retreat_limit = float(rospy.get_param(
            "~parking_retreat_distance_m", 0.12))
        retreat_speed = abs(float(rospy.get_param(
            "~parking_retreat_speed", 0.040)))
        yaw_tolerance = math.radians(float(rospy.get_param(
            "~parking_retreat_yaw_tolerance_deg", 2.0)))
        self.publish_zero(8)
        self.publish_state(
            "WHITE_BOX_CARDINAL_RETREAT",
            target_yaw_deg=math.degrees(self.cardinal_target_yaw),
            retreat_limit=retreat_limit)
        rospy.logwarn(
            "WHITE_BOX_CARDINAL_RETREAT start target_yaw=%.1fdeg limit=%.3fm",
            math.degrees(self.cardinal_target_yaw), retreat_limit)

        frame_stable = 0
        moved = 0.0
        start = time.time()
        rate = rospy.Rate(12)
        while not rospy.is_shutdown() and time.time() - start < 5.0:
            current_xy, current_yaw = self.snapshot_odom()
            moved = self.odom_distance(start_xy, current_xy)
            if current_yaw is None or moved >= retreat_limit:
                break
            yaw_error = norm_angle(self.cardinal_target_yaw - current_yaw)
            if abs(yaw_error) > yaw_tolerance:
                self.publish_zero(10)
                rospy.logerr(
                    "WHITE_BOX_CARDINAL_RETREAT paused yaw_error=%.2fdeg",
                    math.degrees(yaw_error))
                if not self.align_cardinal_heading():
                    break
                continue

            result = self.detect_square_once()
            if self.square_is_geometrically_valid(result):
                frame_stable += 1
                if frame_stable >= 2:
                    rospy.logwarn(
                        "WHITE_BOX_CARDINAL_RETREAT frame acquired moved=%.3fm",
                        moved)
                    break
            else:
                frame_stable = 0

            command = Twist()
            command.linear.x = -retreat_speed
            self.cmd_pub.publish(command)
            rospy.logwarn_throttle(
                0.5,
                "WHITE_BOX_CARDINAL_RETREAT moved=%.3f/%.3f yaw=%.2fdeg",
                moved, retreat_limit, math.degrees(current_yaw))
            rate.sleep()

        self.publish_zero(20)
        rospy.logwarn(
            "WHITE_BOX_CARDINAL_RETREAT done moved=%.3f stable=%d",
            moved, frame_stable)
        if moved < 0.02 and frame_stable < 2:
            self.parking_close_mode = False
            self.publish_parking_mode()
            return False
        return True


if __name__ == "__main__":
    node = CardinalRetreatParkingManager()
    rospy.spin()

