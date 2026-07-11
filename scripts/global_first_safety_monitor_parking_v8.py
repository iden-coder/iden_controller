#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Heading-gated parking safety with an odometry lateral-travel guard."""

import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from global_first_safety_monitor_parking_v7 import (
    HeadingGatedParkingSafetyMonitor,
)


class LateralGuardParkingSafetyMonitor(HeadingGatedParkingSafetyMonitor):
    def __init__(self):
        self.odom_xy = None
        self.odom_yaw = None
        self.parking_anchor = None
        self.lateral_limit = 0.07
        super(LateralGuardParkingSafetyMonitor, self).__init__()
        self.lateral_limit = float(rospy.get_param(
            "~parking_lateral_travel_limit_m", 0.07))
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.odom_sub = rospy.Subscriber(
            odom_topic, Odometry, self.cb_odom_guard, queue_size=1)

    def cb_odom_guard(self, msg):
        pose = msg.pose.pose
        self.odom_xy = (pose.position.x, pose.position.y)
        q = pose.orientation
        self.odom_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def cb_parking_mode(self, msg):
        was_active = bool(getattr(self, "parking_mode", False))
        super(LateralGuardParkingSafetyMonitor, self).cb_parking_mode(msg)
        if msg.data and not was_active and self.odom_xy is not None:
            self.parking_anchor = (
                self.odom_xy[0], self.odom_xy[1], self.odom_yaw)
            rospy.logwarn(
                "PARKING_LATERAL_GUARD armed limit=%.3fm",
                self.lateral_limit)
        elif not msg.data:
            self.parking_anchor = None

    def lateral_displacement(self):
        if self.parking_anchor is None or self.odom_xy is None:
            return 0.0
        x0, y0, yaw0 = self.parking_anchor
        dx = self.odom_xy[0] - x0
        dy = self.odom_xy[1] - y0
        return -dx * math.sin(yaw0) + dy * math.cos(yaw0)

    def cb_cmd(self, msg):
        gated = Twist()
        gated.linear.x = msg.linear.x
        gated.linear.y = msg.linear.y
        gated.angular.z = msg.angular.z
        lateral = self.lateral_displacement()
        blocked = ((lateral >= self.lateral_limit and gated.linear.y > 0.0) or
                   (lateral <= -self.lateral_limit and gated.linear.y < 0.0))
        if blocked:
            gated.linear.y = 0.0
            rospy.logerr_throttle(
                0.5,
                "PARKING_LATERAL_GUARD blocked outward motion displacement=%.3fm limit=%.3fm",
                lateral, self.lateral_limit)
        super(LateralGuardParkingSafetyMonitor, self).cb_cmd(gated)


if __name__ == "__main__":
    try:
        LateralGuardParkingSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
