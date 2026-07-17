#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parking V9 plus directional holonomic safety outside parking mode."""

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_parking_v5 import clamp
from global_first_safety_monitor_parking_v9 import (
    ControlledRetreatParkingSafetyMonitor,
)


class MissionHolonomicSafetyMonitor(ControlledRetreatParkingSafetyMonitor):
    def __init__(self):
        self.normal_max_lateral = 0.06
        super(MissionHolonomicSafetyMonitor, self).__init__()
        self.normal_max_lateral = float(rospy.get_param(
            "~normal_max_lateral_speed", 0.06))
        rospy.logwarn(
            "MISSION_HOLONOMIC_SAFETY_V10 normal_lateral_max=%.3f",
            self.normal_max_lateral)

    def cb_cmd(self, msg):
        if self.parking_mode_active():
            super(MissionHolonomicSafetyMonitor, self).cb_cmd(msg)
            return

        if not self.enabled:
            self.pub_cmd.publish(msg)
            return
        if not self.scan_seen or not self.scan_fresh():
            self.publish_zero("MISSION_SCAN_UNAVAILABLE")
            return

        requested_xz = (abs(msg.linear.x) > 1.0e-4 or
                        abs(msg.angular.z) > 1.0e-4)
        if requested_xz:
            vx, wz, action = self.compute(msg.linear.x, msg.angular.z)
        else:
            vx, wz, action = 0.0, 0.0, "LATERAL_ONLY"

        vy = clamp(float(msg.linear.y),
                   -self.normal_max_lateral,
                   self.normal_max_lateral)
        if vy > 0.0:
            if self.min_left <= self.side_stop:
                vy = 0.0
                action = "LATERAL_LEFT_STOP"
            elif self.min_left < self.side_slow:
                ratio = clamp(
                    (self.min_left - self.side_stop) /
                    max(self.side_slow - self.side_stop, 1.0e-3),
                    0.20, 1.0)
                vy *= ratio
                action = "LATERAL_LEFT_SLOW"
        elif vy < 0.0:
            if self.min_right <= self.side_stop:
                vy = 0.0
                action = "LATERAL_RIGHT_STOP"
            elif self.min_right < self.side_slow:
                ratio = clamp(
                    (self.min_right - self.side_stop) /
                    max(self.side_slow - self.side_stop, 1.0e-3),
                    0.20, 1.0)
                vy *= ratio
                action = "LATERAL_RIGHT_SLOW"

        out = Twist()
        out.linear.x = vx
        out.linear.y = vy
        out.angular.z = wz
        self.pub_cmd.publish(out)
        if action not in ("CLEAR", "LATERAL_ONLY"):
            rospy.logwarn_throttle(
                0.7,
                "MissionSafetyV10 %s in=(%.3f,%.3f,%.3f) "
                "out=(%.3f,%.3f,%.3f)",
                action, msg.linear.x, msg.linear.y, msg.angular.z,
                out.linear.x, out.linear.y, out.angular.z)


if __name__ == "__main__":
    try:
        MissionHolonomicSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
