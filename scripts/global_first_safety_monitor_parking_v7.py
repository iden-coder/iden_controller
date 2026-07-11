#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Heading-gated parking safety: large yaw correction forbids advance."""

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_parking_v6 import (
    CommandGatedParkingSafetyMonitor,
)


class HeadingGatedParkingSafetyMonitor(CommandGatedParkingSafetyMonitor):
    def cb_cmd(self, msg):
        if not self.parking_mode_active():
            super(HeadingGatedParkingSafetyMonitor, self).cb_cmd(msg)
            return

        gated = Twist()
        gated.linear.x = msg.linear.x
        gated.linear.y = msg.linear.y
        gated.angular.z = msg.angular.z
        heading_gate_wz = float(rospy.get_param(
            "~parking_heading_gate_wz", 0.070))
        close_wall_m = float(rospy.get_param(
            "~parking_close_wall_m", 0.18))
        close_max_wz = float(rospy.get_param(
            "~parking_close_max_wz", 0.055))

        if abs(gated.angular.z) > heading_gate_wz:
            if gated.linear.x > 0.0:
                rospy.logerr_throttle(
                    0.5,
                    "PARKING_HEADING_GATE forward blocked: wz=%.3f wall=%.3f",
                    gated.angular.z, self.min_front)
            gated.linear.x = 0.0
            gated.linear.y = 0.0
        if self.min_front < close_wall_m:
            gated.angular.z = max(
                -close_max_wz, min(close_max_wz, gated.angular.z))

        super(HeadingGatedParkingSafetyMonitor, self).cb_cmd(gated)


if __name__ == "__main__":
    try:
        HeadingGatedParkingSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass

