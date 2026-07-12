#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parking safety with cardinal-aligned, rear-clearance-gated retreat."""

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_parking_v8 import (
    LateralGuardParkingSafetyMonitor,
)


class ControlledRetreatParkingSafetyMonitor(LateralGuardParkingSafetyMonitor):
    def cb_cmd(self, msg):
        parking_close = self.parking_mode_active()
        if not parking_close or msg.linear.x >= -1.0e-4:
            super(ControlledRetreatParkingSafetyMonitor, self).cb_cmd(msg)
            return

        if not self.enabled or not self.scan_seen or not self.scan_fresh():
            self.publish_zero("PARKING_RETREAT_SCAN_UNAVAILABLE")
            return
        max_reverse = float(rospy.get_param(
            "~parking_max_reverse_speed", 0.045))
        rear_clear = float(rospy.get_param(
            "~parking_retreat_rear_clear_m", 0.32))
        max_retreat_wz = float(rospy.get_param(
            "~parking_retreat_max_wz", 0.035))
        if self.min_rear < rear_clear:
            self.publish_zero("PARKING_RETREAT_REAR_BLOCKED")
            rospy.logerr_throttle(
                0.5, "PARKING_RETREAT_REAR_BLOCKED rear=%.3f need=%.3f",
                self.min_rear, rear_clear)
            return
        if (abs(msg.linear.y) > 1.0e-4 or
                abs(msg.angular.z) > max_retreat_wz):
            self.publish_zero("PARKING_RETREAT_NOT_STRAIGHT")
            rospy.logerr_throttle(
                0.5,
                "PARKING_RETREAT_NOT_STRAIGHT y=%.3f wz=%.3f",
                msg.linear.y, msg.angular.z)
            return

        out = Twist()
        out.linear.x = -min(abs(msg.linear.x), max_reverse)
        self.pub_cmd.publish(out)
        rospy.logwarn_throttle(
            0.6,
            "PARKING_CONTROLLED_RETREAT rear=%.3f cmd_x=%.3f",
            self.min_rear, out.linear.x)


if __name__ == "__main__":
    try:
        ControlledRetreatParkingSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass

