#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V10 safety plus a strictly gated straight recovery reverse command."""

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_parking_v10 import (
    MissionHolonomicSafetyMonitor,
)


class RecoveryReverseSafetyMonitor(MissionHolonomicSafetyMonitor):
    def __init__(self):
        super(RecoveryReverseSafetyMonitor, self).__init__()
        self.recovery_reverse_max = float(rospy.get_param(
            "~room_recovery_reverse_max_mps", 0.060))
        self.recovery_rear_need = float(rospy.get_param(
            "~room_recovery_rear_clear_m", 0.31))
        rospy.logwarn(
            "MISSION_SAFETY_V11 recovery_reverse=%.3f rear_need=%.3f",
            self.recovery_reverse_max, self.recovery_rear_need)

    def cb_cmd(self, msg):
        if self.parking_mode_active() or msg.linear.x >= -1.0e-4:
            super(RecoveryReverseSafetyMonitor, self).cb_cmd(msg)
            return
        if not self.enabled or not self.scan_seen or not self.scan_fresh():
            self.publish_zero("ROOM_RECOVERY_SCAN_UNAVAILABLE")
            return
        if abs(msg.linear.y) > 1.0e-4 or abs(msg.angular.z) > 0.02:
            self.publish_zero("ROOM_RECOVERY_NOT_STRAIGHT")
            rospy.logerr_throttle(
                0.5, "ROOM_RECOVERY_NOT_STRAIGHT y=%.3f wz=%.3f",
                msg.linear.y, msg.angular.z)
            return
        if self.min_rear < self.recovery_rear_need:
            self.publish_zero("ROOM_RECOVERY_REAR_BLOCKED")
            rospy.logerr_throttle(
                0.5, "ROOM_RECOVERY_REAR_BLOCKED rear=%.3f need=%.3f",
                self.min_rear, self.recovery_rear_need)
            return
        output = Twist()
        output.linear.x = -min(
            abs(msg.linear.x), self.recovery_reverse_max)
        self.pub_cmd.publish(output)
        rospy.logwarn_throttle(
            0.35, "ROOM_RECOVERY_REVERSE_ALLOWED rear=%.3f cmd_x=%.3f",
            self.min_rear, output.linear.x)


if __name__ == "__main__":
    try:
        RecoveryReverseSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
