#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Two-stage safety with non-duplicated protection during body-aware sweeps."""

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from global_first_safety_monitor_omni_v1 import DirectionalOmniSafetyMonitor
from two_stage_profile_safety_v1 import TwoStageProfileSafetyMonitor


class ContinuousScanTwoStageSafetyMonitor(TwoStageProfileSafetyMonitor):
    def __init__(self):
        self.continuous_scan_active = False
        super(ContinuousScanTwoStageSafetyMonitor, self).__init__()
        self.scan_cone_turn_hard_stop = float(rospy.get_param(
            "~continuous_scan_cone_turn_hard_stop_m", 0.10))
        self.scan_mode_sub = rospy.Subscriber(
            rospy.get_param("~continuous_scan_safety_topic",
                            "/continuous_scan/safety_mode"),
            Bool, self.cb_continuous_scan_mode, queue_size=1)
        rospy.logwarn(
            "CONTINUOUS_SCAN_SAFETY_READY rectangular_guard_primary=true "
            "semantic_emergency=%.2fm",
            self.scan_cone_turn_hard_stop)

    def cb_continuous_scan_mode(self, msg):
        active = bool(msg.data)
        if active == self.continuous_scan_active:
            return
        self.continuous_scan_active = active
        rospy.logwarn(
            "CONTINUOUS_SCAN_SAFETY_%s rectangular_guard_primary=%s",
            "ACTIVE" if active else "CLEAR", str(active).lower())

    def cb_cmd(self, msg):
        if not self.continuous_scan_active:
            super(ContinuousScanTwoStageSafetyMonitor, self).cb_cmd(msg)
            return

        # The navigator has already simulated the real rectangular footprint
        # over this angular command. Avoid applying the broad cone-turn slowdown
        # a second time; retain directional critical stops and a final semantic
        # emergency stop for an extremely close return.
        out = Twist()
        out.linear.x = msg.linear.x
        out.linear.y = msg.linear.y
        out.angular.z = msg.angular.z
        if (abs(out.angular.z) > 1.0e-4 and
                self.semantic_cone_turn <= self.scan_cone_turn_hard_stop):
            out.angular.z = 0.0
            rospy.logwarn_throttle(
                0.4,
                "CONTINUOUS_SCAN_SEMANTIC_EMERGENCY cone_turn=%.3f",
                self.semantic_cone_turn)
        DirectionalOmniSafetyMonitor.cb_cmd(self, out)


if __name__ == "__main__":
    try:
        ContinuousScanTwoStageSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
