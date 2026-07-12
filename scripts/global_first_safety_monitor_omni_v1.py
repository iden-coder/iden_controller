#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Directional safety filter that preserves holonomic navigation commands."""

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_indoor_v1 import IndoorProfileSafetyMonitor
from global_first_safety_monitor_parking_v5 import clamp


class DirectionalOmniSafetyMonitor(IndoorProfileSafetyMonitor):
    def __init__(self):
        super(DirectionalOmniSafetyMonitor, self).__init__()
        self.nav_max_forward = float(rospy.get_param(
            "~nav_max_forward_mps", 0.30))
        self.nav_max_lateral = float(rospy.get_param(
            "~nav_max_lateral_mps", 0.17))
        self.nav_max_angular = float(rospy.get_param(
            "~nav_max_angular_rps", 0.36))
        rospy.logwarn(
            "DIRECTIONAL_OMNI_SAFETY_READY limits=(%.2f,%.2f,%.2f)",
            self.nav_max_forward, self.nav_max_lateral,
            self.nav_max_angular)

    def cb_cmd(self, msg):
        if self.parking_passthrough and self.parking_mode_active():
            super(DirectionalOmniSafetyMonitor, self).cb_cmd(msg)
            return
        if not self.indoor_active:
            # Preserve the proven first-stage navigation and doorway behavior.
            # Holonomic directional filtering starts only after the pose has
            # crossed the large-room activation boundary.
            super(DirectionalOmniSafetyMonitor, self).cb_cmd(msg)
            return
        if not self.enabled:
            self.pub_cmd.publish(msg)
            return
        if not self.scan_seen or not self.scan_fresh():
            self.publish_zero("OMNI_SCAN_UNAVAILABLE")
            return

        vx = clamp(float(msg.linear.x), 0.0, self.nav_max_forward)
        vy = clamp(float(msg.linear.y),
                   -self.nav_max_lateral, self.nav_max_lateral)
        wz = clamp(float(msg.angular.z),
                   -self.nav_max_angular, self.nav_max_angular)
        action = "OMNI_CLEAR"

        # Restrict only motion toward an obstacle.  A wall beside the robot
        # must not suppress forward motion or a lateral command away from it.
        if vx > 0.0 and self.min_front <= self.front_critical:
            vx = 0.0
            action = "OMNI_FRONT_CRITICAL"
        elif vx > 0.0 and self.min_front <= self.front_stop:
            vx = 0.0
            action = "OMNI_FRONT_STOP"
        elif vx > 0.0 and self.min_front < self.front_slow:
            ratio = clamp(
                (self.min_front - self.front_stop) /
                max(self.front_slow - self.front_stop, 1.0e-3), 0.35, 1.0)
            vx *= ratio
            action = "OMNI_FRONT_SLOW"

        if vy > 0.0:
            if self.min_left <= self.side_critical:
                vy = 0.0
                action = "OMNI_LEFT_CRITICAL"
            elif self.min_left <= self.side_stop:
                vy = 0.0
                action = "OMNI_LEFT_STOP"
            elif self.min_left < self.side_slow:
                ratio = clamp(
                    (self.min_left - self.side_stop) /
                    max(self.side_slow - self.side_stop, 1.0e-3), 0.30, 1.0)
                vy *= ratio
                action = "OMNI_LEFT_SLOW"
        elif vy < 0.0:
            if self.min_right <= self.side_critical:
                vy = 0.0
                action = "OMNI_RIGHT_CRITICAL"
            elif self.min_right <= self.side_stop:
                vy = 0.0
                action = "OMNI_RIGHT_STOP"
            elif self.min_right < self.side_slow:
                ratio = clamp(
                    (self.min_right - self.side_stop) /
                    max(self.side_slow - self.side_stop, 1.0e-3), 0.30, 1.0)
                vy *= ratio
                action = "OMNI_RIGHT_SLOW"

        if wz > 0.0 and self.min_left <= self.side_critical:
            wz = 0.0
        if wz < 0.0 and self.min_right <= self.side_critical:
            wz = 0.0

        out = Twist()
        out.linear.x = vx
        out.linear.y = vy
        out.angular.z = wz
        self.pub_cmd.publish(out)
        if action != "OMNI_CLEAR":
            rospy.logwarn_throttle(
                0.6,
                "DirectionalOmniSafety: %s scan=(%.3f,%.3f,%.3f) "
                "in=(%.3f,%.3f,%.3f) out=(%.3f,%.3f,%.3f)",
                action, self.min_front, self.min_left, self.min_right,
                msg.linear.x, msg.linear.y, msg.angular.z,
                out.linear.x, out.linear.y, out.angular.z)


if __name__ == "__main__":
    try:
        DirectionalOmniSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
