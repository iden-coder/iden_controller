#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parking safety that never creates motion from a zero command."""

import math

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_parking_v5 import HolonomicParkingSafetyMonitor
from global_first_safety_monitor_parking_v5 import clamp


class CommandGatedParkingSafetyMonitor(HolonomicParkingSafetyMonitor):
    def publish_zero(self, action):
        self.escape_until = rospy.Time(0)
        self.pub_cmd.publish(Twist())
        if action not in ("HOLD_ZERO", "CLEAR"):
            rospy.logwarn_throttle(0.7, "ParkingSafety: %s", action)

    def cb_cmd(self, msg):
        # A safety layer may restrict requested motion, but must never invent
        # reverse or turning motion after the controller has asked to stop.
        if (abs(msg.linear.x) < 1.0e-4 and
                abs(msg.linear.y) < 1.0e-4 and
                abs(msg.angular.z) < 1.0e-4):
            self.publish_zero("HOLD_ZERO")
            return

        parking_close = self.parking_mode_active()
        if not parking_close:
            super(CommandGatedParkingSafetyMonitor, self).cb_cmd(msg)
            return
        if not self.enabled:
            self.pub_cmd.publish(msg)
            return
        if not self.scan_seen or not self.scan_fresh():
            self.publish_zero("PARKING_SCAN_UNAVAILABLE")
            return

        vx = max(0.0, float(msg.linear.x))
        vy = clamp(float(msg.linear.y), -self.parking_max_lateral,
                   self.parking_max_lateral)
        wz = clamp(float(msg.angular.z), -self.max_turn_wz,
                   self.max_turn_wz)
        action = "PARKING_CLEAR"

        # Close-range side rays see the square rails and wall corners.  They
        # may block motion toward a genuinely close side, but never inject an
        # automatic turn that would rotate the chassis into the wall.
        side_hard = float(rospy.get_param("~parking_side_hard_m", 0.105))
        if self.min_left < side_hard:
            vy = min(vy, 0.0)
            if wz > 0.0:
                wz = 0.0
            action = "PARKING_LEFT_HARD"
        if self.min_right < side_hard:
            vy = max(vy, 0.0)
            if wz < 0.0:
                wz = 0.0
            action = "PARKING_RIGHT_HARD"

        if self.min_front < self.parking_front_critical:
            self.publish_zero("PARKING_FRONT_CRITICAL")
            return
        if vx > 0.0 and self.min_front <= self.parking_front_stop:
            vx = 0.0
            action = "PARKING_FRONT_STOP"
        elif vx > 0.0 and self.min_front < self.parking_front_slow:
            ratio = clamp(
                (self.min_front - self.parking_front_stop) /
                max(self.parking_front_slow - self.parking_front_stop, 1.0e-3),
                0.55, 1.0)
            vx *= ratio
            action = "PARKING_FRONT_SLOW"

        out = Twist()
        out.linear.x = vx
        out.linear.y = vy
        out.angular.z = wz
        self.pub_cmd.publish(out)
        if action != "PARKING_CLEAR":
            rospy.logwarn_throttle(
                0.7,
                "CommandGatedParkingSafety: %s f=%.3f l=%.3f r=%.3f in=(%.3f,%.3f,%.3f) out=(%.3f,%.3f,%.3f)",
                action, self.min_front, self.min_left, self.min_right,
                msg.linear.x, msg.linear.y, msg.angular.z,
                out.linear.x, out.linear.y, out.angular.z)


if __name__ == "__main__":
    try:
        CommandGatedParkingSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass

