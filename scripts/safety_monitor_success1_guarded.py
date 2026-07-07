#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool, SetBoolResponse


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class SafetyMonitorGuarded:
    def __init__(self):
        rospy.init_node("safety_monitor")

        self.stop_zone = rospy.get_param("~stop_zone", 0.16)
        self.side_stop_zone = rospy.get_param("~side_stop_zone", 0.09)
        self.slowdown_zone = rospy.get_param("~slowdown_zone", 0.46)
        self.side_slowdown_zone = rospy.get_param("~side_slowdown_zone", 0.28)
        self.slowdown_ratio = rospy.get_param("~slowdown_ratio", 0.38)
        self.monitor_front_angle_deg = rospy.get_param("~monitor_front_angle_deg", 38.0)
        self.monitor_side_angle_min = rospy.get_param("~monitor_side_angle_min", 35.0)
        self.monitor_side_angle_max = rospy.get_param("~monitor_side_angle_max", 75.0)
        self.monitor_rear_angle_deg = rospy.get_param("~monitor_rear_angle_deg", 35.0)
        self.min_angular_ratio = rospy.get_param("~min_angular_ratio", 0.20)
        self.max_stop_angular = abs(rospy.get_param("~max_stop_angular", 0.10))
        self.max_escape_reverse = abs(rospy.get_param("~max_escape_reverse", 0.05))
        self.allow_reverse = rospy.get_param("~allow_reverse", False)
        self.rear_stop_zone = rospy.get_param("~rear_stop_zone", 0.22)
        self.rear_slowdown_zone = rospy.get_param("~rear_slowdown_zone", 0.38)
        self.scan_timeout = rospy.get_param("~scan_timeout", 0.7)
        self.swap_left_right = rospy.get_param("~swap_left_right", False)
        self.enabled = rospy.get_param("~enabled", True)

        self.min_front_dist = float("inf")
        self.min_left_dist = float("inf")
        self.min_right_dist = float("inf")
        self.min_rear_dist = float("inf")
        self.scan_received = False
        self.last_scan_time = rospy.Time(0)

        self.sub_scan = rospy.Subscriber(
            rospy.get_param("~scan_topic", "/scan"),
            LaserScan, self.cb_scan, queue_size=1)
        self.sub_cmd = rospy.Subscriber(
            rospy.get_param("~cmd_vel_in_topic", "/cmd_vel_raw"),
            Twist, self.cb_cmd_vel, queue_size=1)
        self.pub_cmd = rospy.Publisher(
            rospy.get_param("~cmd_vel_out_topic", "/cmd_vel"),
            Twist, queue_size=1)
        rospy.Service("~toggle", SetBool, self.cb_toggle)

        rospy.logwarn(
            "SafetyMonitorGuarded started: stop=%.2f side=%.2f rear=%.2f slow=%.2f swap_lr=%s",
            self.stop_zone, self.side_stop_zone, self.rear_stop_zone, self.slowdown_zone,
            str(self.swap_left_right))

    def cb_toggle(self, req):
        self.enabled = req.data
        rospy.logwarn("SafetyMonitorGuarded: %s", "ENABLED" if self.enabled else "DISABLED")
        return SetBoolResponse(success=True, message="OK")

    def scan_min_in_range(self, scan_msg, angle_min_deg, angle_max_deg):
        lo = math.radians(angle_min_deg)
        hi = math.radians(angle_max_deg)
        best = float("inf")
        for i, value in enumerate(scan_msg.ranges):
            if math.isnan(value) or math.isinf(value):
                continue
            if not (scan_msg.range_min <= value <= scan_msg.range_max):
                continue
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            if lo <= angle <= hi and value < best:
                best = value
        return best

    def cb_scan(self, msg):
        self.last_scan_time = rospy.Time.now()
        self.scan_received = True
        half = self.monitor_front_angle_deg
        left = self.scan_min_in_range(
            msg, self.monitor_side_angle_min, self.monitor_side_angle_max)
        right = self.scan_min_in_range(
            msg, -self.monitor_side_angle_max, -self.monitor_side_angle_min)
        if self.swap_left_right:
            left, right = right, left
        self.min_front_dist = self.scan_min_in_range(msg, -half, half)
        self.min_left_dist = left
        self.min_right_dist = right
        rear_half = self.monitor_rear_angle_deg
        rear_a = self.scan_min_in_range(msg, 180.0 - rear_half, 180.0)
        rear_b = self.scan_min_in_range(msg, -180.0, -180.0 + rear_half)
        self.min_rear_dist = min(rear_a, rear_b)

    def scan_is_fresh(self):
        if not self.scan_received:
            return False
        return (rospy.Time.now() - self.last_scan_time).to_sec() <= self.scan_timeout

    def zero(self, reason):
        return 0.0, 0.0, reason

    def compute_action(self, linear_x, angular_z):
        if not self.enabled:
            return linear_x, angular_z, "DISABLED"
        if not self.scan_received:
            return self.zero("NO_SCAN")
        if not self.scan_is_fresh():
            return self.zero("SCAN_TIMEOUT")

        f = self.min_front_dist
        l = self.min_left_dist
        r = self.min_right_dist
        rear = self.min_rear_dist
        lx = linear_x
        wz = angular_z

        if lx < 0.0:
            if not self.allow_reverse:
                lx = 0.0
                return lx, wz, "NO_REVERSE"
            if rear < self.rear_stop_zone:
                return 0.0, 0.0, "STOP_REAR"
            if rear < self.rear_slowdown_zone:
                t = (rear - self.rear_stop_zone) / max(
                    self.rear_slowdown_zone - self.rear_stop_zone, 1e-3)
                ratio = clamp(t, 0.0, 1.0)
                return lx * ratio, wz * max(0.35, ratio), "REAR_SLOW"

        if f < self.stop_zone:
            limited_wz = clamp(wz * self.min_angular_ratio,
                               -self.max_stop_angular, self.max_stop_angular)
            if l < self.side_stop_zone or r < self.side_stop_zone:
                return self.zero("STOP_ALL_CLOSE")
            if lx < 0.0:
                return max(lx, -self.max_escape_reverse), limited_wz, "ESCAPE_REVERSE"
            return 0.0, limited_wz, "STOP_FRONT"

        if l < self.side_stop_zone or r < self.side_stop_zone:
            if lx > 0.0:
                lx = min(lx, 0.03)
            if wz > 0.0 and l < self.side_stop_zone:
                wz = 0.0
            if wz < 0.0 and r < self.side_stop_zone:
                wz = 0.0
            return lx, wz, "SIDE_TOO_CLOSE"

        if lx > 0.0 and f < self.slowdown_zone:
            t = (f - self.stop_zone) / max(self.slowdown_zone - self.stop_zone, 1e-3)
            ratio = self.slowdown_ratio + (1.0 - self.slowdown_ratio) * clamp(t, 0.0, 1.0)
            lx *= ratio
            wz *= max(0.35, ratio)
            return lx, wz, "FRONT_SLOW"

        if abs(wz) > 0.08:
            side = l if wz > 0.0 else r
            if side < self.side_slowdown_zone:
                t = (side - self.side_stop_zone) / max(
                    self.side_slowdown_zone - self.side_stop_zone, 1e-3)
                ratio = clamp(t, 0.18, 1.0)
                return lx * max(0.35, ratio), wz * ratio, "TURN_SIDE_SLOW"

        side_min = min(l, r)
        if lx > 0.0 and side_min < self.side_slowdown_zone:
            t = (side_min - self.side_stop_zone) / max(
                self.side_slowdown_zone - self.side_stop_zone, 1e-3)
            ratio = clamp(t, 0.30, 1.0)
            return lx * ratio, wz * max(0.50, ratio), "SIDE_SLOW"

        return lx, wz, "CLEAR"

    def cb_cmd_vel(self, msg):
        new_lx, new_wz, action = self.compute_action(msg.linear.x, msg.angular.z)
        out = Twist()
        out.linear.x = new_lx
        out.linear.y = msg.linear.y if action in ("CLEAR", "DISABLED") else 0.0
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = new_wz
        self.pub_cmd.publish(out)

        if action not in ("CLEAR", "DISABLED"):
            rospy.logwarn_throttle(
                0.8,
                "SafetyMonitorGuarded: %s | f=%.2f l=%.2f r=%.2f rear=%.2f | in=(%.3f, %.3f) out=(%.3f, %.3f)",
                action, self.min_front_dist, self.min_left_dist, self.min_right_dist,
                self.min_rear_dist,
                msg.linear.x, msg.angular.z, out.linear.x, out.angular.z)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        SafetyMonitorGuarded().run()
    except rospy.ROSInterruptException:
        pass
