#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool, SetBoolResponse


class SafetyMonitorSafeFast:
    def __init__(self):
        rospy.init_node('safety_monitor')

        self.stop_zone = rospy.get_param('~stop_zone', 0.11)
        self.side_stop_zone = rospy.get_param('~side_stop_zone', 0.055)
        self.slowdown_zone = rospy.get_param('~slowdown_zone', 0.32)
        self.slowdown_ratio = rospy.get_param('~slowdown_ratio', 0.72)
        self.monitor_front_angle_deg = rospy.get_param('~monitor_front_angle_deg', 35.0)
        self.monitor_side_angle_min = rospy.get_param('~monitor_side_angle_min', 35.0)
        self.monitor_side_angle_max = rospy.get_param('~monitor_side_angle_max', 65.0)
        self.min_angular_ratio = rospy.get_param('~min_angular_ratio', 0.35)
        self.max_escape_reverse = abs(rospy.get_param('~max_escape_reverse', 0.10))
        self.enabled = rospy.get_param('~enabled', True)

        self.min_front_dist = float('inf')
        self.min_left_dist = float('inf')
        self.min_right_dist = float('inf')
        self.scan_received = False
        self.last_scan_time = rospy.Time.now()

        self.sub_scan = rospy.Subscriber(
            rospy.get_param('~scan_topic', '/scan'),
            LaserScan, self.cb_scan, queue_size=1)
        self.sub_cmd = rospy.Subscriber(
            rospy.get_param('~cmd_vel_in_topic', '/cmd_vel_raw'),
            Twist, self.cb_cmd_vel, queue_size=1)
        self.pub_cmd = rospy.Publisher(
            rospy.get_param('~cmd_vel_out_topic', '/cmd_vel'),
            Twist, queue_size=1)

        rospy.Service('~toggle', SetBool, self.cb_toggle)
        rospy.loginfo(
            "SafetyMonitorSafeFast started: stop=%.2fm slow=%.2fm ratio=%.0f%%",
            self.stop_zone, self.slowdown_zone, self.slowdown_ratio * 100)

    def cb_toggle(self, req):
        self.enabled = req.data
        rospy.logwarn("SafetyMonitorSafeFast: %s", "ENABLED" if self.enabled else "DISABLED")
        return SetBoolResponse(success=True, message="OK")

    def scan_min_in_range(self, scan_msg, angle_min_deg, angle_max_deg):
        if not scan_msg or not scan_msg.ranges:
            return float('inf')
        min_rad = math.radians(angle_min_deg)
        max_rad = math.radians(angle_max_deg)
        best = float('inf')
        for i, r in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            if min_rad <= angle <= max_rad and scan_msg.range_min <= r <= scan_msg.range_max:
                best = min(best, r)
        return best

    def cb_scan(self, msg):
        self.last_scan_time = rospy.Time.now()
        self.scan_received = True
        half = self.monitor_front_angle_deg
        self.min_front_dist = self.scan_min_in_range(msg, -half, half)
        self.min_left_dist = self.scan_min_in_range(
            msg, -self.monitor_side_angle_max, -self.monitor_side_angle_min)
        self.min_right_dist = self.scan_min_in_range(
            msg, self.monitor_side_angle_min, self.monitor_side_angle_max)

    def compute_action(self, linear_x, angular_z):
        if not self.scan_received:
            return 0.0, 0.0, "NO_SCAN"
        if (rospy.Time.now() - self.last_scan_time).to_sec() > 1.0:
            return 0.0, 0.0, "SCAN_TIMEOUT"

        f = self.min_front_dist
        l = self.min_left_dist
        r = self.min_right_dist

        if f < self.stop_zone:
            limited_wz = angular_z * self.min_angular_ratio
            if l < self.side_stop_zone or r < self.side_stop_zone:
                return 0.0, 0.0, "STOP_ALL"
            if linear_x < 0.0:
                return max(linear_x, -self.max_escape_reverse), limited_wz, "ESCAPE_REVERSE"
            return 0.0, limited_wz, "STOP_FRONT"

        if abs(angular_z) > 0.1:
            if angular_z > 0 and l < self.side_stop_zone:
                return linear_x * 0.3, 0.0, "STOP_LEFT"
            if angular_z < 0 and r < self.side_stop_zone:
                return linear_x * 0.3, 0.0, "STOP_RIGHT"

        if f < self.slowdown_zone:
            t = (f - self.stop_zone) / max(self.slowdown_zone - self.stop_zone, 1e-3)
            ratio = self.slowdown_ratio + (1.0 - self.slowdown_ratio) * t
            ratio = max(0.0, min(1.0, ratio))
            return linear_x * ratio, angular_z * ratio, "SLOWDOWN"

        if abs(angular_z) > 0.3 and (l < self.slowdown_zone or r < self.slowdown_zone):
            side = min(l, r)
            t = (side - self.side_stop_zone) / max(self.slowdown_zone - self.side_stop_zone, 1e-3)
            t = max(0.2, min(1.0, t))
            return linear_x, angular_z * t, "SIDE_SLOW"

        return linear_x, angular_z, "CLEAR"

    def cb_cmd_vel(self, msg):
        if not self.enabled:
            self.pub_cmd.publish(msg)
            return

        new_lx, new_az, action = self.compute_action(msg.linear.x, msg.angular.z)
        out = Twist()
        out.linear.x = new_lx
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = new_az
        self.pub_cmd.publish(out)

        if action != "CLEAR":
            rospy.logwarn_throttle(
                1.0,
                "SafetyMonitorSafeFast: %s | f=%.2f l=%.2f r=%.2f | cmd->(%.3f, %.3f)",
                action, self.min_front_dist, self.min_left_dist, self.min_right_dist,
                out.linear.x, out.angular.z)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        SafetyMonitorSafeFast().run()
    except rospy.ROSInterruptException:
        pass
