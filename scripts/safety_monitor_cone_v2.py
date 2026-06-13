#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立安全监测节点 — 最后一道防线

架构: move_base → safety_monitor → robot

动作:
  STOP:     前方 < stop_zone 或侧面 < side_stop_zone → 停车
  SLOWDOWN: 前方 < slowdown_zone → 线性减速
  SIDE_SLOW: 转弯时侧面太近 → 限制角速度
"""

import rospy
import math
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String as StringMsg
from std_srvs.srv import SetBool, SetBoolResponse


class SafetyMonitor:
    def __init__(self):
        rospy.init_node('safety_monitor')

        self.stop_zone = rospy.get_param('~stop_zone', 0.15)
        self.side_stop_zone = rospy.get_param('~side_stop_zone', 0.08)
        self.slowdown_zone = rospy.get_param('~slowdown_zone', 0.50)
        self.slowdown_ratio = rospy.get_param('~slowdown_ratio', 0.40)
        self.max_linear = rospy.get_param('~max_linear', 0.30)
        self.max_reverse_linear = rospy.get_param('~max_reverse_linear', 0.035)
        self.max_angular = rospy.get_param('~max_angular', 0.90)
        self.side_slow_linear_ratio = rospy.get_param('~side_slow_linear_ratio', 0.60)
        self.monitor_front_angle_deg = rospy.get_param('~monitor_front_angle_deg', 45.0)
        self.monitor_side_angle_min = rospy.get_param('~monitor_side_angle_min', 35.0)
        self.monitor_side_angle_max = rospy.get_param('~monitor_side_angle_max', 65.0)
        self.min_angular_ratio = rospy.get_param('~min_angular_ratio', 0.30)
        self.cone_zone_param = rospy.get_param('~cone_zone_param', '/iden_controller/cone_zone')
        self.cone_steer_zone = rospy.get_param('~cone_steer_zone', 0.75)
        self.cone_turn_zone = rospy.get_param('~cone_turn_zone', 0.24)
        self.cone_max_linear = rospy.get_param('~cone_max_linear', 0.16)
        self.cone_min_linear = rospy.get_param('~cone_min_linear', 0.035)
        self.cone_steer_gain = rospy.get_param('~cone_steer_gain', 0.85)
        self.cone_max_angular = rospy.get_param('~cone_max_angular', 0.85)
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
            rospy.get_param('~cmd_vel_in_topic', '/move_base/cmd_vel'),
            Twist, self.cb_cmd_vel, queue_size=1)
        self.pub_cmd = rospy.Publisher(
            rospy.get_param('~cmd_vel_out_topic', '/cmd_vel'),
            Twist, queue_size=1)

        rospy.Service('~toggle', SetBool, self.cb_toggle)

        rospy.loginfo("SafetyMonitor 启动")
        rospy.loginfo("  stop=%.2fm slowdown=%.2fm ratio=%.0f%% front_angle=±%.0f°",
                      self.stop_zone, self.slowdown_zone,
                      self.slowdown_ratio * 100, self.monitor_front_angle_deg)

    def cone_mode(self):
        try:
            return bool(rospy.get_param(self.cone_zone_param, False))
        except Exception:
            return False

    def cb_toggle(self, req):
        self.enabled = req.data
        rospy.logwarn("SafetyMonitor: %s", "ENABLED" if self.enabled else "DISABLED")
        return SetBoolResponse(success=True, message="OK")

    def scan_min_in_range(self, scan_msg, angle_min_deg, angle_max_deg):
        if not scan_msg or len(scan_msg.ranges) == 0:
            return float('inf'), 0.0
        min_rad = math.radians(angle_min_deg)
        max_rad = math.radians(angle_max_deg)
        min_dist = float('inf')
        for i, r in enumerate(scan_msg.ranges):
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            if min_rad <= angle <= max_rad:
                if scan_msg.range_min <= r <= scan_msg.range_max:
                    if r < min_dist:
                        min_dist = r
        return min_dist, 0.0

    def cb_scan(self, msg):
        self.last_scan_time = rospy.Time.now()
        self.scan_received = True
        half = self.monitor_front_angle_deg
        self.min_front_dist, _ = self.scan_min_in_range(msg, -half, half)
        self.min_left_dist, _ = self.scan_min_in_range(
            msg, self.monitor_side_angle_min, self.monitor_side_angle_max)
        self.min_right_dist, _ = self.scan_min_in_range(
            msg, -self.monitor_side_angle_max, -self.monitor_side_angle_min)

    def compute_action(self, linear_x, angular_z):
        if not self.scan_received:
            return 0.0, 0.0, "NO_SCAN"
        if (rospy.Time.now() - self.last_scan_time).to_sec() > 1.0:
            return 0.0, 0.0, "SCAN_TIMEOUT"

        f = self.min_front_dist
        l = self.min_left_dist
        r = self.min_right_dist
        cone = self.cone_mode()

        if cone and linear_x < 0.0 and min(l, r) < self.cone_turn_zone:
            return 0.0, 0.0, "CONE_NO_REVERSE_SIDE"

        if cone and linear_x > 0.0 and f < self.cone_steer_zone:
            steer_dir = 1.0 if l > r else -1.0
            openness = abs(l - r)
            proximity = (self.cone_steer_zone - f) / max(self.cone_steer_zone - self.stop_zone, 1e-3)
            proximity = max(0.0, min(1.0, proximity))
            steer = steer_dir * self.cone_steer_gain * (0.45 + 0.55 * proximity)
            if openness < 0.04:
                steer *= 0.6
            steer = max(-self.cone_max_angular, min(self.cone_max_angular, steer))

            if f < self.stop_zone:
                if l < self.side_stop_zone and r < self.side_stop_zone:
                    return 0.0, 0.0, "CONE_STOP_ALL"
                if min(l, r) < self.cone_turn_zone:
                    return 0.0, 0.0, "CONE_STOP_TIGHT"
                return 0.0, steer, "CONE_TURN_IN_PLACE"

            ratio = (f - self.stop_zone) / max(self.cone_steer_zone - self.stop_zone, 1e-3)
            ratio = max(0.0, min(1.0, ratio))
            safe_v = self.cone_min_linear + (self.cone_max_linear - self.cone_min_linear) * ratio
            vx = min(linear_x, safe_v)
            wz = angular_z * 0.25 + steer * 0.75
            wz = max(-self.cone_max_angular, min(self.cone_max_angular, wz))
            return vx, wz, "CONE_STEER"

        # 前方 STOP
        if f < self.stop_zone:
            limited_wz = angular_z * self.min_angular_ratio
            if l < self.side_stop_zone or r < self.side_stop_zone:
                return 0.0, 0.0, "STOP_ALL"
            return 0.0, limited_wz, "STOP_FRONT"

        # 侧面 STOP (转弯中)
        if abs(angular_z) > 0.1:
            if angular_z > 0 and l < self.side_stop_zone:
                return linear_x * 0.3, 0.0, "STOP_LEFT"
            if angular_z < 0 and r < self.side_stop_zone:
                return linear_x * 0.3, 0.0, "STOP_RIGHT"

        # 前方 SLOWDOWN
        if f < self.slowdown_zone:
            t = (f - self.stop_zone) / (self.slowdown_zone - self.stop_zone)
            ratio = self.slowdown_ratio + (1.0 - self.slowdown_ratio) * t
            ratio = max(0.0, min(1.0, ratio))
            return linear_x * ratio, angular_z * ratio, "SLOWDOWN"

        # 侧面 SLOWDOWN
        if abs(angular_z) > 0.3:
            if l < self.slowdown_zone or r < self.slowdown_zone:
                side = min(l, r)
                t = (side - self.side_stop_zone) / (self.slowdown_zone - self.side_stop_zone)
                t = max(0.2, min(1.0, t))
                return linear_x * self.side_slow_linear_ratio, angular_z * t, "SIDE_SLOW"

        return linear_x, angular_z, "CLEAR"

    def cb_cmd_vel(self, msg):
        if not self.enabled:
            self.pub_cmd.publish(msg)
            return

        new_lx, new_az, action = self.compute_action(msg.linear.x, msg.angular.z)
        if self.cone_mode():
            max_forward = self.cone_max_linear
        else:
            max_forward = self.max_linear
        new_lx = max(-self.max_reverse_linear, min(max_forward, new_lx))
        new_az = max(-self.max_angular, min(self.max_angular, new_az))
        out = Twist()
        out.linear.x = new_lx
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = new_az
        self.pub_cmd.publish(out)

        if action != "CLEAR":
            rospy.logwarn_throttle(1.0,
                "SafetyMonitor: %s | f=%.2f l=%.2f r=%.2f | cmd→(%.3f, %.3f)",
                action, self.min_front_dist, self.min_left_dist, self.min_right_dist,
                out.linear.x, out.angular.z)

    def run(self):
        rospy.spin()


if __name__ == '__main__':
    try:
        monitor = SafetyMonitor()
        monitor.run()
    except rospy.ROSInterruptException:
        pass
