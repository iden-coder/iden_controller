#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_srvs.srv import SetBool, SetBoolResponse


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class GlobalFirstSafetyMonitor:
    def __init__(self):
        rospy.init_node("global_first_safety_monitor")

        self.enabled = rospy.get_param("~enabled", True)
        self.cmd_vel_in_topic = rospy.get_param("~cmd_vel_in_topic", "/cmd_vel_raw")
        self.cmd_vel_out_topic = rospy.get_param("~cmd_vel_out_topic", "/cmd_vel")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")

        self.front_stop = rospy.get_param("~front_stop_m", 0.22)
        self.front_critical = rospy.get_param("~front_critical_m", 0.15)
        self.front_slow = rospy.get_param("~front_slow_m", 0.58)
        self.side_stop = rospy.get_param("~side_stop_m", 0.16)
        self.side_critical = rospy.get_param("~side_critical_m", 0.12)
        self.side_slow = rospy.get_param("~side_slow_m", 0.32)
        self.side_avoid = rospy.get_param("~side_avoid_m", 0.23)
        self.rear_stop = rospy.get_param("~rear_stop_m", 0.22)
        self.scan_timeout = rospy.get_param("~scan_timeout", 0.7)

        self.front_angle_deg = rospy.get_param("~front_angle_deg", 42.0)
        self.side_angle_min_deg = rospy.get_param("~side_angle_min_deg", 35.0)
        self.side_angle_max_deg = rospy.get_param("~side_angle_max_deg", 82.0)
        self.rear_angle_deg = rospy.get_param("~rear_angle_deg", 35.0)

        self.min_turn_wz = rospy.get_param("~min_turn_wz", 0.12)
        self.max_turn_wz = rospy.get_param("~max_turn_wz", 0.26)
        self.clear_slow_ratio = rospy.get_param("~clear_slow_ratio", 0.55)
        self.allow_reverse = rospy.get_param("~allow_reverse", False)
        self.swap_left_right = rospy.get_param("~swap_left_right", False)
        self.escape_enabled = rospy.get_param("~escape_enabled", True)
        self.escape_back_speed = abs(rospy.get_param("~escape_back_speed", 0.08))
        self.escape_back_time = rospy.get_param("~escape_back_time_s", 0.9)
        self.escape_cooldown = rospy.get_param("~escape_cooldown_s", 0.25)
        self.escape_rear_clear = rospy.get_param("~escape_rear_clear_m", 0.34)
        self.escape_front_release = rospy.get_param("~escape_front_release_m", 0.34)
        self.escape_turn_wz = abs(rospy.get_param("~escape_turn_wz", 0.08))

        self.min_front = float("inf")
        self.min_left = float("inf")
        self.min_right = float("inf")
        self.min_rear = float("inf")
        self.scan_seen = False
        self.last_scan_time = rospy.Time(0)
        self.escape_until = rospy.Time(0)
        self.last_escape_end = rospy.Time(0)

        self.pub_cmd = rospy.Publisher(self.cmd_vel_out_topic, Twist, queue_size=1)
        self.sub_scan = rospy.Subscriber(
            self.scan_topic, LaserScan, self.cb_scan, queue_size=1)
        self.sub_cmd = rospy.Subscriber(
            self.cmd_vel_in_topic, Twist, self.cb_cmd, queue_size=1)
        rospy.Service("~toggle", SetBool, self.cb_toggle)

        rospy.logwarn(
            "GlobalFirstSafetyMonitor started: front_stop=%.2f side_stop=%.2f critical=(%.2f, %.2f) escape=%s",
            self.front_stop, self.side_stop, self.front_critical, self.side_critical,
            str(self.escape_enabled))

    def cb_toggle(self, req):
        self.enabled = req.data
        return SetBoolResponse(success=True, message="OK")

    def scan_min(self, msg, lo_deg, hi_deg):
        lo = math.radians(lo_deg)
        hi = math.radians(hi_deg)
        best = float("inf")
        for i, value in enumerate(msg.ranges):
            if math.isnan(value) or math.isinf(value):
                continue
            if value < msg.range_min or value > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            if lo <= angle <= hi and value < best:
                best = value
        return best

    def cb_scan(self, msg):
        self.scan_seen = True
        self.last_scan_time = rospy.Time.now()
        half = self.front_angle_deg
        left = self.scan_min(msg, self.side_angle_min_deg, self.side_angle_max_deg)
        right = self.scan_min(msg, -self.side_angle_max_deg, -self.side_angle_min_deg)
        if self.swap_left_right:
            left, right = right, left
        rear_half = self.rear_angle_deg
        rear_a = self.scan_min(msg, 180.0 - rear_half, 180.0)
        rear_b = self.scan_min(msg, -180.0, -180.0 + rear_half)
        self.min_front = self.scan_min(msg, -half, half)
        self.min_left = left
        self.min_right = right
        self.min_rear = min(rear_a, rear_b)

    def scan_fresh(self):
        return (self.scan_seen and
                (rospy.Time.now() - self.last_scan_time).to_sec() <= self.scan_timeout)

    def zero(self, reason):
        return 0.0, 0.0, reason

    def turn_away_from_wall(self, wz):
        if self.min_left < self.side_critical and self.min_right < self.side_critical:
            return 0.0
        if self.min_left < self.side_critical:
            return -self.max_turn_wz
        if self.min_right < self.side_critical:
            return self.max_turn_wz
        preferred = self.max_turn_wz if self.min_left >= self.min_right else -self.max_turn_wz
        if abs(wz) < self.min_turn_wz or wz * preferred < 0.0:
            return preferred
        return clamp(wz, -self.max_turn_wz, self.max_turn_wz)

    def rear_clear_for_escape(self):
        return self.min_rear >= max(self.rear_stop, self.escape_rear_clear)

    def escape_turn(self):
        if self.escape_turn_wz <= 0.0:
            return 0.0
        if self.min_left < self.side_critical and self.min_right < self.side_critical:
            return 0.0
        if self.min_left < self.side_critical:
            return -min(self.escape_turn_wz, self.max_turn_wz)
        if self.min_right < self.side_critical:
            return min(self.escape_turn_wz, self.max_turn_wz)
        if self.min_left == float("inf") and self.min_right == float("inf"):
            return 0.0
        return min(self.escape_turn_wz, self.max_turn_wz) if self.min_left >= self.min_right else -min(self.escape_turn_wz, self.max_turn_wz)

    def escape_active(self, now):
        return (self.escape_until - now).to_sec() > 0.0

    def maybe_escape(self, reason, now):
        if not self.escape_enabled:
            return None
        if not self.rear_clear_for_escape():
            if self.escape_active(now):
                self.last_escape_end = now
            self.escape_until = rospy.Time(0)
            return None

        if self.escape_active(now):
            if self.min_front >= self.escape_front_release:
                self.last_escape_end = now
                self.escape_until = rospy.Time(0)
                return None
            return -self.escape_back_speed, self.escape_turn(), reason + "_ESCAPE"

        if (now - self.last_escape_end).to_sec() < self.escape_cooldown:
            return None
        self.escape_until = now + rospy.Duration(self.escape_back_time)
        return -self.escape_back_speed, self.escape_turn(), reason + "_ESCAPE"

    def compute(self, vx, wz):
        if not self.enabled:
            return vx, wz, "DISABLED"
        if not self.scan_seen:
            return self.zero("NO_SCAN")
        if not self.scan_fresh():
            return self.zero("SCAN_TIMEOUT")

        if vx < 0.0:
            if not self.allow_reverse:
                vx = 0.0
            elif not self.rear_clear_for_escape():
                return self.zero("STOP_REAR")
            else:
                vx = -min(abs(vx), self.escape_back_speed)

        now = rospy.Time.now()

        if self.min_front < self.front_critical:
            escape = self.maybe_escape("FRONT_CRITICAL", now)
            if escape is not None:
                return escape
            return 0.0, self.turn_away_from_wall(wz), "FRONT_CRITICAL"

        if self.min_left < self.side_critical or self.min_right < self.side_critical:
            safe_wz = self.turn_away_from_wall(wz)
            return 0.0, safe_wz, "SIDE_CRITICAL"

        if self.min_front < self.front_stop:
            escape = self.maybe_escape("STOP_FRONT", now)
            if escape is not None:
                return escape
            safe_wz = self.turn_away_from_wall(wz)
            return 0.0, safe_wz, "STOP_FRONT"

        if self.escape_active(now):
            self.last_escape_end = now
            self.escape_until = rospy.Time(0)

        if self.min_left < self.side_stop or self.min_right < self.side_stop:
            if self.min_left < self.side_stop and wz > 0.0:
                wz = -self.min_turn_wz
            if self.min_right < self.side_stop and wz < 0.0:
                wz = self.min_turn_wz
            return 0.0, clamp(wz, -self.max_turn_wz, self.max_turn_wz), "STOP_SIDE"

        if self.min_left < self.side_avoid and wz > 0.0:
            return 0.0, -self.min_turn_wz, "AVOID_LEFT"
        if self.min_right < self.side_avoid and wz < 0.0:
            return 0.0, self.min_turn_wz, "AVOID_RIGHT"

        if vx > 0.0 and self.min_front < self.front_slow:
            t = (self.min_front - self.front_stop) / max(self.front_slow - self.front_stop, 1e-3)
            ratio = self.clear_slow_ratio + (1.0 - self.clear_slow_ratio) * clamp(t, 0.0, 1.0)
            vx *= ratio
            wz *= max(0.55, ratio)
            return vx, wz, "FRONT_SLOW"

        side_min = min(self.min_left, self.min_right)
        if side_min < self.side_slow:
            t = (side_min - self.side_stop) / max(self.side_slow - self.side_stop, 1e-3)
            ratio = clamp(t, 0.45, 1.0)
            vx *= ratio
            return vx, wz, "SIDE_SLOW"

        return vx, wz, "CLEAR"

    def cb_cmd(self, msg):
        vx, wz, action = self.compute(msg.linear.x, msg.angular.z)
        out = Twist()
        out.linear.x = vx
        out.angular.z = wz
        self.pub_cmd.publish(out)
        if action not in ("CLEAR", "DISABLED"):
            rospy.logwarn_throttle(
                0.7,
                "GlobalFirstSafetyMonitor: %s | f=%.2f l=%.2f r=%.2f rear=%.2f | in=(%.3f, %.3f) out=(%.3f, %.3f)",
                action, self.min_front, self.min_left, self.min_right, self.min_rear,
                msg.linear.x, msg.angular.z, out.linear.x, out.angular.z)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        GlobalFirstSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
