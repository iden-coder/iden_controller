#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Continuous-scan safety with a heartbeat-gated parking handoff."""

import math

import rospy
from geometry_msgs.msg import Twist

from global_first_safety_monitor_omni_v1 import DirectionalOmniSafetyMonitor
from two_stage_profile_safety_scan_v2 import (
    ContinuousScanTwoStageSafetyMonitor,
)


class ContinuousScanParkingSafetyMonitor(
        ContinuousScanTwoStageSafetyMonitor):
    def __init__(self):
        self.close_cone_block_since = None
        self.recovery_until = rospy.Time(0)
        self.recovery_cooldown_until = rospy.Time(0)
        # The parent constructor creates subscribers. Keep complete defaults in
        # place in case a command callback arrives before parameter loading.
        self.recovery_trigger_m = 0.17
        self.recovery_release_m = 0.27
        self.recovery_delay_s = 0.9
        self.recovery_speed = 0.055
        self.recovery_duration_s = 1.8
        self.recovery_rear_start_m = 0.31
        self.recovery_rear_abort_m = 0.18
        self.recovery_cooldown_s = 1.5
        self.recovery_lateral_speed = 0.045
        self.recovery_lateral_side_clear = 0.30
        self.recovery_lateral_side_bias = 0.08
        self.recovery_lateral_cmd = 0.0
        super(ContinuousScanParkingSafetyMonitor, self).__init__()
        self.recovery_trigger_m = float(rospy.get_param(
            "~close_cone_recovery_trigger_m", 0.17))
        self.recovery_release_m = float(rospy.get_param(
            "~close_cone_recovery_release_m", 0.27))
        self.recovery_delay_s = float(rospy.get_param(
            "~close_cone_recovery_delay_s", 0.9))
        self.recovery_speed = abs(float(rospy.get_param(
            "~close_cone_recovery_speed_mps", 0.055)))
        self.recovery_duration_s = float(rospy.get_param(
            "~close_cone_recovery_duration_s", 1.8))
        self.recovery_rear_start_m = float(rospy.get_param(
            "~close_cone_recovery_rear_start_m", 0.31))
        self.recovery_rear_abort_m = float(rospy.get_param(
            "~close_cone_recovery_rear_abort_m", 0.18))
        self.recovery_cooldown_s = float(rospy.get_param(
            "~close_cone_recovery_cooldown_s", 1.5))
        self.recovery_lateral_speed = abs(float(rospy.get_param(
            "~close_cone_recovery_lateral_mps", 0.045)))
        self.recovery_lateral_side_clear = float(rospy.get_param(
            "~close_cone_recovery_side_clear_m", 0.30))
        self.recovery_lateral_side_bias = abs(float(rospy.get_param(
            "~close_cone_recovery_side_bias_m", 0.08)))
        rospy.logwarn(
            "CLOSE_CONE_RECOVERY_READY trigger=%.2fm reverse=%.3fmps/%.1fs "
            "lateral=%.3fmps side_clear=%.2fm rear_start=%.2fm",
            self.recovery_trigger_m, self.recovery_speed,
            self.recovery_duration_s, self.recovery_lateral_speed,
            self.recovery_lateral_side_clear, self.recovery_rear_start_m)

    def _select_recovery_lateral(self, msg):
        if abs(msg.linear.y) > 1.0e-4:
            side_clear = (self.min_left if msg.linear.y > 0.0
                          else self.min_right)
            if side_clear >= self.recovery_lateral_side_clear:
                return math.copysign(self.recovery_lateral_speed,
                                     msg.linear.y)
            return 0.0

        left_clear = self.min_left
        right_clear = self.min_right
        bias = self.recovery_lateral_side_bias
        if (left_clear >= self.recovery_lateral_side_clear and
                math.isfinite(right_clear) and
                left_clear >= right_clear + bias):
            return self.recovery_lateral_speed
        if (right_clear >= self.recovery_lateral_side_clear and
                math.isfinite(left_clear) and
                right_clear >= left_clear + bias):
            return -self.recovery_lateral_speed
        return 0.0

    def _publish_recovery_reverse(self, now):
        if self.min_rear < self.recovery_rear_abort_m:
            self.recovery_until = rospy.Time(0)
            self.recovery_lateral_cmd = 0.0
            self.recovery_cooldown_until = (
                now + rospy.Duration(self.recovery_cooldown_s))
            self.pub_cmd.publish(Twist())
            rospy.logerr_throttle(
                0.5, "CLOSE_CONE_RECOVERY_REAR_ABORT rear=%.3f",
                self.min_rear)
            return
        output = Twist()
        output.linear.x = -self.recovery_speed
        side_clear = (self.min_left if self.recovery_lateral_cmd > 0.0
                      else self.min_right)
        if (abs(self.recovery_lateral_cmd) > 1.0e-4 and
                side_clear >= self.recovery_lateral_side_clear):
            output.linear.y = self.recovery_lateral_cmd
        self.pub_cmd.publish(output)
        rospy.logwarn_throttle(
            0.30,
            "CLOSE_CONE_RECOVERY_REVERSING cone_front=%.3f rear=%.3f "
            "side=%.3f cmd=(-%.3f,%.3f)",
            self.semantic_cone_front, self.min_rear, side_clear,
            self.recovery_speed, output.linear.y)

    def cb_cmd(self, msg):
        if self.parking_passthrough and self.parking_mode_active():
            # Skip cone/wall semantic reinterpretation only while the parking
            # controller is publishing its close-mode heartbeat. The inherited
            # indoor layer performs the actual gated passthrough.
            DirectionalOmniSafetyMonitor.cb_cmd(self, msg)
            return
        now = rospy.Time.now()
        if self.recovery_until > now:
            self._publish_recovery_reverse(now)
            return
        if self.recovery_until != rospy.Time(0):
            self.recovery_until = rospy.Time(0)
            self.recovery_cooldown_until = (
                now + rospy.Duration(self.recovery_cooldown_s))
            self.close_cone_block_since = None
            used_lateral = self.recovery_lateral_cmd
            self.recovery_lateral_cmd = 0.0
            self.pub_cmd.publish(Twist())
            rospy.logwarn(
                "CLOSE_CONE_RECOVERY_COMPLETE lateral=%.3f "
                "planner_replan_expected=true", used_lateral)
            return

        wants_motion = (msg.linear.x > 0.02 or
                        abs(msg.angular.z) > 0.05)
        close_cone = (
            self.room_profile_active and not self.continuous_scan_active and
            self.semantic_cone_front <= self.recovery_trigger_m)
        if close_cone and wants_motion:
            if self.close_cone_block_since is None:
                self.close_cone_block_since = now
            blocked_s = (now - self.close_cone_block_since).to_sec()
            if (blocked_s >= self.recovery_delay_s and
                    now >= self.recovery_cooldown_until and
                    self.min_rear >= self.recovery_rear_start_m):
                self.recovery_until = (
                    now + rospy.Duration(self.recovery_duration_s))
                self.recovery_lateral_cmd = self._select_recovery_lateral(msg)
                rospy.logwarn(
                    "CLOSE_CONE_RECOVERY_START blocked=%.2fs cone_front=%.3f "
                    "rear=%.3f lateral=%.3f diagonal=%s",
                    blocked_s, self.semantic_cone_front, self.min_rear,
                    self.recovery_lateral_cmd,
                    abs(self.recovery_lateral_cmd) > 1.0e-4)
                self._publish_recovery_reverse(now)
                return
        elif self.semantic_cone_front >= self.recovery_release_m:
            self.close_cone_block_since = None
        super(ContinuousScanParkingSafetyMonitor, self).cb_cmd(msg)


if __name__ == "__main__":
    try:
        ContinuousScanParkingSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
