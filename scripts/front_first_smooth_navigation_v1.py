#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Front-first continuous path tracking with light holonomic avoidance.

This controller follows the proven graph path with a continuous lookahead
law.  Normal tracking uses vx/wz only.  A small vy term is introduced only
when a real front laser obstruction requires it; no trajectory lattice can
declare an otherwise drivable narrow corridor invalid.
"""

import math

import rospy
from geometry_msgs.msg import Twist

from global_first_graph_nav_2249fcf import (
    RosGlobalFirstGraphNavigator,
    clamp,
    norm_angle,
)
from universal_omni_navigation_v1 import UniversalOmniNavigator


class FrontFirstSmoothNavigator(RosGlobalFirstGraphNavigator):
    # Reuse the validated map-content cache, not the universal local planner.
    load_static_map = UniversalOmniNavigator.load_static_map

    def __init__(self):
        self.front_first_ready = False
        self.requested_vy = 0.0
        self.last_vy = 0.0
        self.unstick_until = rospy.Time(0)
        self.unstick_sign = 1.0
        self.max_forward_cmd = 0.48
        self.max_lateral_cmd = 0.055
        self.max_turn_cmd = 0.72
        super(FrontFirstSmoothNavigator, self).__init__()

        self.cruise_speed = float(rospy.get_param(
            "~front_first_cruise_mps", 0.48))
        self.minimum_curve_speed = float(rospy.get_param(
            "~front_first_min_curve_mps", 0.065))
        self.max_forward_cmd = self.cruise_speed
        self.max_lateral_cmd = float(rospy.get_param(
            "~front_first_max_lateral_mps", 0.055))
        self.max_turn_cmd = float(rospy.get_param(
            "~front_first_max_turn_rps", 0.72))
        self.heading_gain = float(rospy.get_param(
            "~front_first_heading_gain", 1.45))
        self.cross_track_gain = float(rospy.get_param(
            "~front_first_cross_track_gain", 0.55))
        self.curve_slow_angle = math.radians(float(rospy.get_param(
            "~front_first_curve_slow_deg", 52.0)))

        self.avoid_influence = float(rospy.get_param(
            "~avoid_front_influence_m", 0.52))
        self.avoid_stop = float(rospy.get_param(
            "~avoid_front_stop_m", 0.20))
        self.lateral_trigger = float(rospy.get_param(
            "~avoid_lateral_trigger_m", 0.32))
        self.lateral_side_need = float(rospy.get_param(
            "~avoid_lateral_side_need_m", 0.18))
        self.avoid_turn_gain = float(rospy.get_param(
            "~avoid_turn_gain_rps", 0.42))
        self.wall_center_gain = float(rospy.get_param(
            "~wall_center_gain", 0.22))

        self.forward_accel = float(rospy.get_param(
            "~front_first_accel_mps2", 0.62))
        self.forward_decel = float(rospy.get_param(
            "~front_first_decel_mps2", 0.90))
        self.lateral_accel = float(rospy.get_param(
            "~front_first_lateral_accel_mps2", 0.32))
        self.turn_accel = float(rospy.get_param(
            "~front_first_turn_accel_rps2", 1.45))

        self.front_first_stuck_s = float(rospy.get_param(
            "~front_first_stuck_s", 4.2))
        self.unstick_duration_s = float(rospy.get_param(
            "~front_first_unstick_duration_s", 0.85))
        self.unstick_turn = float(rospy.get_param(
            "~front_first_unstick_turn_rps", 0.26))
        self.unstick_forward = float(rospy.get_param(
            "~front_first_unstick_forward_mps", 0.055))
        self.unstick_lateral = float(rospy.get_param(
            "~front_first_unstick_lateral_mps", 0.035))

        self.front_first_ready = True
        self.path_world = []
        self.path_index = 0
        self.last_plan_time = rospy.Time(0)
        rospy.logwarn(
            "FRONT_FIRST_SMOOTH_READY cruise=%.2f lateral_max=%.3f "
            "continuous_curves=true normal_vy_zero=true",
            self.cruise_speed, self.max_lateral_cmd)

    def select_target(self):
        speed = abs(self.last_cmd[0])
        original = self.lookahead_dist
        self.lookahead_dist = clamp(0.35 + 0.55 * speed, 0.36, 0.62)
        try:
            return super(FrontFirstSmoothNavigator, self).select_target()
        finally:
            self.lookahead_dist = original

    def compute_cmd(self, target):
        x, y, yaw = self.pose
        dx = target[0] - x
        dy = target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        heading_error = math.atan2(local_y, local_x)

        # Continuous front-first pure pursuit.  Even a large heading error
        # retains a small forward component so bends are drawn as curves.
        angle_ratio = clamp(abs(heading_error) /
                            max(self.curve_slow_angle, 1e-3), 0.0, 1.0)
        speed = self.cruise_speed * (1.0 - 0.78 * angle_ratio ** 1.35)
        speed = max(self.minimum_curve_speed, speed)
        goal_dist = self.distance_to_active_goal()
        if goal_dist < self.final_slow_radius:
            ratio = clamp(goal_dist / max(self.final_slow_radius, 1e-3),
                          0.0, 1.0)
            speed = min(speed, self.final_creep_speed +
                        ratio * (self.final_slow_speed - self.final_creep_speed))

        wz = (self.heading_gain * heading_error +
              self.cross_track_gain * local_y)
        vy = 0.0

        # Smooth reactive steering is active only for a front obstruction.
        # Side walls alone never create lateral motion.
        if self.front < self.avoid_influence:
            severity = clamp(
                (self.avoid_influence - self.front) /
                max(self.avoid_influence - self.avoid_stop, 1e-3),
                0.0, 1.0)
            sign = 1.0 if self.left >= self.right else -1.0
            wz += sign * self.avoid_turn_gain * severity
            speed *= 1.0 - 0.72 * severity
            speed = max(0.025, speed)
            chosen_side = self.left if sign > 0.0 else self.right
            if (self.front < self.lateral_trigger and
                    chosen_side > self.lateral_side_need):
                vy = sign * self.max_lateral_cmd * severity
        elif self.left < 0.38 and self.right < 0.38:
            # Mild centering affects steering only, never lateral velocity.
            wz += self.wall_center_gain * (self.right - self.left)

        if self.unstick_until > rospy.Time.now():
            speed = min(speed, self.unstick_forward)
            vy = self.unstick_sign * self.unstick_lateral
            wz += self.unstick_sign * self.unstick_turn

        self.requested_vy = clamp(vy,
                                  -self.max_lateral_cmd,
                                  self.max_lateral_cmd)
        return (clamp(speed, 0.0, self.max_forward_cmd),
                clamp(wz, -self.max_turn_cmd, self.max_turn_cmd))

    def apply_scan_guard(self, cmd):
        vx, wz = cmd
        if self.front <= self.avoid_stop:
            vx = 0.0
            sign = 1.0 if self.left >= self.right else -1.0
            wz = sign * max(abs(wz), 0.16)
            chosen_side = self.left if sign > 0.0 else self.right
            if chosen_side > self.lateral_side_need:
                self.requested_vy = sign * min(
                    self.max_lateral_cmd, 0.04)
            else:
                self.requested_vy = 0.0
        if self.requested_vy > 0.0 and self.left <= self.side_stop_m:
            self.requested_vy = 0.0
        if self.requested_vy < 0.0 and self.right <= self.side_stop_m:
            self.requested_vy = 0.0
        return vx, wz

    def apply_goal_approach_limit(self, cmd):
        # Speed is already continuously scaled in compute_cmd.
        return cmd

    def smooth_cmd(self, cmd):
        now = rospy.Time.now()
        dt = clamp((now - self.last_control_time).to_sec(),
                   1.0 / max(self.control_rate_hz, 1.0), 0.15)
        self.last_control_time = now
        target_vx, target_wz = cmd
        last_vx, last_wz = self.last_cmd
        accel = self.forward_accel if target_vx >= last_vx else self.forward_decel
        vx = clamp(target_vx, last_vx - accel * dt, last_vx + accel * dt)
        wz = clamp(target_wz,
                   last_wz - self.turn_accel * dt,
                   last_wz + self.turn_accel * dt)
        vy = clamp(self.requested_vy,
                   self.last_vy - self.lateral_accel * dt,
                   self.last_vy + self.lateral_accel * dt)
        if target_vx <= 1e-5:
            vx = 0.0
        if abs(self.requested_vy) <= 1e-5:
            vy = 0.0
        self.last_cmd = (vx, wz)
        self.last_vy = vy
        self.requested_vy = vy
        return vx, wz

    def check_goal(self):
        if (self.pose is not None and
                self.distance_to_active_goal() <= self.current_target_tolerance()):
            self.requested_vy = 0.0
            self.last_vy = 0.0
        return super(FrontFirstSmoothNavigator, self).check_goal()

    def begin_curve_unstick(self):
        self.unstick_sign = 1.0 if self.left >= self.right else -1.0
        self.unstick_until = rospy.Time.now() + rospy.Duration(
            self.unstick_duration_s)
        self.last_progress_time = rospy.Time.now()
        rospy.logwarn(
            "FRONT_FIRST_CURVE_UNSTICK sign=%+.0f duration=%.2fs",
            self.unstick_sign, self.unstick_duration_s)

    def control_loop(self, _event):
        if not self.front_first_ready:
            self.publish_zero("FRONT_FIRST_INITIALIZING")
            return
        if self.finished:
            self.publish_zero("FINISHED")
            return
        if self.grid is None or self.planner is None:
            self.publish_zero("NO_MAP")
            return
        if not self.pose_fresh() or not self.start_pose_ok():
            self.publish_zero("WAIT_POSE")
            return
        if not self.scan_fresh():
            self.publish_zero("NO_SCAN")
            return
        if not self.path_world and not self.plan_from_current_pose(
                "initial", force=False):
            return
        if self.check_goal():
            return
        self.update_progress()
        if ((rospy.Time.now() - self.last_progress_time).to_sec() >
                self.front_first_stuck_s and
                self.distance_to_active_goal() >
                self.current_target_tolerance() + 0.16):
            self.begin_curve_unstick()
        target = self.select_target()
        if target is None:
            self.path_world = []
            self.plan_from_current_pose("path exhausted", force=False)
            return
        cmd = self.compute_cmd(target)
        cmd = self.apply_scan_guard(cmd)
        cmd = self.smooth_cmd(cmd)
        rospy.logwarn_throttle(
            0.6,
            "FRONT_FIRST_CMD x=%.3f y=%.3f wz=%.3f front=%.3f",
            cmd[0], self.requested_vy, cmd[1], self.front)
        self.publish_cmd(cmd[0], cmd[1])

    def publish_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = clamp(vx, 0.0, self.max_forward_cmd)
        msg.linear.y = clamp(self.requested_vy,
                             -self.max_lateral_cmd,
                             self.max_lateral_cmd)
        msg.angular.z = clamp(wz, -self.max_turn_cmd, self.max_turn_cmd)
        self.cmd_pub.publish(msg)

    def publish_zero(self, reason):
        self.requested_vy = 0.0
        self.last_vy = 0.0
        super(FrontFirstSmoothNavigator, self).publish_zero(reason)


if __name__ == "__main__":
    FrontFirstSmoothNavigator().spin()
