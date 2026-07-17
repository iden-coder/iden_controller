#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Stable dynamic route with a room-only rectangular swept-footprint guard.

The inherited first-stage controller is passed through unchanged.  Starting
at the d1 route stage, each turning command is rolled forward against current
laser surfaces using the real oriented rectangular footprint.  The guard may
only reduce forward/turn speed; it never reverses, changes turn direction, or
introduces lateral motion.
"""

import math

import rospy

from front_first_stable_dynamic_route_test_v1 import (
    FrontFirstStableDynamicRouteTest,
)


INF = float("inf")


class FrontFirstRoomRectSweepRouteTest(FrontFirstStableDynamicRouteTest):
    def __init__(self):
        # The parent starts a ROS timer before this constructor completes.
        self.rect_guard_ready = False
        self.rect_scan_seen = False
        self.rect_scan_points = []
        self.rect_scan_time = rospy.Time(0)
        self.rect_guard_was_limiting = False
        super(FrontFirstRoomRectSweepRouteTest, self).__init__()

        self.rect_guard_enabled = bool(rospy.get_param(
            "~rect_sweep_guard_enabled", True))
        self.robot_half_length = float(rospy.get_param(
            "~rect_robot_half_length_m", 0.171))
        self.robot_half_width = float(rospy.get_param(
            "~rect_robot_half_width_m", 0.128))
        self.footprint_margin = float(rospy.get_param(
            "~rect_footprint_margin_m", 0.015))
        self.guard_horizon = float(rospy.get_param(
            "~rect_guard_horizon_s", 0.80))
        self.guard_dt = float(rospy.get_param(
            "~rect_guard_sim_dt_s", 0.05))
        self.guard_scan_range = float(rospy.get_param(
            "~rect_guard_scan_range_m", 0.90))
        self.guard_scan_stride = max(1, int(rospy.get_param(
            "~rect_guard_scan_stride", 2)))
        self.guard_scan_timeout = float(rospy.get_param(
            "~rect_guard_scan_timeout_s", 0.45))
        self.guard_min_turn_rate = float(rospy.get_param(
            "~rect_guard_min_turn_rate_rps", 0.08))
        self.guard_worsen_allowance = float(rospy.get_param(
            "~rect_guard_worsen_allowance_m", 0.003))
        self.guard_log_period = float(rospy.get_param(
            "~rect_guard_log_period_s", 0.35))
        self.turn_scales = self._scale_list(rospy.get_param(
            "~rect_guard_turn_scales", [0.75, 0.50, 0.25, 0.0]))
        self.speed_scales = self._scale_list(rospy.get_param(
            "~rect_guard_speed_scales", [1.0, 0.75, 0.50, 0.25, 0.0]))

        self.rect_guard_ready = True
        corner_radius = math.hypot(
            self.robot_half_length + self.footprint_margin,
            self.robot_half_width + self.footprint_margin)
        rospy.logwarn(
            "ROOM_RECT_SWEEP_READY activation=%s footprint=(%.3fx%.3f)m "
            "margin=%.3fm corner_radius=%.3fm horizon=%.2fs first_stage_unchanged=true",
            self.dynamic_activation_point,
            2.0 * self.robot_half_length, 2.0 * self.robot_half_width,
            self.footprint_margin, corner_radius, self.guard_horizon)

    @staticmethod
    def _scale_list(raw):
        values = []
        if isinstance(raw, (list, tuple)):
            for value in raw:
                try:
                    values.append(max(0.0, min(1.0, float(value))))
                except (TypeError, ValueError):
                    pass
        values.append(0.0)
        return sorted(set(values), reverse=True)

    def cb_scan(self, msg):
        super(FrontFirstRoomRectSweepRouteTest, self).cb_scan(msg)
        if not self.rect_guard_ready:
            return
        self.rect_scan_points = self._scan_points_in_base(msg)
        self.rect_scan_time = rospy.Time.now()
        self.rect_scan_seen = True

    def _scan_points_in_base(self, msg):
        frame_id = msg.header.frame_id or self.base_frame
        tx = 0.0
        ty = 0.0
        yaw = 0.0
        if frame_id != self.base_frame:
            try:
                trans, rot = self.tf_listener.lookupTransform(
                    self.base_frame, frame_id, rospy.Time(0))
                tx, ty = trans[0], trans[1]
                yaw = self.tf.transformations.euler_from_quaternion(rot)[2]
            except Exception:
                # The configured base-to-laser transform should normally be
                # available.  An empty point set leaves final authority with
                # the existing scan safety monitor.
                return []

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        points = []
        for index in range(0, len(msg.ranges), self.guard_scan_stride):
            distance = msg.ranges[index]
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            if distance > self.guard_scan_range:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)
            points.append((tx + cos_yaw * lx - sin_yaw * ly,
                           ty + sin_yaw * lx + cos_yaw * ly))
        return points

    def _footprint_gap(self, x, y, theta):
        if not self.rect_scan_points:
            return INF
        half_length = self.robot_half_length + self.footprint_margin
        half_width = self.robot_half_width + self.footprint_margin
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        best = INF
        for obstacle_x, obstacle_y in self.rect_scan_points:
            dx = obstacle_x - x
            dy = obstacle_y - y
            local_x = cos_theta * dx + sin_theta * dy
            local_y = -sin_theta * dx + cos_theta * dy
            outside_x = max(abs(local_x) - half_length, 0.0)
            outside_y = max(abs(local_y) - half_width, 0.0)
            if outside_x > 0.0 or outside_y > 0.0:
                gap = math.hypot(outside_x, outside_y)
            else:
                gap = -min(half_length - abs(local_x),
                           half_width - abs(local_y))
            if gap < best:
                best = gap
        return best

    def _trajectory_result(self, vx, wz):
        if not self.rect_scan_points:
            return True, INF
        start_gap = self._footprint_gap(0.0, 0.0, 0.0)
        minimum_gap = start_gap
        x = 0.0
        y = 0.0
        theta = 0.0
        steps = max(3, int(math.ceil(self.guard_horizon / self.guard_dt)))
        for _ in range(steps):
            x += math.cos(theta) * vx * self.guard_dt
            y += math.sin(theta) * vx * self.guard_dt
            theta += wz * self.guard_dt
            gap = self._footprint_gap(x, y, theta)
            minimum_gap = min(minimum_gap, gap)
            if start_gap > 0.0:
                if gap <= 0.0:
                    return False, minimum_gap
            elif gap < start_gap - self.guard_worsen_allowance:
                return False, minimum_gap
        return True, minimum_gap

    def _guard_scan_fresh(self):
        if not self.rect_scan_seen:
            return False
        return ((rospy.Time.now() - self.rect_scan_time).to_sec() <=
                self.guard_scan_timeout)

    def _closest_safe_command(self, requested_vx, requested_wz):
        safe, minimum_gap = self._trajectory_result(
            requested_vx, requested_wz)
        if safe:
            return requested_vx, requested_wz, minimum_gap, False

        best = None
        speed_base = max(abs(requested_vx), 0.05)
        turn_base = max(abs(requested_wz), self.guard_min_turn_rate)
        for turn_scale in self.turn_scales:
            candidate_wz = requested_wz * turn_scale
            for speed_scale in self.speed_scales:
                candidate_vx = requested_vx * speed_scale
                safe, gap = self._trajectory_result(
                    candidate_vx, candidate_wz)
                if not safe:
                    continue
                speed_error = abs(requested_vx - candidate_vx) / speed_base
                turn_error = abs(requested_wz - candidate_wz) / turn_base
                # Preserve forward progress first, then as much of the desired
                # same-direction turn as the rectangular sweep allows.
                score = 0.62 * speed_error ** 2 + 0.38 * turn_error ** 2
                candidate = (score, -gap, candidate_vx, candidate_wz, gap)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            return 0.0, 0.0, minimum_gap, True
        return best[2], best[3], best[4], True

    def publish_cmd(self, vx, wz):
        if (not self.rect_guard_ready or not self.rect_guard_enabled or
                not self._dynamic_layer_active() or
                abs(wz) < self.guard_min_turn_rate or
                (abs(vx) < 1.0e-5 and abs(wz) < 1.0e-5)):
            super(FrontFirstRoomRectSweepRouteTest, self).publish_cmd(vx, wz)
            return

        if not self._guard_scan_fresh():
            self.last_cmd = (0.0, 0.0)
            rospy.logwarn_throttle(
                1.0, "ROOM_RECT_SWEEP_STOP reason=scan_unavailable")
            super(FrontFirstRoomRectSweepRouteTest, self).publish_cmd(0.0, 0.0)
            return

        safe_vx, safe_wz, minimum_gap, limited = self._closest_safe_command(
            max(0.0, vx), wz)
        if limited:
            self.last_cmd = (safe_vx, safe_wz)
            rospy.logwarn_throttle(
                self.guard_log_period,
                "ROOM_RECT_SWEEP_LIMIT in=(%.3f,%.3f) out=(%.3f,%.3f) "
                "min_gap=%.3f points=%d",
                vx, wz, safe_vx, safe_wz, minimum_gap,
                len(self.rect_scan_points))
        elif self.rect_guard_was_limiting:
            rospy.logwarn("ROOM_RECT_SWEEP_CLEAR")
        self.rect_guard_was_limiting = limited
        super(FrontFirstRoomRectSweepRouteTest, self).publish_cmd(
            safe_vx, safe_wz)


if __name__ == "__main__":
    FrontFirstRoomRectSweepRouteTest().spin()
