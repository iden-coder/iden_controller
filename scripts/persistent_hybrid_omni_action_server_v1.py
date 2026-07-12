#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Global graph navigation with a short-horizon holonomic trajectory layer.

The existing graph planner remains responsible for the route.  Inside the
large room this node samples vx/vy/wz commands, predicts their swept motion,
and chooses a collision-free command against the current laser scan.  This is
an ROS1/2-D adaptation of the trajectory-selection architecture used by RM
sentry navigation stacks; no 3-D point-cloud pipeline is required.
"""

import math

import rospy
from geometry_msgs.msg import Twist

from persistent_first_graph_action_server_v3 import (
    CompactConeExclusionActionServer,
)
from global_first_graph_nav_2249fcf import clamp, norm_angle


class HybridOmniActionServer(CompactConeExclusionActionServer):
    def __init__(self):
        # Parent constructors create a ROS timer.  Keep overridden callbacks
        # on the proven parent path until every omni parameter is initialized.
        self.omni_enabled = False
        self.local_vy = 0.0
        self.last_vy = 0.0
        self.local_blocked_since = None
        self.last_local_replan = rospy.Time(0)
        self.last_selected_side = 0
        super(HybridOmniActionServer, self).__init__()

        self.omni_enabled = bool(rospy.get_param("~omni_local_enabled", True))
        self.omni_control_horizon = float(rospy.get_param(
            "~omni_horizon_s", 1.10))
        self.omni_sim_dt = float(rospy.get_param("~omni_sim_dt_s", 0.11))
        self.omni_scan_range = float(rospy.get_param(
            "~omni_scan_range_m", 1.35))
        self.omni_scan_stride = max(1, int(rospy.get_param(
            "~omni_scan_stride", 3)))
        self.omni_max_forward = float(rospy.get_param(
            "~omni_max_forward_mps", 0.27))
        self.omni_max_lateral = float(rospy.get_param(
            "~omni_max_lateral_mps", 0.16))
        self.omni_max_angular = float(rospy.get_param(
            "~omni_max_angular_rps", 0.34))
        self.omni_min_speed = float(rospy.get_param(
            "~omni_min_translation_mps", 0.075))
        self.omni_lateral_accel = float(rospy.get_param(
            "~omni_lateral_accel_mps2", 0.42))

        # Scan points close to static occupied map cells are walls.  Other
        # points are treated as temporary obstacles, normally traffic cones.
        self.static_match_radius = float(rospy.get_param(
            "~omni_static_match_radius_m", 0.11))
        self.wall_hard_clearance = float(rospy.get_param(
            "~omni_wall_hard_clearance_m", 0.18))
        self.wall_preferred_clearance = float(rospy.get_param(
            "~omni_wall_preferred_clearance_m", 0.23))
        self.dynamic_hard_clearance = float(rospy.get_param(
            "~omni_dynamic_hard_clearance_m", 0.22))
        self.dynamic_preferred_clearance = float(rospy.get_param(
            "~omni_dynamic_preferred_clearance_m", 0.32))
        self.static_map_hard_clearance = float(rospy.get_param(
            "~omni_static_map_hard_clearance_m", 0.18))

        self.local_block_replan_s = float(rospy.get_param(
            "~omni_block_replan_s", 1.40))
        self.local_replan_interval_s = float(rospy.get_param(
            "~omni_replan_min_interval_s", 1.20))
        self.score_path_weight = float(rospy.get_param(
            "~omni_score_path_weight", 4.2))
        self.score_goal_weight = float(rospy.get_param(
            "~omni_score_goal_weight", 3.0))
        self.score_dynamic_weight = float(rospy.get_param(
            "~omni_score_dynamic_weight", 2.8))
        self.score_wall_weight = float(rospy.get_param(
            "~omni_score_wall_weight", 0.75))
        self.score_lateral_weight = float(rospy.get_param(
            "~omni_score_lateral_weight", 0.25))
        self.score_turn_weight = float(rospy.get_param(
            "~omni_score_turn_weight", 0.18))
        self.score_switch_weight = float(rospy.get_param(
            "~omni_score_side_switch_weight", 0.32))

        rospy.logwarn(
            "HYBRID_OMNI_READY horizon=%.2fs vmax=(%.2f,%.2f,%.2f) "
            "clearance wall=(%.2f,%.2f) dynamic=(%.2f,%.2f)",
            self.omni_control_horizon, self.omni_max_forward,
            self.omni_max_lateral, self.omni_max_angular,
            self.wall_hard_clearance, self.wall_preferred_clearance,
            self.dynamic_hard_clearance, self.dynamic_preferred_clearance)

    def reset_for_goal(self, goal):
        self.local_vy = 0.0
        self.last_vy = 0.0
        self.local_blocked_since = None
        self.last_selected_side = 0
        super(HybridOmniActionServer, self).reset_for_goal(goal)

    def static_occupied_near(self, wx, wy):
        if self.grid is None:
            return False
        center = self.grid.world_to_map(wx, wy)
        if center is None:
            return True
        radius_cells = max(
            1, int(math.ceil(self.static_match_radius / self.grid.resolution)))
        cx, cy = center
        r2 = radius_cells * radius_cells
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > r2:
                    continue
                mx = cx + dx
                my = cy + dy
                if not self.grid.in_bounds(mx, my):
                    return True
                value = self.grid.data[self.grid.index(mx, my)]
                if value < 0 and self.grid.unknown_is_obstacle:
                    return True
                if value >= self.grid.occupied_threshold:
                    return True
        return False

    def classified_scan_points(self):
        walls = []
        dynamic = []
        if self.scan is None or self.pose is None:
            return walls, dynamic
        px, py, yaw = self.pose
        for i in range(0, len(self.scan.ranges), self.omni_scan_stride):
            value = self.scan.ranges[i]
            if (math.isnan(value) or math.isinf(value) or
                    value < self.scan.range_min or
                    value > min(self.scan.range_max, self.omni_scan_range)):
                continue
            angle = self.scan.angle_min + i * self.scan.angle_increment
            bx = value * math.cos(angle)
            by = value * math.sin(angle)
            wx = px + math.cos(yaw) * bx - math.sin(yaw) * by
            wy = py + math.sin(yaw) * bx + math.cos(yaw) * by
            if self.static_occupied_near(wx, wy):
                walls.append((bx, by))
            else:
                dynamic.append((bx, by))
        return walls, dynamic

    def local_path_points(self, max_points=45):
        if not self.path_world or self.pose is None:
            return []
        x, y, yaw = self.pose
        start = max(0, self.path_index - 2)
        end = min(len(self.path_world), start + max_points * 2)
        stride = max(1, int(math.ceil(max(1, end - start) / float(max_points))))
        result = []
        for i in range(start, end, stride):
            dx = self.path_world[i][0] - x
            dy = self.path_world[i][1] - y
            result.append((
                math.cos(yaw) * dx + math.sin(yaw) * dy,
                -math.sin(yaw) * dx + math.cos(yaw) * dy))
        return result

    def candidate_commands(self, target):
        x, y, yaw = self.pose
        dx = target[0] - x
        dy = target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        desired = math.atan2(local_y, max(local_x, 1.0e-4))
        goal_dist = self.distance_to_active_goal()
        speed_scale = clamp(goal_dist / 0.50, 0.45, 1.0)
        nominal_speed = max(
            self.omni_min_speed, self.omni_max_forward * speed_scale)
        nominal_wz = clamp(
            0.85 * desired, -self.omni_max_angular, self.omni_max_angular)

        # Bias around the global path direction but include broad side steps.
        offsets_deg = (0, 18, -18, 35, -35, 55, -55, 72, -72)
        commands = []
        for offset_deg in offsets_deg:
            direction = desired + math.radians(offset_deg)
            for scale in (1.0, 0.66):
                speed = nominal_speed * scale
                vx = max(0.025, speed * math.cos(direction))
                vy = clamp(
                    speed * math.sin(direction),
                    -self.omni_max_lateral, self.omni_max_lateral)
                for wz_scale in (1.0, 0.35):
                    commands.append((
                        vx, vy,
                        clamp(nominal_wz * wz_scale,
                              -self.omni_max_angular,
                              self.omni_max_angular)))
        # Rotation is a valid last resort, but movement wins whenever a safe
        # trajectory makes useful progress.
        commands.append((0.0, 0.0, nominal_wz))
        commands.append((0.0, 0.0,
                         self.omni_max_angular if desired >= 0.0
                         else -self.omni_max_angular))
        return commands, (local_x, local_y)

    @staticmethod
    def nearest_distance(x, y, points):
        best = float("inf")
        for px, py in points:
            best = min(best, math.hypot(px - x, py - y))
        return best

    def evaluate_candidate(self, command, target_local, path_local,
                           walls, dynamic):
        vx, vy, wz = command
        x = 0.0
        y = 0.0
        theta = 0.0
        min_wall = float("inf")
        min_dynamic = float("inf")
        path_error_sum = 0.0
        steps = max(2, int(math.ceil(
            self.omni_control_horizon / self.omni_sim_dt)))

        for _ in range(steps):
            x += (math.cos(theta) * vx - math.sin(theta) * vy) * self.omni_sim_dt
            y += (math.sin(theta) * vx + math.cos(theta) * vy) * self.omni_sim_dt
            theta = norm_angle(theta + wz * self.omni_sim_dt)

            if self.grid is not None and self.pose is not None:
                px, py, yaw = self.pose
                wx = px + math.cos(yaw) * x - math.sin(yaw) * y
                wy = py + math.sin(yaw) * x + math.cos(yaw) * y
                cell = self.grid.world_to_map(wx, wy)
                if cell is None:
                    return None
                if self.grid.clearance_m(cell[0], cell[1]) < self.static_map_hard_clearance:
                    return None

            wall_clear = self.nearest_distance(x, y, walls)
            dynamic_clear = self.nearest_distance(x, y, dynamic)
            min_wall = min(min_wall, wall_clear)
            min_dynamic = min(min_dynamic, dynamic_clear)
            if wall_clear < self.wall_hard_clearance:
                return None
            if dynamic_clear < self.dynamic_hard_clearance:
                return None
            if path_local:
                path_error_sum += self.nearest_distance(x, y, path_local)

        goal_error = math.hypot(target_local[0] - x, target_local[1] - y)
        avg_path_error = path_error_sum / float(steps) if path_local else goal_error
        wall_penalty = 0.0
        if min_wall < self.wall_preferred_clearance:
            wall_penalty = (
                (self.wall_preferred_clearance - min_wall) /
                max(self.wall_preferred_clearance - self.wall_hard_clearance,
                    1.0e-3)) ** 2
        dynamic_penalty = 0.0
        if min_dynamic < self.dynamic_preferred_clearance:
            dynamic_penalty = (
                (self.dynamic_preferred_clearance - min_dynamic) /
                max(self.dynamic_preferred_clearance - self.dynamic_hard_clearance,
                    1.0e-3)) ** 2

        side = 1 if vy > 0.025 else (-1 if vy < -0.025 else 0)
        switch_penalty = (
            self.score_switch_weight
            if self.last_selected_side and side and side != self.last_selected_side
            else 0.0)
        progress = math.hypot(x, y)
        stop_penalty = 2.0 if progress < 0.015 else 0.0
        score = (
            self.score_path_weight * avg_path_error +
            self.score_goal_weight * goal_error +
            self.score_dynamic_weight * dynamic_penalty +
            self.score_wall_weight * wall_penalty +
            self.score_lateral_weight * abs(vy) +
            self.score_turn_weight * abs(wz) +
            switch_penalty + stop_penalty - 0.65 * progress)
        return score, side, min_wall, min_dynamic

    def choose_omni_command(self, target):
        walls, dynamic = self.classified_scan_points()
        path_local = self.local_path_points()
        commands, target_local = self.candidate_commands(target)
        best = None
        for command in commands:
            result = self.evaluate_candidate(
                command, target_local, path_local, walls, dynamic)
            if result is None:
                continue
            score, side, min_wall, min_dynamic = result
            if best is None or score < best[0]:
                best = (score, command, side, min_wall, min_dynamic)
        if best is None:
            return None
        _, command, side, min_wall, min_dynamic = best
        if side:
            self.last_selected_side = side
        rospy.logwarn_throttle(
            0.8,
            "OMNI_LOCAL_SELECTED cmd=(%.3f,%.3f,%.3f) "
            "clear=(wall %.3f,dynamic %.3f) points=(%d,%d)",
            command[0], command[1], command[2], min_wall, min_dynamic,
            len(walls), len(dynamic))
        return command

    def compute_cmd(self, target):
        if not self.omni_enabled or not self.indoor_profile_active:
            self.local_vy = 0.0
            return super(HybridOmniActionServer, self).compute_cmd(target)
        selected = self.choose_omni_command(target)
        if selected is None:
            self.local_vy = 0.0
            if self.local_blocked_since is None:
                self.local_blocked_since = rospy.Time.now()
            rospy.logwarn_throttle(
                0.5, "OMNI_LOCAL_NO_VALID_TRAJECTORY stopping before obstacle")
            return 0.0, 0.0
        self.local_blocked_since = None
        self.local_vy = selected[1]
        return selected[0], selected[2]

    def apply_scan_guard(self, cmd):
        if not self.omni_enabled or not self.indoor_profile_active:
            self.local_vy = 0.0
            return super(HybridOmniActionServer, self).apply_scan_guard(cmd)

        vx, wz = cmd
        # The trajectory evaluator owns normal avoidance.  This final guard
        # only handles an observation that became critical after simulation.
        if vx > 0.0 and self.front < self.indoor_wall_front_stop:
            vx = 0.0
        if self.local_vy > 0.0 and self.left < self.indoor_wall_side_stop:
            self.local_vy = 0.0
        if self.local_vy < 0.0 and self.right < self.indoor_wall_side_stop:
            self.local_vy = 0.0

        blocked = (abs(vx) < 1.0e-4 and
                   abs(self.local_vy) < 1.0e-4)
        now = rospy.Time.now()
        if blocked:
            if self.local_blocked_since is None:
                self.local_blocked_since = now
            blocked_for = (now - self.local_blocked_since).to_sec()
            if (blocked_for >= self.local_block_replan_s and
                    (now - self.last_local_replan).to_sec() >=
                    self.local_replan_interval_s):
                self.last_local_replan = now
                added = self.remember_front_dynamic_obstacles(
                    "omni local planner persistently blocked")
                self.path_world = []
                self.path_index = 0
                self.last_plan_time = rospy.Time(0)
                self.plan_from_current_pose(
                    "omni local planner blocked", force=True)
                rospy.logwarn(
                    "OMNI_LOCAL_GLOBAL_REPLAN blocked_for=%.2fs cells=%d",
                    blocked_for, added)
        else:
            self.local_blocked_since = None
        return vx, wz

    def apply_goal_approach_limit(self, cmd):
        before = max(abs(cmd[0]), 1.0e-6)
        limited = super(HybridOmniActionServer, self).apply_goal_approach_limit(cmd)
        if abs(cmd[0]) > 1.0e-6:
            self.local_vy *= clamp(abs(limited[0]) / before, 0.0, 1.0)
        return limited

    def smooth_cmd(self, cmd):
        smoothed = super(HybridOmniActionServer, self).smooth_cmd(cmd)
        dt = max(
            1.0 / max(self.control_rate_hz, 1.0), 0.02)
        max_dvy = self.omni_lateral_accel * dt
        self.local_vy = clamp(
            self.local_vy, self.last_vy - max_dvy, self.last_vy + max_dvy)
        self.local_vy = clamp(
            self.local_vy, -self.omni_max_lateral, self.omni_max_lateral)
        self.last_vy = self.local_vy
        return smoothed

    def check_goal(self):
        if (self.pose is not None and
                self.distance_to_active_goal() <= self.current_target_tolerance()):
            self.local_vy = 0.0
            self.last_vy = 0.0
        return super(HybridOmniActionServer, self).check_goal()

    def publish_cmd(self, vx, wz):
        if not self.action_active:
            return
        msg = Twist()
        msg.linear.x = clamp(vx, 0.0, self.omni_max_forward)
        msg.linear.y = clamp(
            self.local_vy, -self.omni_max_lateral, self.omni_max_lateral)
        msg.angular.z = clamp(
            wz, -self.omni_max_angular, self.omni_max_angular)
        self.cmd_pub.publish(msg)

    def publish_zero(self, reason):
        self.local_vy = 0.0
        self.last_vy = 0.0
        super(HybridOmniActionServer, self).publish_zero(reason)


if __name__ == "__main__":
    HybridOmniActionServer().spin()
