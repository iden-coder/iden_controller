#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified high-speed holonomic navigation for a 2-D laser robot.

There are no room profiles or obstacle-type size assumptions.  A static graph
planner supplies the route; a rolling laser obstacle memory and short-horizon
vx/vy/wz trajectory search perform real-time avoidance everywhere.
"""

import math
import os
import pickle
import threading
import time
import zlib

import rospy
from geometry_msgs.msg import Twist

from global_first_graph_nav_2249fcf import (
    GlobalFirstGraphPlanner,
    GridMap,
    INF,
    RosGlobalFirstGraphNavigator,
    clamp,
    load_map_yaml,
    norm_angle,
    yaw_from_quat,
)


class UniversalOmniNavigator(RosGlobalFirstGraphNavigator):
    def __init__(self):
        self.universal_ready = False
        self.local_vy = 0.0
        self.last_vy = 0.0
        self.last_selected_side = 0
        self.dynamic_memory = {}
        self.dynamic_memory_lock = threading.RLock()
        self.dynamic_cluster_count = 0
        # Laser callbacks may arrive while the parent constructor is loading
        # the map, before ROS parameters below are read.
        self.scan_range = 1.80
        self.scan_stride = 2
        self.memory_ttl = 1.80
        self.memory_voxel = 0.045
        self.static_match_radius = 0.11
        self.dynamic_confirmations = 2
        self.dynamic_confirmation_window = 0.45
        self.max_forward = 0.52
        self.max_lateral = 0.22
        self.max_local_angular = 0.75
        self.local_blocked_since = None
        self.last_local_replan = rospy.Time(0)
        self.recovery_until = rospy.Time(0)
        self.recovery_cmd = (0.0, 0.0, 0.0)
        self.recovery_replan_pending = False
        self.recovery_count = 0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_wz = 0.0
        super(UniversalOmniNavigator, self).__init__()

        self.horizon_s = float(rospy.get_param("~local_horizon_s", 1.25))
        self.sim_dt = float(rospy.get_param("~local_sim_dt_s", 0.10))
        self.scan_range = float(rospy.get_param("~local_scan_range_m", 1.80))
        self.scan_stride = max(1, int(rospy.get_param("~local_scan_stride", 2)))
        self.memory_ttl = float(rospy.get_param(
            "~dynamic_memory_ttl_s", 1.80))
        self.memory_voxel = float(rospy.get_param(
            "~dynamic_memory_voxel_m", 0.045))
        self.dynamic_confirmations = max(1, int(rospy.get_param(
            "~dynamic_confirmations", 2)))
        self.dynamic_confirmation_window = float(rospy.get_param(
            "~dynamic_confirmation_window_s", 0.45))
        self.static_match_radius = float(rospy.get_param(
            "~static_scan_match_radius_m", 0.11))

        self.max_forward = float(rospy.get_param(
            "~local_max_forward_mps", 0.52))
        self.max_lateral = float(rospy.get_param(
            "~local_max_lateral_mps", 0.22))
        self.max_local_angular = float(rospy.get_param(
            "~local_max_angular_rps", 0.75))
        self.min_translation = float(rospy.get_param(
            "~local_min_translation_mps", 0.09))
        self.lateral_accel = float(rospy.get_param(
            "~local_lateral_accel_mps2", 0.65))

        # The scan points describe the observed surface, so these are robot
        # centre-to-surface clearances, not guessed obstacle radii.
        self.wall_hard = float(rospy.get_param(
            "~wall_hard_clearance_m", 0.215))
        self.wall_preferred = float(rospy.get_param(
            "~wall_preferred_clearance_m", 0.255))
        self.dynamic_hard = float(rospy.get_param(
            "~dynamic_hard_clearance_m", 0.225))
        self.dynamic_preferred = float(rospy.get_param(
            "~dynamic_preferred_clearance_m", 0.295))
        self.map_hard = float(rospy.get_param(
            "~map_hard_clearance_m", 0.16))
        self.robot_half_length = float(rospy.get_param(
            "~robot_half_length_m", 0.171))
        self.robot_half_width = float(rospy.get_param(
            "~robot_half_width_m", 0.128))
        self.footprint_margin = float(rospy.get_param(
            "~footprint_hard_margin_m", 0.010))
        self.wall_preferred_gap = float(rospy.get_param(
            "~wall_preferred_gap_m", 0.025))
        self.dynamic_preferred_gap = float(rospy.get_param(
            "~dynamic_preferred_gap_m", 0.050))

        self.block_replan_s = float(rospy.get_param(
            "~local_block_replan_s", 0.90))
        self.local_replan_interval = float(rospy.get_param(
            "~local_replan_interval_s", 1.10))
        self.global_dynamic_inflation = float(rospy.get_param(
            "~global_dynamic_point_inflation_m", 0.20))
        self.universal_stuck_s = float(rospy.get_param(
            "~universal_stuck_s", 3.8))
        self.recovery_duration = float(rospy.get_param(
            "~recovery_duration_s", 0.65))
        self.recovery_lateral_speed = float(rospy.get_param(
            "~recovery_lateral_mps", 0.10))
        self.recovery_turn_speed = float(rospy.get_param(
            "~recovery_turn_rps", 0.30))

        self.score_path = float(rospy.get_param("~score_path", 4.6))
        self.score_goal = float(rospy.get_param("~score_goal", 2.8))
        self.score_dynamic = float(rospy.get_param("~score_dynamic", 3.0))
        self.score_wall = float(rospy.get_param("~score_wall", 0.85))
        self.score_lateral = float(rospy.get_param("~score_lateral", 0.18))
        self.score_turn = float(rospy.get_param("~score_turn", 0.14))
        self.score_switch = float(rospy.get_param("~score_side_switch", 0.35))
        self.score_speed_reward = float(rospy.get_param(
            "~score_speed_reward", 0.90))
        self.score_velocity_continuity = float(rospy.get_param(
            "~score_velocity_continuity", 5.0))
        self.score_angular_continuity = float(rospy.get_param(
            "~score_angular_continuity", 1.2))
        self.linear_jerk = float(rospy.get_param(
            "~linear_jerk_mps3", 2.2))
        self.lateral_jerk = float(rospy.get_param(
            "~lateral_jerk_mps3", 2.0))
        self.angular_jerk = float(rospy.get_param(
            "~angular_jerk_rps3", 4.0))

        self.universal_ready = True
        self.path_world = []
        self.path_index = 0
        self.last_plan_time = rospy.Time(0)
        rospy.logwarn(
            "UNIVERSAL_OMNI_READY goal=(%.3f,%.3f,%.1fdeg) "
            "vmax=(%.2f,%.2f,%.2f) horizon=%.2fs no_room_profiles=true",
            self.goal_x, self.goal_y, math.degrees(self.goal_yaw),
            self.max_forward, self.max_lateral, self.max_local_angular,
            self.horizon_s)

    def load_static_map(self):
        """Load the map and reuse a validated static distance-field cache."""
        from nav_msgs.msg import OccupancyGrid
        from nav_msgs.srv import GetMap

        map_service = rospy.get_param("~map_service", "/static_map")
        map_topic = rospy.get_param("~map_topic", "/map")
        map_yaml = rospy.get_param("~map_yaml", "")
        cache_path = rospy.get_param(
            "~distance_cache_file",
            "/home/ucar/instant_ws/src/iden_controller/config/"
            "universal_omni_distance_cache_v1.pkl")
        msg = None
        try:
            rospy.wait_for_service(map_service, timeout=5.0)
            msg = rospy.ServiceProxy(map_service, GetMap)().map
            rospy.logwarn("UniversalOmni: loaded static map from %s", map_service)
        except Exception as exc:
            rospy.logwarn("UniversalOmni: map service failed: %s", str(exc))
        if msg is None:
            try:
                msg = rospy.wait_for_message(
                    map_topic, OccupancyGrid, timeout=5.0)
            except Exception as exc:
                rospy.logwarn("UniversalOmni: map topic failed: %s", str(exc))

        if msg is not None:
            yaw = yaw_from_quat(msg.info.origin.orientation)
            origin = (msg.info.origin.position.x,
                      msg.info.origin.position.y, yaw)
            self.grid = GridMap(
                msg.info.width, msg.info.height, msg.info.resolution,
                origin, msg.data, self.occupied_threshold,
                self.unknown_is_obstacle)
        elif map_yaml:
            self.grid = load_map_yaml(
                map_yaml, self.occupied_threshold, self.unknown_is_obstacle)
        else:
            raise RuntimeError("no static map available")

        raw = bytes((int(value) + 256) % 256 for value in self.grid.data)
        key = (
            self.grid.width, self.grid.height, round(self.grid.resolution, 7),
            round(self.grid.origin_x, 6), round(self.grid.origin_y, 6),
            round(self.grid.origin_yaw, 6), self.grid.occupied_threshold,
            self.grid.unknown_is_obstacle, zlib.crc32(raw))
        loaded = False
        t0 = time.time()
        try:
            with open(cache_path, "rb") as handle:
                cached = pickle.load(handle)
            if cached.get("key") == key and len(cached.get("dist", [])) == len(raw):
                self.grid.dist_cells = cached["dist"]
                loaded = True
        except Exception:
            pass
        if not loaded:
            self.grid.compute_distance_field()
            try:
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                with open(cache_path, "wb") as handle:
                    pickle.dump(
                        {"key": key, "dist": self.grid.dist_cells},
                        handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                rospy.logwarn("UniversalOmni: cache write failed: %s", str(exc))
        self.planner = GlobalFirstGraphPlanner(self.grid, self.params)
        rospy.logwarn(
            "UNIVERSAL_DISTANCE_FIELD cache=%s elapsed=%.3fs",
            "hit" if loaded else "built", time.time() - t0)

    def static_occupied_near(self, wx, wy):
        if self.grid is None:
            return False
        center = self.grid.world_to_map(wx, wy)
        if center is None:
            return True
        radius = max(1, int(math.ceil(
            self.static_match_radius / self.grid.resolution)))
        cx, cy = center
        for my in range(cy - radius, cy + radius + 1):
            for mx in range(cx - radius, cx + radius + 1):
                if not self.grid.in_bounds(mx, my):
                    return True
                if ((mx - cx) ** 2 + (my - cy) ** 2 > radius ** 2):
                    continue
                value = self.grid.data[self.grid.index(mx, my)]
                if value < 0 and self.grid.unknown_is_obstacle:
                    return True
                if value >= self.grid.occupied_threshold:
                    return True
        return False

    def cb_scan(self, msg):
        super(UniversalOmniNavigator, self).cb_scan(msg)
        if self.pose is None or self.grid is None:
            return
        now = rospy.Time.now()
        now_sec = now.to_sec()
        px, py, yaw = self.pose
        residual_base = []

        for i in range(0, len(msg.ranges), self.scan_stride):
            value = msg.ranges[i]
            if (math.isnan(value) or math.isinf(value) or
                    value < msg.range_min or
                    value > min(msg.range_max, self.scan_range)):
                continue
            angle = msg.angle_min + i * msg.angle_increment
            bx = value * math.cos(angle)
            by = value * math.sin(angle)
            wx = px + math.cos(yaw) * bx - math.sin(yaw) * by
            wy = py + math.sin(yaw) * bx + math.cos(yaw) * by
            if self.static_occupied_near(wx, wy):
                continue
            residual_base.append((bx, by, wx, wy))
            key = (int(round(wx / self.memory_voxel)),
                   int(round(wy / self.memory_voxel)))
            with self.dynamic_memory_lock:
                previous_value = self.dynamic_memory.get(key)
                if (previous_value is not None and
                        now_sec - previous_value[2] <=
                        self.dynamic_confirmation_window):
                    count = min(previous_value[3] + 1, 20)
                    # Low-pass the world point so AMCL jitter does not make a
                    # stationary surface jump between local trajectories.
                    wx = 0.55 * previous_value[0] + 0.45 * wx
                    wy = 0.55 * previous_value[1] + 0.45 * wy
                else:
                    count = 1
                self.dynamic_memory[key] = (wx, wy, now_sec, count)

        # Estimate only the number/shape of observed objects.  Collision
        # checking still uses every surface voxel, so no object radius is
        # assumed by the planner.
        clusters = 0
        previous = None
        for bx, by, _, _ in residual_base:
            if previous is None:
                clusters += 1
            else:
                gap = math.hypot(bx - previous[0], by - previous[1])
                radial = math.hypot(bx, by)
                if gap > max(0.075, 0.035 + 0.035 * radial):
                    clusters += 1
            previous = (bx, by)
        self.dynamic_cluster_count = clusters

        with self.dynamic_memory_lock:
            stale = [key for key, value in self.dynamic_memory.items()
                     if now_sec - value[2] > self.memory_ttl]
            for key in stale:
                del self.dynamic_memory[key]
            memory_size = len(self.dynamic_memory)
            confirmed_size = sum(
                1 for value in self.dynamic_memory.values()
                if value[3] >= self.dynamic_confirmations)
        rospy.logwarn_throttle(
            1.0,
            "UNIVERSAL_OBSTACLES clusters=%d raw_voxels=%d confirmed=%d",
            self.dynamic_cluster_count, memory_size, confirmed_size)

    def current_obstacles_base(self):
        walls = []
        dynamic = []
        if self.pose is None or self.scan is None:
            return walls, dynamic
        px, py, yaw = self.pose
        for i in range(0, len(self.scan.ranges), self.scan_stride):
            value = self.scan.ranges[i]
            if (math.isnan(value) or math.isinf(value) or
                    value < self.scan.range_min or
                    value > min(self.scan.range_max, self.scan_range)):
                continue
            angle = self.scan.angle_min + i * self.scan.angle_increment
            bx = value * math.cos(angle)
            by = value * math.sin(angle)
            wx = px + math.cos(yaw) * bx - math.sin(yaw) * by
            wy = py + math.sin(yaw) * bx + math.cos(yaw) * by
            if self.static_occupied_near(wx, wy):
                walls.append((bx, by))

        now_sec = rospy.Time.now().to_sec()
        with self.dynamic_memory_lock:
            memory_snapshot = list(self.dynamic_memory.values())
        for wx, wy, stamp, count in memory_snapshot:
            if now_sec - stamp > self.memory_ttl:
                continue
            if count < self.dynamic_confirmations:
                continue
            dx = wx - px
            dy = wy - py
            dynamic.append((
                math.cos(yaw) * dx + math.sin(yaw) * dy,
                -math.sin(yaw) * dx + math.cos(yaw) * dy))
        return walls, dynamic

    def local_path_points(self):
        if not self.path_world or self.pose is None:
            return []
        x, y, yaw = self.pose
        result = []
        start = max(0, self.path_index - 2)
        end = min(len(self.path_world), start + 72)
        for i in range(start, end, 3):
            dx = self.path_world[i][0] - x
            dy = self.path_world[i][1] - y
            result.append((
                math.cos(yaw) * dx + math.sin(yaw) * dy,
                -math.sin(yaw) * dx + math.cos(yaw) * dy))
        return result

    @staticmethod
    def nearest_distance(x, y, points):
        best = INF
        for px, py in points:
            best = min(best, math.hypot(px - x, py - y))
        return best

    def footprint_gap(self, x, y, theta, points):
        """Signed obstacle gap to the predicted oriented rectangle.

        Positive is free space outside the footprint; zero is contact;
        negative means the measured surface lies inside the hard footprint.
        """
        if not points:
            return INF
        half_length = self.robot_half_length + self.footprint_margin
        half_width = self.robot_half_width + self.footprint_margin
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        best = INF
        for ox, oy in points:
            dx = ox - x
            dy = oy - y
            local_x = cos_t * dx + sin_t * dy
            local_y = -sin_t * dx + cos_t * dy
            outside_x = max(abs(local_x) - half_length, 0.0)
            outside_y = max(abs(local_y) - half_width, 0.0)
            if outside_x > 0.0 or outside_y > 0.0:
                gap = math.hypot(outside_x, outside_y)
            else:
                gap = -min(half_length - abs(local_x),
                           half_width - abs(local_y))
            best = min(best, gap)
        return best

    def adaptive_cruise_speed(self, walls, dynamic, desired):
        # Measure free distance in the intended travel corridor.  Parallel
        # side walls are excluded, but any surface crossing the swept width is
        # retained.  Full 360-degree collision checks still run afterwards.
        cos_d = math.cos(desired)
        sin_d = math.sin(desired)
        best_gap = INF
        for points, lateral_limit in (
                (walls, self.robot_half_width + self.footprint_margin +
                 self.wall_preferred_gap),
                (dynamic, self.robot_half_width + self.footprint_margin +
                 self.dynamic_preferred_gap)):
            for px, py in points:
                forward = cos_d * px + sin_d * py
                lateral = -sin_d * px + cos_d * py
                if forward > 0.0 and abs(lateral) <= lateral_limit:
                    best_gap = min(
                        best_gap,
                        forward - self.robot_half_length - self.footprint_margin)
        if best_gap >= 0.95:
            return self.max_forward
        if best_gap <= 0.16:
            return max(self.min_translation, self.max_forward * 0.35)
        ratio = (best_gap - 0.16) / (0.95 - 0.16)
        return self.max_forward * (0.35 + 0.65 * ratio)

    def candidate_commands(self, target, walls, dynamic):
        x, y, yaw = self.pose
        dx = target[0] - x
        dy = target[1] - y
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        desired = math.atan2(local_y, max(local_x, 1.0e-4))
        speed = self.adaptive_cruise_speed(walls, dynamic, desired)
        speed *= clamp(self.distance_to_active_goal() / 0.55, 0.30, 1.0)
        speed = max(self.min_translation, speed)
        nominal_wz = clamp(0.80 * desired,
                           -self.max_local_angular, self.max_local_angular)
        commands = []
        for offset in (0, 20, -20, 42, -42, 68, -68):
            direction = desired + math.radians(offset)
            for scale in (1.0, 0.58, 0.32):
                candidate_speed = speed * scale
                vx = max(0.02, candidate_speed * math.cos(direction))
                vy = clamp(candidate_speed * math.sin(direction),
                           -self.max_lateral, self.max_lateral)
                for turn_scale in (1.0, 0.25):
                    commands.append((
                        vx, vy,
                        clamp(nominal_wz * turn_scale,
                              -self.max_local_angular,
                              self.max_local_angular)))
        return commands, (local_x, local_y)

    def evaluate(self, command, target_local, path_local, walls, dynamic):
        vx, vy, wz = command
        x = y = theta = 0.0
        start_wall = self.footprint_gap(0.0, 0.0, 0.0, walls)
        start_dynamic = self.footprint_gap(0.0, 0.0, 0.0, dynamic)
        min_wall = start_wall
        min_dynamic = start_dynamic
        start_map_clearance = INF
        if self.grid is not None and self.pose is not None:
            start_cell = self.grid.world_to_map(self.pose[0], self.pose[1])
            if start_cell is not None:
                start_map_clearance = self.grid.clearance_m(*start_cell)
        path_error = 0.0
        steps = max(3, int(math.ceil(self.horizon_s / self.sim_dt)))
        for _ in range(steps):
            x += (math.cos(theta) * vx - math.sin(theta) * vy) * self.sim_dt
            y += (math.sin(theta) * vx + math.cos(theta) * vy) * self.sim_dt
            theta = norm_angle(theta + wz * self.sim_dt)
            wall_clear = self.footprint_gap(x, y, theta, walls)
            dynamic_clear = self.footprint_gap(x, y, theta, dynamic)
            min_wall = min(min_wall, wall_clear)
            min_dynamic = min(min_dynamic, dynamic_clear)
            # Localization and map discretization can place the initial robot
            # slightly inside a nominal clearance contour.  Permit a command
            # that monotonically increases clearance; reject motion that stays
            # equally close or moves farther into the obstacle.
            if (wall_clear <= 0.0 and
                    wall_clear < start_wall - 0.003):
                return None
            if (dynamic_clear <= 0.0 and
                    dynamic_clear < start_dynamic - 0.003):
                return None

            if self.grid is not None and self.pose is not None:
                px, py, yaw = self.pose
                wx = px + math.cos(yaw) * x - math.sin(yaw) * y
                wy = py + math.sin(yaw) * x + math.cos(yaw) * y
                cell = self.grid.world_to_map(wx, wy)
                if cell is None:
                    return None
                map_clearance = self.grid.clearance_m(*cell)
                if (map_clearance < self.map_hard and
                        map_clearance < start_map_clearance - 0.004):
                    return None
            if path_local:
                path_error += self.nearest_distance(x, y, path_local)

        target_distance = max(math.hypot(*target_local), 1e-6)
        unit_x = target_local[0] / target_distance
        unit_y = target_local[1] / target_distance
        along = x * unit_x + y * unit_y
        cross_track = abs(-unit_y * x + unit_x * y)
        # Passing a short lookahead point is progress, not an overshoot error.
        goal_error = max(0.0, target_distance - along) + cross_track
        avg_path_error = path_error / steps if path_local else goal_error
        wall_penalty = 0.0
        if min_wall < self.wall_preferred_gap:
            wall_penalty = ((self.wall_preferred_gap - min_wall) /
                            max(self.wall_preferred_gap, 1e-3)) ** 2
        dynamic_penalty = 0.0
        if min_dynamic < self.dynamic_preferred_gap:
            dynamic_penalty = ((self.dynamic_preferred_gap - min_dynamic) /
                               max(self.dynamic_preferred_gap, 1e-3)) ** 2
        side = 1 if vy > 0.025 else (-1 if vy < -0.025 else 0)
        switch = (self.score_switch if self.last_selected_side and side and
                  side != self.last_selected_side else 0.0)
        progress = math.hypot(x, y)
        stop = 2.5 if progress < 0.012 else 0.0
        score = (
            self.score_path * avg_path_error +
            self.score_goal * goal_error +
            self.score_dynamic * dynamic_penalty +
            self.score_wall * wall_penalty +
            self.score_lateral * abs(vy) +
            self.score_turn * abs(wz) + switch + stop -
            self.score_speed_reward * progress)
        last_vx, last_wz = self.last_cmd
        score += self.score_velocity_continuity * (
            (vx - last_vx) ** 2 + (vy - self.last_vy) ** 2)
        score += self.score_angular_continuity * (wz - last_wz) ** 2
        return score, side, min_wall, min_dynamic

    def compute_cmd(self, target):
        walls, dynamic = self.current_obstacles_base()
        map_clearance = INF
        if self.grid is not None and self.pose is not None:
            cell = self.grid.world_to_map(self.pose[0], self.pose[1])
            if cell is not None:
                map_clearance = self.grid.clearance_m(*cell)
        rospy.logwarn_throttle(
            1.0,
            "UNIVERSAL_LOCAL_CONTEXT map_clear=%.3f nearest_wall=%.3f "
            "nearest_dynamic=%.3f points=(%d,%d)",
            map_clearance, self.nearest_distance(0.0, 0.0, walls),
            self.nearest_distance(0.0, 0.0, dynamic),
            len(walls), len(dynamic))
        commands, target_local = self.candidate_commands(target, walls, dynamic)
        path_local = self.local_path_points()
        best = None
        for command in commands:
            result = self.evaluate(
                command, target_local, path_local, walls, dynamic)
            if result is None:
                continue
            score, side, min_wall, min_dynamic = result
            if best is None or score < best[0]:
                best = (score, command, side, min_wall, min_dynamic)
        if best is None:
            self.local_vy = 0.0
            if self.local_blocked_since is None:
                self.local_blocked_since = rospy.Time.now()
            rospy.logwarn_throttle(0.4, "UNIVERSAL_LOCAL_NO_VALID_TRAJECTORY")
            return 0.0, 0.0

        self.local_blocked_since = None
        command = best[1]
        self.local_vy = command[1]
        if best[2]:
            self.last_selected_side = best[2]
        rospy.logwarn_throttle(
            0.65,
            "UNIVERSAL_LOCAL_CMD x=%.3f y=%.3f wz=%.3f "
            "clear=(wall %.3f,dynamic %.3f) clusters=%d",
            command[0], command[1], command[2], best[3], best[4],
            self.dynamic_cluster_count)
        return command[0], command[2]

    def refresh_global_dynamic_map(self):
        if self.grid is None or self.planner is None:
            return 0
        # Dynamic laser observations belong to the rolling local layer.  They
        # must never poison the static graph or seal the robot's start cell.
        self.grid.clear_dynamic_blocks()
        self.planner.roadmaps.clear()
        with self.dynamic_memory_lock:
            memory_size = len(self.dynamic_memory)
        rospy.logwarn(
            "UNIVERSAL_STATIC_GLOBAL_RESET local_voxels=%d injected_cells=0",
            memory_size)
        return 0

    def apply_scan_guard(self, cmd):
        vx, wz = cmd
        if vx > 0.0 and self.front < self.front_stop_m:
            vx = 0.0
        if self.local_vy > 0.0 and self.left < self.side_stop_m:
            self.local_vy = 0.0
        if self.local_vy < 0.0 and self.right < self.side_stop_m:
            self.local_vy = 0.0
        blocked = abs(vx) < 1e-4 and abs(self.local_vy) < 1e-4
        now = rospy.Time.now()
        if blocked:
            if self.local_blocked_since is None:
                self.local_blocked_since = now
            if ((now - self.local_blocked_since).to_sec() >= self.block_replan_s and
                    (now - self.last_local_replan).to_sec() >=
                    self.local_replan_interval):
                self.last_local_replan = now
                # Keep the valid static route.  Fresh laser scans continue to
                # change local candidates, and the progress watchdog performs
                # active recovery if the obstruction persists.
                rospy.logwarn(
                    "UNIVERSAL_LOCAL_BLOCKED_WAITING_FOR_FRESH_TRAJECTORY")
        return vx, wz

    def select_target(self):
        speed = math.hypot(self.last_cmd[0], self.last_vy)
        original = self.lookahead_dist
        self.lookahead_dist = clamp(0.34 + 0.62 * speed, 0.36, 0.68)
        try:
            return super(UniversalOmniNavigator, self).select_target()
        finally:
            self.lookahead_dist = original

    def apply_goal_approach_limit(self, cmd):
        before = max(abs(cmd[0]), 1e-6)
        limited = super(UniversalOmniNavigator, self).apply_goal_approach_limit(cmd)
        if abs(cmd[0]) > 1e-6:
            self.local_vy *= clamp(abs(limited[0]) / before, 0.0, 1.0)
        return limited

    def smooth_cmd(self, cmd):
        now = rospy.Time.now()
        # Planning or a busy callback must not turn elapsed wall time into a
        # large acceleration allowance on the next command.
        dt = clamp(
            (now - self.last_control_time).to_sec(),
            1.0 / max(self.control_rate_hz, 1.0), 0.15)
        self.last_control_time = now
        vx, wz = cmd
        last_vx, last_wz = self.last_cmd
        target_vy = self.local_vy

        desired_ax = clamp(
            (vx - last_vx) / dt, -self.linear_accel, self.linear_accel)
        desired_ay = clamp(
            (target_vy - self.last_vy) / dt,
            -self.lateral_accel, self.lateral_accel)
        desired_aw = clamp(
            (wz - last_wz) / dt, -self.angular_accel, self.angular_accel)
        self.accel_x = clamp(
            desired_ax,
            self.accel_x - self.linear_jerk * dt,
            self.accel_x + self.linear_jerk * dt)
        self.accel_y = clamp(
            desired_ay,
            self.accel_y - self.lateral_jerk * dt,
            self.accel_y + self.lateral_jerk * dt)
        self.accel_wz = clamp(
            desired_aw,
            self.accel_wz - self.angular_jerk * dt,
            self.accel_wz + self.angular_jerk * dt)
        vx = last_vx + self.accel_x * dt
        self.local_vy = self.last_vy + self.accel_y * dt
        wz = last_wz + self.accel_wz * dt

        # A hard local guard must stop immediately; jerk limiting is only for
        # normal tracking changes.  The downstream safety monitor remains an
        # independent final veto.
        if cmd[0] <= 1e-5:
            vx = 0.0
            self.accel_x = 0.0
        if abs(target_vy) <= 1e-5:
            self.local_vy = 0.0
            self.accel_y = 0.0
        vx = clamp(vx, 0.0, self.max_forward)
        wz = clamp(wz, -self.max_local_angular, self.max_local_angular)
        self.local_vy = clamp(self.local_vy,
                              -self.max_lateral, self.max_lateral)
        self.last_cmd = (vx, wz)
        self.last_vy = self.local_vy
        return vx, wz

    def start_recovery(self):
        # Recovery is a short low-speed curve, never a fixed-angle in-place
        # rotation.  The normal global path remains active afterwards.
        if self.left >= self.right and self.left > self.side_slow_m:
            vx = 0.045
            vy = self.recovery_lateral_speed * 0.75
            wz = min(self.recovery_turn_speed, 0.14)
        elif self.right > self.side_slow_m:
            vx = 0.045
            vy = -self.recovery_lateral_speed * 0.75
            wz = -min(self.recovery_turn_speed, 0.14)
        else:
            vx = 0.030
            vy = 0.0
            wz = (min(self.recovery_turn_speed, 0.12)
                  if self.left >= self.right
                  else -min(self.recovery_turn_speed, 0.12))
        if self.recovery_count % 2:
            vy = -vy
            wz = -wz
        self.recovery_cmd = (vx, vy, wz)
        self.recovery_until = rospy.Time.now() + rospy.Duration(
            self.recovery_duration)
        self.recovery_replan_pending = True
        self.recovery_count += 1
        self.publish_zero("UNIVERSAL_RECOVERY_START")
        rospy.logwarn(
            "UNIVERSAL_RECOVERY_START count=%d cmd=(%.2f,%.2f,%.2f)",
            self.recovery_count, self.recovery_cmd[0],
            self.recovery_cmd[1], self.recovery_cmd[2])

    def check_goal(self):
        if (self.pose is not None and
                self.distance_to_active_goal() <= self.current_target_tolerance()):
            self.local_vy = 0.0
            self.last_vy = 0.0
        return super(UniversalOmniNavigator, self).check_goal()

    def control_loop(self, _event):
        if not self.universal_ready:
            self.publish_zero("UNIVERSAL_INITIALIZING")
            return
        now = rospy.Time.now()
        if self.recovery_until > now and not self.finished:
            self.local_vy = self.recovery_cmd[1]
            recovery_smoothed = self.smooth_cmd(
                (self.recovery_cmd[0], self.recovery_cmd[2]))
            self.publish_cmd(recovery_smoothed[0], recovery_smoothed[1])
            return
        if self.recovery_replan_pending and not self.finished:
            self.recovery_replan_pending = False
            self.local_vy = 0.0
            self.last_vy = 0.0
            # The static route is still valid.  Replanning the same route here
            # creates a multi-second pause and does not improve local escape.
            self.last_progress_time = now
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
        if ((now - self.last_progress_time).to_sec() > self.universal_stuck_s and
                self.distance_to_active_goal() >
                self.current_target_tolerance() + 0.16):
            self.start_recovery()
            self.last_progress_time = now
            return
        target = self.select_target()
        if target is None:
            self.plan_from_current_pose("path exhausted", force=True)
            return
        cmd = self.compute_cmd(target)
        cmd = self.apply_scan_guard(cmd)
        cmd = self.apply_goal_approach_limit(cmd)
        cmd = self.smooth_cmd(cmd)
        rospy.logwarn_throttle(
            0.6, "UNIVERSAL_SMOOTH_CMD x=%.3f y=%.3f wz=%.3f",
            cmd[0], self.local_vy, cmd[1])
        self.publish_cmd(cmd[0], cmd[1])

    def publish_cmd(self, vx, wz):
        msg = Twist()
        msg.linear.x = clamp(vx, 0.0, self.max_forward)
        msg.linear.y = clamp(self.local_vy,
                             -self.max_lateral, self.max_lateral)
        msg.angular.z = clamp(wz,
                              -self.max_local_angular,
                              self.max_local_angular)
        self.cmd_pub.publish(msg)

    def publish_zero(self, reason):
        self.local_vy = 0.0
        self.last_vy = 0.0
        self.accel_x = 0.0
        self.accel_y = 0.0
        self.accel_wz = 0.0
        super(UniversalOmniNavigator, self).publish_zero(reason)


if __name__ == "__main__":
    UniversalOmniNavigator().spin()
