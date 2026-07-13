#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Drive the first-stage and room-stage navigation points in one process."""

import math

import rospy
from std_msgs.msg import Bool

from front_first_smooth_navigation_v1 import FrontFirstSmoothNavigator
from global_first_graph_nav_2249fcf import clamp, norm_angle
from room_lidar_semantics_v1 import classify_scan


DEFAULT_ROUTE = [
    {"name": "first_stage", "x": -1.48, "y": -0.45,
     "yaw": math.pi, "tolerance": 0.10, "require_yaw": False},
    {"name": "doorway_approach", "x": -0.85, "y": -1.41,
     "yaw": -math.pi / 2.0, "tolerance": 0.16, "require_yaw": False},
    {"name": "room_entry", "x": -0.85, "y": -1.90,
     "yaw": -math.pi / 2.0, "tolerance": 0.20, "require_yaw": False},
    {"name": "d1", "x": -1.43, "y": -2.57,
     "yaw": math.pi, "tolerance": 0.14, "require_yaw": False},
    {"name": "d2", "x": 0.41, "y": -2.00,
     "yaw": math.pi / 2.0, "tolerance": 0.14, "require_yaw": False},
    {"name": "d3", "x": 2.54, "y": -2.51,
     "yaw": -math.pi / 2.0, "tolerance": 0.10, "require_yaw": True},
]


class FrontFirstTwoStageRouteTest(FrontFirstSmoothNavigator):
    def __init__(self):
        self.route_ready = False
        self.route_index = 0
        self.route_points = [dict(point) for point in DEFAULT_ROUTE]
        self.route_enter_time = None
        self.room_profile_active = False
        self.doorway_profile_active = False
        self.room_avoid_sign = 0.0
        self.room_avoid_sign_until = rospy.Time(0)
        self.room_cone_observations = []
        self.room_dynamic_last_seen = rospy.Time(0)
        self.room_dynamic_last_replan = rospy.Time(0)
        self.room_dynamic_blocked_since = None
        self.final_approach_active = False
        self.room_slalom_active = False
        self.room_slalom_point = None
        self.room_slalom_arc = []
        self.room_slalom_arc_index = 0
        self.room_slalom_sign = 0.0
        self.room_slalom_until = rospy.Time(0)
        self.room_slalom_cone_world = None
        self.room_slalom_ignore_world = None
        self.room_slalom_ignore_until = rospy.Time(0)
        self.room_cone_front = float("inf")
        self.room_wall_front = float("inf")
        super(FrontFirstTwoStageRouteTest, self).__init__()

        configured = rospy.get_param("~route_points", DEFAULT_ROUTE)
        self.route_points = self._validate_route(configured)
        self.route_hold_s = float(rospy.get_param("~route_hold_s", 0.45))
        self.route_final_hold_s = float(
            rospy.get_param("~route_final_hold_s", 0.80))
        self.route_yaw_tolerance = math.radians(float(
            rospy.get_param("~route_yaw_tolerance_deg", 9.0)))
        self.route_yaw_speed = float(
            rospy.get_param("~route_yaw_speed_rps", 0.28))
        self.room_trigger_y = float(rospy.get_param(
            "~room_profile_trigger_y", -1.75))
        self.room_cruise_speed = float(rospy.get_param(
            "~room_cruise_mps", 0.40))
        self.room_max_turn = float(rospy.get_param(
            "~room_max_turn_rps", 0.86))
        self.room_max_lateral = float(rospy.get_param(
            "~room_max_lateral_mps", 0.075))
        self.room_front_angle = float(rospy.get_param(
            "~room_front_angle_deg", 48.0))
        self.room_avoid_influence = float(rospy.get_param(
            "~room_avoid_front_influence_m", 0.65))
        self.room_avoid_stop = float(rospy.get_param(
            "~room_avoid_front_stop_m", 0.27))
        self.room_lateral_trigger = float(rospy.get_param(
            "~room_avoid_lateral_trigger_m", 0.43))
        self.room_lateral_side_need = float(rospy.get_param(
            "~room_avoid_lateral_side_need_m", 0.22))
        self.room_avoid_turn_gain = float(rospy.get_param(
            "~room_avoid_turn_gain_rps", 0.60))
        self.room_side_stop = float(rospy.get_param(
            "~room_side_stop_m", 0.19))
        self.room_side_slow = float(rospy.get_param(
            "~room_side_slow_m", 0.30))
        self.room_avoid_hold_s = float(rospy.get_param(
            "~room_avoid_direction_hold_s", 0.90))
        self.room_cone_base_pad = float(rospy.get_param(
            "~room_cone_base_pad_m", 0.10))
        self.room_wall_front_relax = float(rospy.get_param(
            "~room_wall_front_relax_m", 0.09))
        self.room_wall_side_relax = float(rospy.get_param(
            "~room_wall_side_relax_m", 0.05))
        self.doorway_cruise_speed = float(rospy.get_param(
            "~doorway_cruise_mps", 0.30))
        self.doorway_max_lateral = float(rospy.get_param(
            "~doorway_max_lateral_mps", 0.035))
        self.doorway_front_angle = float(rospy.get_param(
            "~doorway_front_angle_deg", 30.0))
        self.doorway_front_influence = float(rospy.get_param(
            "~doorway_front_influence_m", 0.48))
        self.doorway_front_stop = float(rospy.get_param(
            "~doorway_front_stop_m", 0.21))
        self.room_dynamic_replan_enabled = bool(rospy.get_param(
            "~room_dynamic_replan_enabled", True))
        self.room_dynamic_radius = float(rospy.get_param(
            "~room_dynamic_obstacle_radius_m", 0.14))
        self.room_dynamic_path_trigger = float(rospy.get_param(
            "~room_dynamic_path_trigger_m", 0.32))
        self.room_dynamic_observation_range = float(rospy.get_param(
            "~room_dynamic_observation_range_m", 1.20))
        self.room_dynamic_replan_interval = float(rospy.get_param(
            "~room_dynamic_replan_interval_s", 0.90))
        self.room_dynamic_ttl = float(rospy.get_param(
            "~room_dynamic_observation_ttl_s", 2.20))
        self.room_dynamic_confirm_s = float(rospy.get_param(
            "~room_dynamic_confirm_s", 0.25))
        self.room_dynamic_stuck_replan_s = float(rospy.get_param(
            "~room_dynamic_stuck_replan_s", 2.8))
        self.final_approach_radius = float(rospy.get_param(
            "~final_approach_radius_m", 0.34))
        self.final_approach_speed = float(rospy.get_param(
            "~final_approach_speed_mps", 0.18))
        self.final_wall_front_relax = float(rospy.get_param(
            "~final_wall_front_relax_m", 0.17))
        self.final_wall_side_relax = float(rospy.get_param(
            "~final_wall_side_relax_m", 0.10))
        # Slalom-style cone bypass is intentionally disabled. Cones are handled
        # only as ordinary dynamic obstacles plus scan safety limits.
        self.room_slalom_enabled = False
        self.room_slalom_trigger_m = float(rospy.get_param(
            "~room_slalom_trigger_m", 0.78))
        self.room_slalom_lane_half_m = float(rospy.get_param(
            "~room_slalom_lane_half_m", 0.30))
        self.room_slalom_side_offset_m = float(rospy.get_param(
            "~room_slalom_side_offset_m", 0.27))
        self.room_slalom_forward_offset_m = float(rospy.get_param(
            "~room_slalom_forward_offset_m", 0.38))
        self.room_slalom_complete_radius_m = float(rospy.get_param(
            "~room_slalom_complete_radius_m", 0.18))
        self.room_slalom_hold_s = float(rospy.get_param(
            "~room_slalom_hold_s", 1.25))
        self.room_slalom_max_lateral = float(rospy.get_param(
            "~room_slalom_max_lateral_mps", 0.12))
        self.room_slalom_speed_mps = float(rospy.get_param(
            "~room_slalom_speed_mps", 0.34))
        self.room_slalom_turn_rps = float(rospy.get_param(
            "~room_slalom_turn_rps", 1.02))
        self.room_slalom_map_clearance = float(rospy.get_param(
            "~room_slalom_map_clearance_m", 0.18))
        self.room_slalom_line_clearance = float(rospy.get_param(
            "~room_slalom_line_clearance_m", 0.14))
        self.room_slalom_side_min = float(rospy.get_param(
            "~room_slalom_side_min_m", 0.24))
        self.room_emergency_front_abort = float(rospy.get_param(
            "~room_emergency_front_abort_m", 0.12))
        self.room_slalom_abort_front = self.room_emergency_front_abort
        self.room_wall_escape_lateral = float(rospy.get_param(
            "~room_wall_escape_lateral_mps", 0.075))
        self.room_slalom_path_skip_m = float(rospy.get_param(
            "~room_slalom_path_skip_m", 0.55))
        self.room_slalom_ignore_radius_m = float(rospy.get_param(
            "~room_slalom_ignore_radius_m", 0.45))
        self.room_slalom_ignore_s = float(rospy.get_param(
            "~room_slalom_ignore_s", 2.0))
        self.room_slalom_path_gate_m = float(rospy.get_param(
            "~room_slalom_path_gate_m", 0.20))
        self.room_slalom_path_window_m = float(rospy.get_param(
            "~room_slalom_path_window_m", 1.15))
        self.room_slalom_arc_forward = float(rospy.get_param(
            "~room_slalom_arc_forward_m", 0.42))
        self.room_slalom_arc_side = float(rospy.get_param(
            "~room_slalom_arc_side_m", 0.20))
        self.room_slalom_arc_reach = float(rospy.get_param(
            "~room_slalom_arc_reach_m", 0.15))
        self.room_slalom_arc_timeout = float(rospy.get_param(
            "~room_slalom_arc_timeout_s", 3.2))
        self.final_mode_pub = rospy.Publisher(
            rospy.get_param("~final_approach_topic",
                            "/two_stage/final_approach"),
            Bool, queue_size=1, latch=True)
        self.final_mode_pub.publish(Bool(data=False))
        self.slalom_mode_pub = rospy.Publisher(
            rospy.get_param("~room_slalom_topic", "/two_stage/slalom_active"),
            Bool, queue_size=1, latch=True)
        self.slalom_mode_pub.publish(Bool(data=False))

        self.route_index = 0
        self.finished = False
        self.goal_stage = 1
        self.waypoint_enabled = False
        self._activate_route_point(clear_path=True)
        self.route_ready = True
        rospy.logwarn(
            "TWO_STAGE_ROUTE_READY points=%s",
            " -> ".join(point["name"] for point in self.route_points))

    @staticmethod
    def _validate_route(configured):
        if not isinstance(configured, list) or not configured:
            raise ValueError("route_points must be a non-empty list")
        route = []
        for index, raw in enumerate(configured):
            if not isinstance(raw, dict):
                raise ValueError("route point %d is not a dictionary" % index)
            route.append({
                "name": str(raw.get("name", "point_%d" % index)),
                "x": float(raw["x"]),
                "y": float(raw["y"]),
                "yaw": float(raw.get("yaw", 0.0)),
                "tolerance": max(0.06, float(raw.get("tolerance", 0.12))),
                "require_yaw": bool(raw.get("require_yaw", False)),
            })
        return route

    def current_route_point(self):
        return self.route_points[self.route_index]

    def current_target_world(self):
        point = self.current_route_point()
        return point["x"], point["y"]

    def current_target_label(self):
        return "route point %s" % self.current_route_point()["name"]

    def current_target_tolerance(self):
        return self.current_route_point()["tolerance"]

    def _activate_route_point(self, clear_path):
        point = self.current_route_point()
        self.goal_x = point["x"]
        self.goal_y = point["y"]
        self.goal_yaw = point["yaw"]
        self.require_goal_yaw = point["require_yaw"]
        self.active_goal = (point["x"], point["y"])
        self.goal_enter_time = None
        self.route_enter_time = None
        if clear_path:
            self.path_world = []
            self.path_index = 0
        self.last_cmd = (0.0, 0.0)
        self.last_vy = 0.0
        self.requested_vy = 0.0

    def control_loop(self, event):
        if not self.route_ready:
            self.publish_zero("TWO_STAGE_INITIALIZING")
            return
        if (not self.room_profile_active and self.pose is not None and
                self.pose[1] <= self.room_trigger_y):
            self._activate_room_profile()
        elif (not self.room_profile_active and
              not self.doorway_profile_active and self.route_index > 0):
            self._activate_doorway_profile()
        self._update_final_approach()
        if self.room_profile_active and self._update_room_dynamic_layer():
            return
        super(FrontFirstTwoStageRouteTest, self).control_loop(event)

    def _activate_doorway_profile(self):
        self.doorway_profile_active = True
        self.cruise_speed = self.doorway_cruise_speed
        self.max_forward_cmd = self.doorway_cruise_speed
        self.max_lateral_cmd = self.doorway_max_lateral
        self.front_angle_deg = self.doorway_front_angle
        self.avoid_influence = self.doorway_front_influence
        self.avoid_stop = self.doorway_front_stop
        self.lateral_trigger = 0.29
        self.lateral_side_need = 0.18
        self.max_turn_cmd = max(self.max_turn_cmd, 0.82)
        self.minimum_curve_speed = min(self.minimum_curve_speed, 0.045)
        self.last_cmd = (min(self.last_cmd[0], self.doorway_cruise_speed),
                         self.last_cmd[1])
        rospy.logwarn(
            "TWO_STAGE_DOORWAY_PROFILE_ACTIVE cruise=%.2f front_angle=%.1f "
            "route=%s",
            self.cruise_speed, self.front_angle_deg,
            self.current_route_point()["name"])

    def _update_final_approach(self):
        if self.final_approach_active or self.pose is None:
            return
        point = self.current_route_point()
        if point["name"] != "d3":
            return
        requested_dist = math.hypot(
            self.pose[0] - point["x"], self.pose[1] - point["y"])
        if requested_dist > self.final_approach_radius:
            return
        self.final_approach_active = True
        self.active_goal = (point["x"], point["y"])
        self.path_world = [(self.pose[0], self.pose[1]), self.active_goal]
        self.path_index = 0
        self.cruise_speed = min(self.cruise_speed, self.final_approach_speed)
        self.max_forward_cmd = self.cruise_speed
        self.max_lateral_cmd = min(self.max_lateral_cmd, 0.035)
        self.room_avoid_sign = 0.0
        self.final_mode_pub.publish(Bool(data=True))
        rospy.logwarn(
            "TWO_STAGE_FINAL_APPROACH_ACTIVE requested_goal=(%.3f,%.3f) "
            "dist=%.3f speed=%.2f dynamic_replan=false",
            point["x"], point["y"], requested_dist, self.cruise_speed)

    def _activate_room_profile(self):
        self.room_profile_active = True
        self.cruise_speed = self.room_cruise_speed
        self.max_forward_cmd = self.room_cruise_speed
        self.max_turn_cmd = self.room_max_turn
        self.max_lateral_cmd = self.room_max_lateral
        self.front_angle_deg = self.room_front_angle
        self.avoid_influence = self.room_avoid_influence
        self.avoid_stop = self.room_avoid_stop
        self.lateral_trigger = self.room_lateral_trigger
        self.lateral_side_need = self.room_lateral_side_need
        self.avoid_turn_gain = self.room_avoid_turn_gain
        self.side_stop_m = self.room_side_stop
        self.side_slow_m = self.room_side_slow
        self.wall_center_gain = 0.12
        self.forward_accel = min(self.forward_accel, 0.55)
        self.forward_decel = max(self.forward_decel, 1.10)
        self.last_cmd = (min(self.last_cmd[0], self.room_cruise_speed),
                         self.last_cmd[1])
        rospy.logwarn(
            "TWO_STAGE_ROOM_PROFILE_ACTIVE y=%.3f cruise=%.2f "
            "front=(angle=%.1f influence=%.2f stop=%.2f) side_stop=%.2f",
            self.pose[1], self.cruise_speed, self.front_angle_deg,
            self.avoid_influence, self.avoid_stop, self.side_stop_m)

    def cb_scan(self, msg):
        super(FrontFirstTwoStageRouteTest, self).cb_scan(msg)
        if not self.room_profile_active:
            return
        semantic = classify_scan(
            msg, cone_base_pad_m=self.room_cone_base_pad,
            front_angle_deg=self.front_angle_deg,
            side_min_deg=self.side_angle_min_deg,
            side_max_deg=self.side_angle_max_deg,
            wall_front_relax_m=(self.final_wall_front_relax
                                if self.final_approach_active
                                else self.room_wall_front_relax),
            wall_side_relax_m=(self.final_wall_side_relax
                               if self.final_approach_active
                               else self.room_wall_side_relax))
        if self.final_approach_active:
            cone_raw = semantic["cone_front"] + self.room_cone_base_pad
            if (math.isfinite(cone_raw) and
                    math.isfinite(semantic["wall_front"]) and
                    abs(cone_raw - semantic["wall_front"]) <= 0.09):
                semantic["cone_front"] = float("inf")
                semantic["effective_front"] = (
                    semantic["wall_front"] + self.final_wall_front_relax)
        self.front = semantic["effective_front"]
        self.left = semantic["effective_left"]
        self.right = semantic["effective_right"]
        self.room_cone_observations = semantic["cone_observations"]
        self.room_cone_front = semantic["cone_front"]
        self.room_wall_front = semantic["wall_front"]
        if self.room_cone_observations:
            self.room_dynamic_last_seen = rospy.Time.now()
        rospy.logwarn_throttle(
            0.8, "TWO_STAGE_LIDAR_SEMANTIC cones=%d walls=%d "
            "effective=(%.2f,%.2f,%.2f) cone_front=%.2f wall_front=%.2f",
            semantic["cone_clusters"], semantic["wall_clusters"],
            self.front, self.left, self.right, semantic["cone_front"],
            semantic["wall_front"])

    def _cone_world_points(self):
        if self.pose is None:
            return []
        x, y, yaw = self.pose
        points = []
        for cone in self.room_cone_observations:
            if (cone["range"] > self.room_dynamic_observation_range or
                    abs(cone["angle_deg"]) > 75.0 or cone["count"] < 2):
                continue
            bearing = math.radians(cone["angle_deg"])
            # The scan hits the cone body; shift a little toward its center.
            distance = cone["range"] + 0.04
            points.append((x + math.cos(yaw + bearing) * distance,
                           y + math.sin(yaw + bearing) * distance))
            return points

    def _path_heading_vector(self):
        if self.pose is None:
            return 1.0, 0.0
        x, y, _yaw = self.pose
        if self.path_world:
            start = max(self.path_index, 0)
            stop = min(len(self.path_world), start + 90)
            for i in range(start, stop):
                px, py = self.path_world[i]
                dx = px - x
                dy = py - y
                length = math.hypot(dx, dy)
                if length >= 0.30:
                    return dx / length, dy / length
        dx = self.active_goal[0] - x
        dy = self.active_goal[1] - y
        length = math.hypot(dx, dy)
        if length <= 1.0e-6:
            return math.cos(self.pose[2]), math.sin(self.pose[2])
        return dx / length, dy / length

    def _slalom_candidate(self):
        if self.pose is None:
            return None
        x, y, yaw = self.pose
        now = rospy.Time.now()
        best = None
        for cone in self.room_cone_observations:
            if cone["count"] < 2 or cone["range"] > self.room_slalom_trigger_m:
                continue
            bearing = math.radians(cone["angle_deg"])
            local_x = math.cos(bearing) * cone["range"]
            local_y = math.sin(bearing) * cone["range"]
            if local_x < 0.16 or abs(local_y) > self.room_slalom_lane_half_m:
                continue
            cone_center_range = cone["range"] + 0.05
            world_x = x + math.cos(yaw + bearing) * cone_center_range
            world_y = y + math.sin(yaw + bearing) * cone_center_range
            if (self.room_slalom_ignore_world is not None and
                    now < self.room_slalom_ignore_until and
                    math.hypot(world_x - self.room_slalom_ignore_world[0],
                               world_y - self.room_slalom_ignore_world[1]) <=
                    self.room_slalom_ignore_radius_m):
                continue
            path_dist = self._distance_to_upcoming_path((world_x, world_y))
            if path_dist > self.room_slalom_path_gate_m:
                rospy.logwarn_throttle(
                    0.5,
                    "TWO_STAGE_SLALOM_IGNORE_OFF_PATH cone=(%.2f,%.2f) "
                    "path_dist=%.2f gate=%.2f",
                    local_x, local_y, path_dist, self.room_slalom_path_gate_m)
                continue
            score = local_x + 0.35 * abs(local_y)
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "range": cone["range"],
                    "angle_deg": cone["angle_deg"],
                    "local_x": local_x,
                    "local_y": local_y,
                    "world": (world_x, world_y),
                    "path_dist": path_dist,
                }
        return best

    def _distance_to_upcoming_path(self, point):
        if not self.path_world or self.pose is None:
            return 0.0
        x, y, _yaw = self.pose
        start = max(0, self.path_index)
        best = float("inf")
        travelled = 0.0
        prev = (x, y)
        for i in range(start, len(self.path_world)):
            pt = self.path_world[i]
            travelled += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt
            if travelled > self.room_slalom_path_window_m:
                break
            best = min(best, math.hypot(pt[0] - point[0],
                                        pt[1] - point[1]))
        return best

    def _choose_slalom_sign(self, cone):
        if self.room_slalom_active and self.room_slalom_sign != 0.0:
            return self.room_slalom_sign
        if cone["local_y"] > 0.06 and self.right > self.room_side_stop + 0.04:
            return -1.0
        if cone["local_y"] < -0.06 and self.left > self.room_side_stop + 0.04:
            return 1.0
        if self.left > self.right + 0.04:
            return 1.0
        if self.right > self.left + 0.04:
            return -1.0
        return 1.0 if cone["angle_deg"] <= 0.0 else -1.0

    def _side_clearance_for_sign(self, sign):
        return self.left if sign > 0.0 else self.right

    def _map_clearance_at(self, point):
        if self.grid is None:
            return 999.0
        cell = self.grid.world_to_map(point[0], point[1])
        if cell is None:
            return 0.0
        return self.grid.clearance_m(cell[0], cell[1])

    def _line_to_point_safe(self, point, clearance_m):
        if self.grid is None or self.pose is None:
            return True
        start = self.grid.world_to_map(self.pose[0], self.pose[1])
        goal = self.grid.world_to_map(point[0], point[1])
        if start is None or goal is None:
            return False
        return self.grid.line_is_safe(start, goal, clearance_m)

    def _line_world_safe(self, start_world, end_world, clearance_m):
        if self.grid is None:
            return True
        start = self.grid.world_to_map(start_world[0], start_world[1])
        goal = self.grid.world_to_map(end_world[0], end_world[1])
        if start is None or goal is None:
            return False
        return self.grid.line_is_safe(start, goal, clearance_m)

    def _make_slalom_arc(self, cone, sign):
        cx, cy = cone["world"]
        dir_x, dir_y = self._path_heading_vector()
        perp_x, perp_y = -dir_y, dir_x
        rel_x = cx - self.pose[0]
        rel_y = cy - self.pose[1]
        cone_lateral = rel_x * perp_x + rel_y * perp_y
        center_x = cx - perp_x * cone_lateral
        center_y = cy - perp_y * cone_lateral
        a = self.room_slalom_arc_forward
        b = self.room_slalom_arc_side
        arc = []
        # Half-ellipse in the path frame: behind side -> side apex -> ahead center.
        for deg in (145.0, 112.0, 80.0, 48.0, 18.0, 0.0):
            theta = math.radians(deg)
            progress = (145.0 - deg) / 145.0
            along = a * math.cos(theta)
            side = cone_lateral * (1.0 - progress) + sign * b * math.sin(theta)
            point = (center_x + dir_x * along + perp_x * side,
                     center_y + dir_y * along + perp_y * side)
            if self.pose is not None:
                rel_x = point[0] - self.pose[0]
                rel_y = point[1] - self.pose[1]
                if rel_x * dir_x + rel_y * dir_y < 0.04:
                    continue
            arc.append(point)
        return arc

    def _slalom_point_safe(self, point, sign):
        side_clear = self._side_clearance_for_sign(sign)
        if math.isfinite(side_clear) and side_clear < self.room_slalom_side_min:
            return False
        if self._map_clearance_at(point) < self.room_slalom_map_clearance:
            return False
        if not self._line_to_point_safe(point, self.room_slalom_line_clearance):
            return False
        return True

    def _slalom_arc_safe(self, arc, sign):
        if not arc or self.pose is None:
            return False
        previous = (self.pose[0], self.pose[1])
        for point in arc:
            if not self._slalom_point_safe(point, sign):
                return False
            if not self._line_world_safe(previous, point,
                                         self.room_slalom_line_clearance):
                return False
            previous = point
        return True

    def _advance_path_after_slalom(self):
        if self.pose is None or not self.path_world:
            return
        x, y, _yaw = self.pose
        start = max(0, self.path_index)
        best = start
        best_d2 = float("inf")
        for i in range(start, len(self.path_world)):
            px, py = self.path_world[i]
            d2 = (px - x) ** 2 + (py - y) ** 2
            if d2 < best_d2:
                best = i
                best_d2 = d2

        skip = self.room_slalom_path_skip_m
        accum = 0.0
        advanced = best
        prev = (x, y)
        for i in range(best, len(self.path_world)):
            pt = self.path_world[i]
            accum += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt
            advanced = i
            if accum >= skip:
                break
        old_index = self.path_index
        self.path_index = max(self.path_index, advanced)
        rospy.logwarn(
            "TWO_STAGE_SLALOM_COMMIT path_index=%d->%d pose=(%.2f,%.2f)",
            old_index, self.path_index, x, y)

    def _clear_slalom(self, reason, commit=False):
        if commit:
            self._advance_path_after_slalom()
            if self.room_slalom_cone_world is not None:
                self.room_slalom_ignore_world = self.room_slalom_cone_world
                self.room_slalom_ignore_until = rospy.Time.now() + rospy.Duration(
                    self.room_slalom_ignore_s)
        self._set_slalom_active(False)
        self.room_slalom_point = None
        self.room_slalom_arc = []
        self.room_slalom_arc_index = 0
        self.room_slalom_cone_world = None
        self.room_slalom_sign = 0.0
        rospy.logwarn_throttle(0.5, "TWO_STAGE_SLALOM_CLEAR_REASON %s", reason)

    def _set_slalom_active(self, active):
        if active == self.room_slalom_active:
            return
        self.room_slalom_active = active
        if active:
            self.max_lateral_cmd = max(self.max_lateral_cmd,
                                       self.room_slalom_max_lateral)
        elif self.room_profile_active and not self.final_approach_active:
            self.max_lateral_cmd = self.room_max_lateral
        self.slalom_mode_pub.publish(Bool(data=active))
        rospy.logwarn("TWO_STAGE_SLALOM_%s",
                      "ACTIVE" if active else "CLEAR")

    def _update_room_slalom(self):
        if (not self.room_profile_active or self.final_approach_active or
                not self.room_slalom_enabled or self.pose is None):
            self._set_slalom_active(False)
            return None

        if self.front <= self.room_slalom_abort_front:
            if self.room_slalom_active:
                rospy.logwarn_throttle(
                    0.5, "TWO_STAGE_SLALOM_ABORT front=%.3f left=%.3f right=%.3f",
                    self.front, self.left, self.right)
            self._clear_slalom("front abort", commit=False)
            return None

        now = rospy.Time.now()
        if self.room_slalom_active and self.room_slalom_arc:
            while self.room_slalom_arc_index < len(self.room_slalom_arc):
                point = self.room_slalom_arc[self.room_slalom_arc_index]
                dist = math.hypot(self.pose[0] - point[0],
                                  self.pose[1] - point[1])
                if dist > self.room_slalom_arc_reach:
                    break
                self.room_slalom_arc_index += 1
            if self.room_slalom_arc_index >= len(self.room_slalom_arc):
                self._clear_slalom("ellipse arc complete", commit=True)
                return None
            if now > self.room_slalom_until:
                self._clear_slalom("ellipse arc timeout", commit=False)
                return None
            point = self.room_slalom_arc[self.room_slalom_arc_index]
            if self._slalom_point_safe(point, self.room_slalom_sign):
                return point
            rospy.logwarn_throttle(
                0.5, "TWO_STAGE_SLALOM_ABORT unsafe arc point "
                "index=%d target=(%.2f,%.2f) clearance=%.2f",
                self.room_slalom_arc_index, point[0], point[1],
                self._map_clearance_at(point))
            self._clear_slalom("unsafe arc point", commit=False)

        cone = self._slalom_candidate()
        if cone is None:
            self._clear_slalom("no candidate", commit=False)
            return None

        preferred = self._choose_slalom_sign(cone)
        candidates = [preferred, -preferred]
        sign = 0.0
        arc = None
        for candidate_sign in candidates:
            candidate = self._make_slalom_arc(cone, candidate_sign)
            if self._slalom_arc_safe(candidate, candidate_sign):
                sign = candidate_sign
                arc = candidate
                break
            probe = candidate[-1] if candidate else cone["world"]
            rospy.logwarn_throttle(
                0.5, "TWO_STAGE_SLALOM_REJECT sign=%+.0f arc_end=(%.2f,%.2f) "
                "map_clear=%.2f side=%.2f",
                candidate_sign, probe[0], probe[1],
                self._map_clearance_at(probe),
                self._side_clearance_for_sign(candidate_sign))
        if arc is None:
            self._clear_slalom("all candidates rejected", commit=False)
            return None

        self.room_slalom_sign = sign
        self.room_slalom_arc = arc
        self.room_slalom_arc_index = 0
        self.room_slalom_point = arc[0]
        self.room_slalom_cone_world = cone["world"]
        self.room_slalom_until = now + rospy.Duration(
            self.room_slalom_arc_timeout)
        self._set_slalom_active(True)
        rospy.logwarn_throttle(
            0.35,
            "TWO_STAGE_SLALOM_ELLIPSE sign=%+.0f cone=(x=%.2f,y=%.2f,r=%.2f) "
            "path_dist=%.2f points=%d first=(%.2f,%.2f) last=(%.2f,%.2f)",
            sign, cone["local_x"], cone["local_y"], cone["range"],
            cone["path_dist"], len(arc), arc[0][0], arc[0][1],
            arc[-1][0], arc[-1][1])
        return arc[0]

    def select_target(self):
        if self.room_slalom_active:
            self._clear_slalom("disabled; cones are ordinary obstacles",
                               commit=False)
        return super(FrontFirstTwoStageRouteTest, self).select_target()

    def _cones_intersect_path(self, cone_points):
        if not cone_points or not self.path_world:
            return False
        path = self.path_world[self.path_index:self.path_index + 180]
        trigger2 = self.room_dynamic_path_trigger ** 2
        for cx, cy in cone_points:
            for px, py in path:
                if (px - cx) ** 2 + (py - cy) ** 2 <= trigger2:
                    return True
        return False

    def _rebuild_dynamic_map(self, cone_points, reason):
        self.publish_zero("ROOM_DYNAMIC_REPLAN")
        self.grid.clear_dynamic_blocks()
        added = 0
        for wx, wy in cone_points:
            added += self.grid.add_dynamic_block(
                wx, wy, self.room_dynamic_radius)
        self.grid.compute_distance_field()
        self.planner.roadmaps.clear()
        self.path_world = []
        self.path_index = 0
        self.last_plan_time = rospy.Time(0)
        self.room_dynamic_last_replan = rospy.Time.now()
        rospy.logwarn(
            "TWO_STAGE_DYNAMIC_MAP reason=%s cones=%d cells=%d radius=%.2f",
            reason, len(cone_points), added, self.room_dynamic_radius)
        self.plan_from_current_pose(reason, force=True)

    def _update_room_dynamic_layer(self):
        if (self.final_approach_active or
                not self.room_dynamic_replan_enabled or self.grid is None or
                self.planner is None or self.pose is None):
            return False
        now = rospy.Time.now()
        since_replan = (now - self.room_dynamic_last_replan).to_sec()
        cone_points = self._cone_world_points()
        path_blocked = self._cones_intersect_path(cone_points)
        if path_blocked:
            if self.room_dynamic_blocked_since is None:
                self.room_dynamic_blocked_since = now
        else:
            self.room_dynamic_blocked_since = None
        confirmed = (self.room_dynamic_blocked_since is not None and
                     (now - self.room_dynamic_blocked_since).to_sec() >=
                     self.room_dynamic_confirm_s)
        stuck = ((now - self.last_progress_time).to_sec() >=
                 self.room_dynamic_stuck_replan_s and
                 self.distance_to_active_goal() >
                 self.current_target_tolerance() + 0.16)
        if (stuck and cone_points and
                since_replan >= self.room_dynamic_replan_interval):
            self._rebuild_dynamic_map(cone_points, "room stuck with cones")
            self.room_dynamic_blocked_since = None
            self.last_progress_time = now
            return True
        if (since_replan >= self.room_dynamic_replan_interval and confirmed):
            self._rebuild_dynamic_map(cone_points, "room cone blocks path")
            self.room_dynamic_blocked_since = None
            return True
        if (self.grid.dynamic_blocked and
                (now - self.room_dynamic_last_seen).to_sec() >=
                self.room_dynamic_ttl and
                since_replan >= self.room_dynamic_replan_interval):
            self._rebuild_dynamic_map([], "room cone layer expired")
            return True
        return False

    def compute_cmd(self, target):
        if self.room_profile_active and self.front <= self.room_emergency_front_abort:
            side_sign = 1.0 if self.left >= self.right else -1.0
            side_clear = self.left if side_sign > 0.0 else self.right
            self.room_avoid_sign = side_sign
            self.room_avoid_sign_until = rospy.Time.now() + rospy.Duration(0.45)
            self.requested_vy = 0.0
            rospy.logwarn_throttle(
                0.4, "TWO_STAGE_WALL_ESCAPE front=%.3f sign=%+.0f "
                "side_clear=%.3f vy=%.3f",
                self.front, side_sign, side_clear, self.requested_vy)
            return 0.0, 0.0

        if not self.room_profile_active or self.front >= self.avoid_influence:
            cmd = super(FrontFirstTwoStageRouteTest, self).compute_cmd(target)
            if self.room_slalom_active:
                self._apply_slalom_feedforward(target)
                cmd = (min(cmd[0], self.room_slalom_speed_mps),
                       clamp(cmd[1], -self.room_slalom_turn_rps,
                             self.room_slalom_turn_rps))
            return cmd

        now = rospy.Time.now()
        chosen_clearance = (self.left if self.room_avoid_sign > 0.0
                            else self.right)
        opposite_clearance = (self.right if self.room_avoid_sign > 0.0
                              else self.left)
        chosen_blocked = (self.room_avoid_sign != 0.0 and
                          chosen_clearance < self.lateral_side_need and
                          opposite_clearance > chosen_clearance + 0.05)
        if self.room_slalom_active and self.room_slalom_sign != 0.0:
            self.room_avoid_sign = self.room_slalom_sign
            self.room_avoid_sign_until = now + rospy.Duration(
                self.room_slalom_hold_s)
        elif (self.room_avoid_sign == 0.0 or
                now >= self.room_avoid_sign_until or chosen_blocked):
            self.room_avoid_sign = 1.0 if self.left >= self.right else -1.0
            self.room_avoid_sign_until = now + rospy.Duration(
                self.room_avoid_hold_s)
            rospy.logwarn_throttle(
                0.5, "TWO_STAGE_CONE_SIDE sign=%+.0f hold=%.2fs scan=(%.2f,%.2f,%.2f)",
                self.room_avoid_sign, self.room_avoid_hold_s,
                self.front, self.left, self.right)

        original_left, original_right = self.left, self.right
        if self.room_avoid_sign > 0.0 and self.left < self.right:
            self.left = self.right + 1.0e-3
        elif self.room_avoid_sign < 0.0 and self.right < self.left:
            self.right = self.left + 1.0e-3
        try:
            cmd = super(FrontFirstTwoStageRouteTest, self).compute_cmd(target)
            if self.room_slalom_active:
                self._apply_slalom_feedforward(target)
                cmd = (min(cmd[0], self.room_slalom_speed_mps),
                       clamp(cmd[1], -self.room_slalom_turn_rps,
                             self.room_slalom_turn_rps))
            return cmd
        finally:
            self.left, self.right = original_left, original_right

    def publish_cmd(self, vx, wz):
        if self.room_profile_active and not self.final_approach_active:
            if abs(self.requested_vy) > 1.0e-4:
                rospy.logwarn_throttle(
                    0.5,
                    "TWO_STAGE_LATERAL_BLOCKED_IN_ROOM requested_vy=%.3f",
                    self.requested_vy)
            self.requested_vy = 0.0
            self.last_vy = 0.0
        super(FrontFirstTwoStageRouteTest, self).publish_cmd(vx, wz)

    def _apply_slalom_feedforward(self, target):
        if self.pose is None or self.room_slalom_sign == 0.0:
            return
        _x, _y, yaw = self.pose
        dx = target[0] - self.pose[0]
        dy = target[1] - self.pose[1]
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        side_clearance = (self.left if self.room_slalom_sign > 0.0
                          else self.right)
        if side_clearance <= self.room_side_stop + 0.035:
            return
        desired = clamp(0.48 * local_y,
                        -self.room_slalom_max_lateral,
                        self.room_slalom_max_lateral)
        if desired * self.room_slalom_sign < 0.0:
            desired = self.room_slalom_sign * min(
                abs(desired), self.room_slalom_max_lateral)
        if abs(desired) < 0.035:
            desired = self.room_slalom_sign * 0.035
        if abs(desired) > abs(self.requested_vy):
            self.requested_vy = desired

    def apply_scan_guard(self, cmd):
        if (self.room_slalom_active and
                self.room_cone_front <= self.avoid_stop and
                self.room_wall_front > self.avoid_stop + 0.08):
            original_front = self.front
            self.front = max(self.front, self.avoid_stop + 0.035)
            try:
                guarded = super(FrontFirstTwoStageRouteTest,
                                self).apply_scan_guard(cmd)
            finally:
                self.front = original_front
            if self.room_cone_front <= 0.14:
                return 0.0, guarded[1]
            return guarded
        return super(FrontFirstTwoStageRouteTest, self).apply_scan_guard(cmd)

    def check_goal(self):
        point = self.current_route_point()
        if self.distance_to_active_goal() > point["tolerance"]:
            self.route_enter_time = None
            return False

        self.requested_vy = 0.0
        self.last_vy = 0.0
        if point["require_yaw"]:
            yaw_error = norm_angle(point["yaw"] - self.pose[2])
            if abs(yaw_error) > self.route_yaw_tolerance:
                self.route_enter_time = None
                self.publish_cmd(
                    0.0,
                    clamp(yaw_error, -self.route_yaw_speed,
                          self.route_yaw_speed))
                rospy.logwarn_throttle(
                    0.8, "TWO_STAGE_FINAL_YAW point=%s error=%.1fdeg",
                    point["name"], math.degrees(yaw_error))
                return True

        now = rospy.Time.now()
        if self.route_enter_time is None:
            self.route_enter_time = now
            self.publish_zero("ROUTE_POINT_HOLD")
            rospy.logwarn(
                "TWO_STAGE_POINT_REACHED index=%d/%d name=%s pose=(%.3f,%.3f,%.1fdeg)",
                self.route_index + 1, len(self.route_points), point["name"],
                self.pose[0], self.pose[1], math.degrees(self.pose[2]))
            return True

        final_point = self.route_index == len(self.route_points) - 1
        hold_s = self.route_final_hold_s if final_point else self.route_hold_s
        if (now - self.route_enter_time).to_sec() < hold_s:
            self.publish_zero("ROUTE_POINT_HOLD")
            return True

        if final_point:
            self.finished = True
            self.publish_zero("TWO_STAGE_ROUTE_COMPLETE")
            self.log_status("two-stage route complete; node remains stopped")
            rospy.logwarn("TWO_STAGE_ROUTE_COMPLETE all %d points reached",
                          len(self.route_points))
            return True

        previous = point["name"]
        self.route_index += 1
        self._activate_route_point(clear_path=True)
        following = self.current_route_point()["name"]
        self.publish_zero("ROUTE_POINT_ADVANCE")
        rospy.logwarn("TWO_STAGE_ADVANCE %s -> %s", previous, following)
        self.plan_from_current_pose("route advance", force=True)
        return True


if __name__ == "__main__":
    FrontFirstTwoStageRouteTest().spin()
