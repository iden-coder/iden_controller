#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map-coverage workshop search with the existing fast parking controller."""

import math
import subprocess
import time

import rospy
from nav_msgs.srv import GetMap
from std_msgs.msg import String

from factory_room_delivery_manager import norm_angle
from factory_room_delivery_manager_center_only_v4 import (
    NearbyViewFactoryDeliveryManager,
)
from factory_room_map_coverage_core_v1 import (
    CoverageMapError,
    OccupancyMap,
    build_bottom_right_anchor,
    build_coverage_model,
    choose_parking_wall_motion,
    estimate_wall_tangent_hint,
    evaluate_target_observation,
    update_entry_crossing_stability,
)


VALID_WORKSHOPS = {
    "食品加工车间",
    "日用品加工车间",
    "电子产品生产车间",
}


class MapCoverageFactoryDeliveryManager(NearbyViewFactoryDeliveryManager):
    def __init__(self):
        self._coverage_ocr_sequence = 0
        self._coverage_entry_process = None
        self._coverage_entry_status = ""
        self._coverage_entry_complete = False
        super(MapCoverageFactoryDeliveryManager, self).__init__()
        self.coverage_map_service = rospy.get_param(
            "~coverage_map_service", "/static_map")
        self.coverage_seed_x = float(rospy.get_param(
            "~coverage_room_seed_x", 0.40))
        self.coverage_seed_y = float(rospy.get_param(
            "~coverage_room_seed_y", -2.30))
        self.coverage_search_radius_x = float(rospy.get_param(
            "~coverage_search_radius_x", 3.50))
        self.coverage_search_radius_y = float(rospy.get_param(
            "~coverage_search_radius_y", 2.00))
        self.coverage_standoff = float(rospy.get_param(
            "~coverage_camera_standoff_m", 0.72))
        self.coverage_candidate_spacing = float(rospy.get_param(
            "~coverage_candidate_spacing_m", 0.72))
        self.coverage_bin_spacing = float(rospy.get_param(
            "~coverage_wall_bin_spacing_m", 0.24))
        self.coverage_half_fov_deg = float(rospy.get_param(
            "~coverage_effective_half_fov_deg", 35.0))
        self.coverage_corner_trim = float(rospy.get_param(
            "~coverage_corner_trim_m", 0.30))
        self.coverage_min_clearance = float(rospy.get_param(
            "~coverage_min_static_clearance_m", 0.24))
        self.coverage_observe_s = float(rospy.get_param(
            "~coverage_ocr_observe_s", 2.20))
        self.coverage_non_target_confirmations = int(rospy.get_param(
            "~coverage_non_target_confirmations", 2))
        self.coverage_map_wait_s = float(rospy.get_param(
            "~coverage_map_wait_s", 12.0))
        self.coverage_retry_pause_s = float(rospy.get_param(
            "~coverage_retry_pause_s", 0.8))
        self.coverage_max_passes = int(rospy.get_param(
            "~coverage_max_passes", 0))
        self.coverage_lidar_correction_deg = float(rospy.get_param(
            "~coverage_lidar_correction_max_deg", 14.0))
        self.coverage_target_center_gate_ratio = float(rospy.get_param(
            "~coverage_target_center_gate_ratio", 0.18))
        self.coverage_target_edge_margin_px = float(rospy.get_param(
            "~coverage_target_edge_margin_px", 20.0))
        self.coverage_target_local_lateral_max = float(rospy.get_param(
            "~coverage_target_local_lateral_max_m", 0.28))
        self.coverage_target_side_margin = float(rospy.get_param(
            "~coverage_target_side_margin_m", 0.035))
        self.coverage_hint_min_improvement = float(rospy.get_param(
            "~coverage_hint_min_improvement_m", 0.05))
        self.coverage_hint_min_goal_move = float(rospy.get_param(
            "~coverage_hint_min_goal_move_m", 0.21))
        self.coverage_preferred_wall_bonus = float(rospy.get_param(
            "~coverage_preferred_wall_bonus", 5.0))
        self.coverage_d3_anchor_enabled = bool(rospy.get_param(
            "~coverage_d3_anchor_enabled", True))
        self.coverage_d3_right_inset = float(rospy.get_param(
            "~coverage_d3_right_inset_m", 0.42))
        self.coverage_d3_bottom_inset = float(rospy.get_param(
            "~coverage_d3_bottom_inset_m", 0.80))
        self.coverage_d3_pre_approach = float(rospy.get_param(
            "~coverage_d3_pre_approach_m", 0.48))
        self.coverage_d3_defer_reached_views = int(rospy.get_param(
            "~coverage_d3_defer_reached_views", 2))
        self.coverage_parking_reacquire_attempts = int(rospy.get_param(
            "~coverage_parking_reacquire_attempts", 2))
        self.coverage_parking_reverse_speed = float(rospy.get_param(
            "~coverage_parking_reverse_speed_mps", 0.035))
        self.coverage_parking_reverse_max = float(rospy.get_param(
            "~coverage_parking_reverse_max_m", 0.08))
        self.coverage_parking_reverse_rear_clear = float(rospy.get_param(
            "~coverage_parking_reverse_rear_clear_m", 0.18))
        self.coverage_parking_hard_front = float(rospy.get_param(
            "~coverage_parking_hard_front_m", 0.075))
        self.coverage_entry_enabled = bool(rospy.get_param(
            "~coverage_entry_enabled", True))
        self.coverage_entry_launch_pkg = rospy.get_param(
            "~coverage_entry_launch_pkg", "iden_controller")
        self.coverage_entry_launch_file = rospy.get_param(
            "~coverage_entry_launch_file",
            "factory_room_entry_map_coverage_v5.launch")
        self.coverage_entry_node_name = rospy.get_param(
            "~coverage_entry_node_name",
            "/factory_room_entry_map_coverage_v5")
        self.coverage_entry_status_topic = rospy.get_param(
            "~coverage_entry_status_topic",
            "/factory_room_entry_map_coverage_v5/status")
        self.coverage_entry_goal_x = float(rospy.get_param(
            "~coverage_entry_goal_x", -0.85))
        self.coverage_entry_goal_y = float(rospy.get_param(
            "~coverage_entry_goal_y", -1.90))
        self.coverage_entry_goal_yaw = float(rospy.get_param(
            "~coverage_entry_goal_yaw", -math.pi / 2.0))
        self.coverage_entry_trigger_y = float(rospy.get_param(
            "~coverage_entry_trigger_y", -1.78))
        self.coverage_entry_less_than = bool(rospy.get_param(
            "~coverage_entry_less_than", True))
        self.coverage_entry_stable_samples = int(rospy.get_param(
            "~coverage_entry_stable_samples", 3))
        self.coverage_entry_timeout = float(rospy.get_param(
            "~coverage_entry_timeout_s", 150.0))
        self.coverage_entry_map_yaml = rospy.get_param(
            "~coverage_entry_map_yaml", "")
        self.coverage_entry_stop_nodes = rospy.get_param(
            "~coverage_entry_stop_nodes",
            ["/global_first_graph_nav", "/global_first_nav_qr_room_scan"])
        self._coverage_entry_sub = rospy.Subscriber(
            self.coverage_entry_status_topic, String,
            self._coverage_entry_status_callback, queue_size=20)
        self._selection_used_priority_hint = False
        self.coverage_model = None
        rospy.logwarn(
            "MAP_COVERAGE_MANAGER_V5 service=%s standoff=%.2f fov=+/-%.1f "
            "observe=%.1fs passes=%s",
            self.coverage_map_service, self.coverage_standoff,
            self.coverage_half_fov_deg, self.coverage_observe_s,
            "until_timeout" if self.coverage_max_passes <= 0
            else str(self.coverage_max_passes))
        rospy.logwarn(
            "MAP_COVERAGE_ENTRY_GATE_V5 enabled=%s goal=(%.2f,%.2f) "
            "trigger_y=%.2f stable=%d",
            str(self.coverage_entry_enabled).lower(),
            self.coverage_entry_goal_x, self.coverage_entry_goal_y,
            self.coverage_entry_trigger_y,
            self.coverage_entry_stable_samples)

    def _coverage_entry_status_callback(self, msg):
        self._coverage_entry_status = msg.data or ""

    @staticmethod
    def _kill_ros_node(node_name):
        try:
            subprocess.call(
                ["rosnode", "kill", str(node_name)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _stop_coverage_entry_navigation(self):
        node_name = getattr(self, "coverage_entry_node_name", "")
        if node_name:
            self._kill_ros_node(node_name)
        process = getattr(self, "_coverage_entry_process", None)
        self._coverage_entry_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        if hasattr(self, "cmd_pub"):
            self.publish_zero(6)

    def _entry_pose_gate(self, stable_count):
        pose = self.snapshot_pose()
        pose_y = pose[1] if pose is not None else None
        count, ready = update_entry_crossing_stability(
            stable_count, pose_y, self.coverage_entry_trigger_y,
            self.coverage_entry_stable_samples,
            less_than=self.coverage_entry_less_than)
        return count, ready, pose_y

    def _ensure_large_room_entry(self):
        if self._coverage_entry_complete or not self.coverage_entry_enabled:
            return True

        stable_count = 0
        ready = False
        pose_y = None
        for _ in range(max(1, self.coverage_entry_stable_samples)):
            stable_count, ready, pose_y = self._entry_pose_gate(stable_count)
            if ready or stable_count == 0:
                break
            rospy.sleep(0.05)
        if ready:
            self._coverage_entry_complete = True
            rospy.logwarn(
                "MAP_COVERAGE_ENTRY_ALREADY_CROSSED pose_y=%.3f",
                pose_y)
            return True

        self.ocr_control("disable")
        self.target_found.clear()
        self.move_client.cancel_all_goals()
        self._stop_coverage_entry_navigation()
        stop_nodes = self.coverage_entry_stop_nodes
        if isinstance(stop_nodes, str):
            stop_nodes = [stop_nodes]
        for node_name in stop_nodes:
            self._kill_ros_node(node_name)

        self._coverage_entry_status = ""
        command = [
            "roslaunch", self.coverage_entry_launch_pkg,
            self.coverage_entry_launch_file,
            "goal_x:={:.3f}".format(self.coverage_entry_goal_x),
            "goal_y:={:.3f}".format(self.coverage_entry_goal_y),
            "goal_yaw:={:.10f}".format(self.coverage_entry_goal_yaw),
        ]
        if self.coverage_entry_map_yaml:
            command.append("map_yaml:={}".format(
                self.coverage_entry_map_yaml))
        self.publish_state(
            "MAP_COVERAGE_ENTERING_LARGE_ROOM",
            goal_x=self.coverage_entry_goal_x,
            goal_y=self.coverage_entry_goal_y,
            trigger_y=self.coverage_entry_trigger_y)
        rospy.logwarn(
            "MAP_COVERAGE_ENTRY_START goal=(%.3f,%.3f) yaw=%.1fdeg "
            "coverage_locked=true",
            self.coverage_entry_goal_x, self.coverage_entry_goal_y,
            math.degrees(self.coverage_entry_goal_yaw))
        try:
            self._coverage_entry_process = subprocess.Popen(command)
        except Exception as exc:
            rospy.logerr("MAP_COVERAGE_ENTRY_LAUNCH_FAILED error=%s", exc)
            return False

        started = time.time()
        stable_count = 0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and not self._mission_expired():
            stable_count, ready, pose_y = self._entry_pose_gate(stable_count)
            if ready:
                self._coverage_entry_complete = True
                self._stop_coverage_entry_navigation()
                self.clear_navigation_costmaps()
                self.publish_state(
                    "MAP_COVERAGE_LARGE_ROOM_ENTERED",
                    pose_y=pose_y, trigger_y=self.coverage_entry_trigger_y)
                rospy.logwarn(
                    "MAP_COVERAGE_ENTRY_CROSSED pose_y=%.3f trigger_y=%.3f; "
                    "second-room coverage unlocked",
                    pose_y, self.coverage_entry_trigger_y)
                rospy.sleep(0.25)
                return True

            process = self._coverage_entry_process
            if process is not None and process.poll() is not None:
                rospy.logerr(
                    "MAP_COVERAGE_ENTRY_PROCESS_EXIT code=%s status=%s "
                    "pose_y=%s",
                    process.returncode, self._coverage_entry_status,
                    "unknown" if pose_y is None else "%.3f" % pose_y)
                self._stop_coverage_entry_navigation()
                return False
            if time.time() - started >= self.coverage_entry_timeout:
                rospy.logerr(
                    "MAP_COVERAGE_ENTRY_TIMEOUT status=%s pose_y=%s",
                    self._coverage_entry_status,
                    "unknown" if pose_y is None else "%.3f" % pose_y)
                self._stop_coverage_entry_navigation()
                return False
            rospy.logwarn_throttle(
                1.0,
                "MAP_COVERAGE_ENTRY_WAIT pose_y=%s trigger_y=%.3f "
                "stable=%d/%d status=%s",
                "unknown" if pose_y is None else "%.3f" % pose_y,
                self.coverage_entry_trigger_y, stable_count,
                self.coverage_entry_stable_samples,
                self._coverage_entry_status[-80:])
            rate.sleep()

        self._stop_coverage_entry_navigation()
        return False

    def ocr_callback(self, msg):
        super(MapCoverageFactoryDeliveryManager, self).ocr_callback(msg)
        with self.lock:
            self._coverage_ocr_sequence += 1

    def _load_coverage_model(self):
        if self.coverage_model is not None:
            return self.coverage_model
        last_error = None
        for attempt in range(3):
            if rospy.is_shutdown() or self._mission_expired():
                return None
            try:
                rospy.wait_for_service(
                    self.coverage_map_service,
                    timeout=self.coverage_map_wait_s)
                response = rospy.ServiceProxy(
                    self.coverage_map_service, GetMap)()
                msg = response.map
                origin = msg.info.origin.position
                grid = OccupancyMap(
                    msg.info.width, msg.info.height, msg.info.resolution,
                    origin.x, origin.y, msg.data)
                self.coverage_model = build_coverage_model(
                    grid,
                    seed_x=self.coverage_seed_x,
                    seed_y=self.coverage_seed_y,
                    search_radius_x=self.coverage_search_radius_x,
                    search_radius_y=self.coverage_search_radius_y,
                    standoff=self.coverage_standoff,
                    candidate_spacing=self.coverage_candidate_spacing,
                    bin_spacing=self.coverage_bin_spacing,
                    half_fov_deg=self.coverage_half_fov_deg,
                    corner_trim=self.coverage_corner_trim,
                    min_candidate_clearance=self.coverage_min_clearance)
                if self.coverage_d3_anchor_enabled:
                    d3_anchor = build_bottom_right_anchor(
                        grid, self.coverage_model,
                        right_inset=self.coverage_d3_right_inset,
                        bottom_inset=self.coverage_d3_bottom_inset,
                        half_fov_deg=self.coverage_half_fov_deg,
                        min_candidate_clearance=
                        self.coverage_min_clearance)
                    self.coverage_model["candidates"].insert(0, d3_anchor)
                    rospy.logwarn(
                        "MAP_COVERAGE_D3_ANCHOR_READY x=%.3f y=%.3f "
                        "yaw=-90deg pre=%.2fm clearance=%.2fm",
                        d3_anchor["x"], d3_anchor["y"],
                        self.coverage_d3_pre_approach,
                        d3_anchor["clearance"])
                bounds = self.coverage_model["bounds"]
                wall_counts = {
                    wall: sum(
                        1 for candidate in self.coverage_model["candidates"]
                        if candidate["wall"] == wall)
                    for wall in ("left", "top", "right", "bottom")
                }
                self.publish_state(
                    "MAP_COVERAGE_MODEL_READY", bounds=bounds,
                    candidates=len(self.coverage_model["candidates"]),
                    wall_counts=wall_counts)
                rospy.logwarn(
                    "MAP_COVERAGE_MODEL_READY bounds=(%.2f,%.2f)x"
                    "(%.2f,%.2f) bins=%d views=%d walls=%s",
                    bounds["left"], bounds["right"], bounds["bottom"],
                    bounds["top"], len(self.coverage_model["bins"]),
                    len(self.coverage_model["candidates"]), wall_counts)
                return self.coverage_model
            except Exception as exc:
                last_error = exc
                rospy.logwarn(
                    "MAP_COVERAGE_MAP_RETRY attempt=%d/3 error=%s",
                    attempt + 1, exc)
                rospy.sleep(self.coverage_retry_pause_s)
        rospy.logerr("MAP_COVERAGE_MAP_FAILED error=%s", last_error)
        return None

    def _candidate_score(self, candidate, covered_bins, failures):
        new_bins = set(candidate["visible_bins"]) - covered_bins
        pose = self.snapshot_pose()
        distance = 0.0
        turn = 0.0
        if pose is not None:
            distance = math.hypot(
                candidate["x"] - pose[0], candidate["y"] - pose[1])
            turn = abs(norm_angle(candidate["yaw"] - pose[2]))
        failure_count = failures.get(candidate["id"], 0)
        score = (
            4.0 * len(new_bins) - 0.65 * distance - 0.18 * turn +
            0.65 * min(candidate["clearance"], 0.90) -
            3.0 * failure_count)
        return score, len(new_bins), distance

    def _select_view(self, covered_bins, resolved_walls, failures,
                     attempted_views, priority_hint=None,
                     preferred_wall=None, reached_in_pass=0):
        self._selection_used_priority_hint = False
        exhausted_hint_wall = None
        if priority_hint is not None:
            hinted = []
            current_error = abs(
                priority_hint["tangent"] -
                priority_hint["current_tangent"])
            pose = self.snapshot_pose()
            for candidate in self.coverage_model["candidates"]:
                if candidate["wall"] != priority_hint["wall"]:
                    continue
                if candidate["id"] in attempted_views:
                    continue
                tangent_error = abs(
                    candidate["tangent"] - priority_hint["tangent"])
                route_distance = 0.0
                if pose is not None:
                    route_distance = math.hypot(
                        candidate["x"] - pose[0], candidate["y"] - pose[1])
                improvement = current_error - tangent_error
                if improvement < self.coverage_hint_min_improvement:
                    continue
                if route_distance < self.coverage_hint_min_goal_move:
                    continue
                hinted.append((tangent_error, route_distance, candidate))
            if hinted:
                hinted.sort(key=lambda entry: (entry[0], entry[1]))
                self._selection_used_priority_hint = True
                return hinted[0][2]
            exhausted_hint_wall = priority_hint["wall"]
            rospy.logwarn(
                "MAP_COVERAGE_HINT_EXHAUSTED wall=%s current=%.3f "
                "target=%.3f; backward candidates rejected",
                priority_hint["wall"], priority_hint["current_tangent"],
                priority_hint["tangent"])
        mandatory = None
        for candidate in self.coverage_model["candidates"]:
            if (candidate.get("mandatory") and
                    candidate["wall"] not in resolved_walls and
                    candidate["id"] not in attempted_views):
                mandatory = candidate
                break
        if (mandatory is not None and
                reached_in_pass >= self.coverage_d3_defer_reached_views):
            candidate = mandatory
            rospy.logwarn(
                "MAP_COVERAGE_MANDATORY_VIEW_SELECT view=%s x=%.3f "
                "y=%.3f yaw=%.1fdeg reached=%d",
                candidate["id"], candidate["x"], candidate["y"],
                math.degrees(candidate["yaw"]), reached_in_pass)
            return candidate
        if mandatory is not None:
            rospy.logwarn_throttle(
                1.0,
                "MAP_COVERAGE_MANDATORY_DEFERRED view=%s reached=%d/%d; "
                "selecting a nearer informative view first",
                mandatory["id"], reached_in_pass,
                self.coverage_d3_defer_reached_views)
        choices = []
        for candidate in self.coverage_model["candidates"]:
            if (candidate.get("mandatory") and mandatory is not None and
                    reached_in_pass <
                    self.coverage_d3_defer_reached_views):
                continue
            if candidate["wall"] == exhausted_hint_wall:
                continue
            if candidate["wall"] in resolved_walls:
                continue
            if candidate["id"] in attempted_views:
                continue
            score, new_count, distance = self._candidate_score(
                candidate, covered_bins, failures)
            if new_count <= 0:
                continue
            if candidate["wall"] == preferred_wall:
                score += self.coverage_preferred_wall_bonus
            choices.append((score, new_count, -distance, candidate))
        if not choices:
            return mandatory
        choices.sort(key=lambda entry: entry[:3], reverse=True)
        return choices[0][3]

    def _face_wall_safely(self, candidate):
        self.target_found.clear()
        pose = self.snapshot_pose()
        if pose is None:
            return False
        map_error = norm_angle(candidate["yaw"] - pose[2])
        if abs(map_error) > math.radians(3.0):
            self.publish_state(
                "MAP_COVERAGE_WALL_TURN", view=candidate["id"],
                wall=candidate["wall"], error_deg=math.degrees(map_error))
            if not self.rotate_relative(map_error):
                return False
        rospy.sleep(0.35)

        snapshot = self.snapshot_center()
        wall_model = snapshot.get("wall")
        if (wall_model is None or
                time.time() - snapshot.get("scan_time", 0.0) > 1.0):
            return True
        lidar_error = float(wall_model.get("heading_error", 0.0))
        correction_limit = math.radians(self.coverage_lidar_correction_deg)
        if (math.radians(2.2) < abs(lidar_error) <= correction_limit):
            rospy.logwarn(
                "MAP_COVERAGE_LIDAR_CORRECTION view=%s error=%.2fdeg",
                candidate["id"], math.degrees(lidar_error))
            return self.rotate_relative(lidar_error)
        return True

    def _navigate_view(self, candidate):
        if candidate["id"] != "map_corner_d3":
            return self.navigate_once(
                candidate["id"], candidate["x"], candidate["y"],
                candidate["yaw"])

        bounds = self.coverage_model["bounds"]
        pre_y = min(
            bounds["top"] - 0.32,
            candidate["y"] + self.coverage_d3_pre_approach)
        pose = self.snapshot_pose()
        pre_distance = (
            float("inf") if pose is None else
            math.hypot(candidate["x"] - pose[0], pre_y - pose[1]))
        if pre_distance > 0.24:
            self.publish_state(
                "MAP_COVERAGE_D3_PRE_APPROACH",
                x=candidate["x"], y=pre_y,
                final_x=candidate["x"], final_y=candidate["y"])
            if not self.navigate_once(
                    "map_corner_d3_pre", candidate["x"], pre_y,
                    candidate["yaw"]):
                rospy.logwarn(
                    "MAP_COVERAGE_D3_PRE_UNREACHABLE; deferring the "
                    "wall-side anchor until a later coverage pass")
                return False
        self.publish_state(
            "MAP_COVERAGE_D3_FINAL_APPROACH",
            x=candidate["x"], y=candidate["y"], yaw=candidate["yaw"])
        return self.navigate_once(
            candidate["id"], candidate["x"], candidate["y"],
            candidate["yaw"])

    def _target_geometry(self, payload):
        if (payload is None or not payload.get("stable") or
                str(payload.get("label", "")).strip() !=
                self.target_warehouse):
            return None
        bbox = payload.get("bbox")
        width = float(payload.get("image_width", 0.0))
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or width <= 1.0:
            return None
        center = 0.5 * (float(bbox[0]) + float(bbox[2]))
        target = width * self.optical_center_ratio
        return {
            "center": center,
            "target": target,
            "error": center - target,
            "width": width,
            "bbox": [float(value) for value in bbox],
        }

    def _evaluate_target_for_parking(self, geometry):
        snapshot = self.snapshot_center()
        wall_model = snapshot.get("wall")
        wall_fresh = (
            wall_model is not None and
            time.time() - snapshot.get("scan_time", 0.0) <= 1.0)
        wall_distance = (
            float(wall_model.get("distance", self.coverage_standoff))
            if wall_fresh else self.coverage_standoff)
        quality = evaluate_target_observation(
            geometry["bbox"], geometry["width"], geometry["error"],
            wall_distance, self.coverage_half_fov_deg,
            gate_ratio=self.coverage_target_center_gate_ratio,
            edge_margin_px=self.coverage_target_edge_margin_px,
            max_lateral_shift=self.coverage_target_local_lateral_max)
        requested_sign = self.lateral_command_sign * geometry["error"]
        side_clearance = (
            snapshot.get("left_clear", float("inf"))
            if requested_sign > 0.0 else
            snapshot.get("right_clear", float("inf")))
        available_lateral = max(
            0.0, side_clearance - self.lateral_hard_clearance -
            self.coverage_target_side_margin)
        quality["wall_fresh"] = wall_fresh
        quality["wall_distance"] = wall_distance
        quality["side_clearance"] = side_clearance
        quality["available_lateral"] = available_lateral
        quality["accepted"] = bool(
            quality["accepted"] and wall_fresh and
            quality["lateral_shift"] <= available_lateral)
        return quality

    def _target_hint(self, candidate, geometry):
        wall = self.coverage_model["walls"][candidate["wall"]]
        pose = self.snapshot_pose()
        if pose is None:
            current_tangent = candidate["tangent"]
        elif wall["axis"] == "x":
            current_tangent = float(pose[0])
        else:
            current_tangent = float(pose[1])
        wall_distance = float(geometry.get(
            "wall_distance", self.coverage_standoff))
        tangent = estimate_wall_tangent_hint(
            candidate["wall"], current_tangent, geometry["error"],
            geometry["width"], wall_distance,
            self.coverage_half_fov_deg, wall["start"], wall["end"],
            self.coverage_corner_trim)
        return {
            "wall": candidate["wall"],
            "tangent": tangent,
            "current_tangent": current_tangent,
            "error_px": geometry["error"],
            "center_px": geometry["center"],
            "target_px": geometry["target"],
        }

    def _observe_view(self, candidate):
        with self.lock:
            self.latest_ocr = None
            start_sequence = self._coverage_ocr_sequence
        self.target_found.clear()
        self.ocr_control("reset")
        self.ocr_control("enable")
        self.publish_state(
            "MAP_COVERAGE_OBSERVING", view=candidate["id"],
            wall=candidate["wall"], target=self.target_warehouse,
            observe_s=self.coverage_observe_s)

        label_counts = {}
        last_sequence = start_sequence
        target_hint = None
        deadline = time.time() + self.coverage_observe_s
        rate = rospy.Rate(12)
        while (not rospy.is_shutdown() and time.time() < deadline and
               not self._mission_expired()):
            with self.lock:
                sequence = self._coverage_ocr_sequence
                payload = (None if self.latest_ocr is None
                           else dict(self.latest_ocr))
            if sequence != last_sequence:
                last_sequence = sequence
                if payload is not None and payload.get("stable"):
                    label = str(payload.get("label", "")).strip()
                    geometry = self._target_geometry(payload)
                    if geometry is not None:
                        quality = self._evaluate_target_for_parking(
                            geometry)
                        geometry["wall_distance"] = quality["wall_distance"]
                        if quality["accepted"]:
                            self.publish_zero(5)
                            rospy.logwarn(
                                "MAP_COVERAGE_TARGET_FOUND view=%s wall=%s "
                                "target=%s error=%.1fpx shift=%.3fm "
                                "side=%.3fm clipped=false",
                                candidate["id"], candidate["wall"],
                                self.target_warehouse, geometry["error"],
                                quality["lateral_shift"],
                                quality["side_clearance"])
                            return "target", self.target_warehouse
                        target_hint = self._target_hint(
                            candidate, geometry)
                        self.target_found.clear()
                        rospy.logwarn(
                            "MAP_COVERAGE_TARGET_OFFCENTER view=%s wall=%s "
                            "u=%.1f target=%.1f error=%.1fpx clipped=%s "
                            "shift=%.3fm available=%.3fm wall_fresh=%s "
                            "next_t=%.3f",
                            candidate["id"], candidate["wall"],
                            geometry["center"], geometry["target"],
                            geometry["error"],
                            str(quality["clipped"]).lower(),
                            quality["lateral_shift"],
                            quality["available_lateral"],
                            str(quality["wall_fresh"]).lower(),
                            target_hint["tangent"])
                        rate.sleep()
                        continue
                    if label in VALID_WORKSHOPS:
                        label_counts[label] = label_counts.get(label, 0) + 1
                        if (label != self.target_warehouse and
                                label_counts[label] >=
                                self.coverage_non_target_confirmations):
                            rospy.logwarn(
                                "MAP_COVERAGE_NON_TARGET_CONFIRMED view=%s "
                                "wall=%s label=%s count=%d",
                                candidate["id"], candidate["wall"], label,
                                label_counts[label])
                            self.ocr_control("disable")
                            return "non_target", label
            elif self.target_found.is_set():
                # Base callbacks set the event before this subclass validates
                # that the target belongs to the wall in front of the robot.
                self.target_found.clear()
            rate.sleep()
        self.ocr_control("disable")
        self.target_found.clear()
        if target_hint is not None:
            return "target_hint", target_hint
        return "empty", ""

    def find_target_workshop(self):
        if not self._ensure_large_room_entry():
            rospy.logerr(
                "MAP_COVERAGE_LOCKED_ENTRY_NOT_CONFIRMED; no second-room "
                "coverage goal was sent")
            return False
        model = self._load_coverage_model()
        if model is None:
            rospy.logwarn(
                "MAP_COVERAGE_DEGRADED_FALLBACK using legacy observation "
                "points because /static_map was unavailable")
            return super(MapCoverageFactoryDeliveryManager,
                         self).find_target_workshop()

        resolved_walls = set()
        pass_index = 0
        all_bins = set(model["bins"])
        while not rospy.is_shutdown() and not self._mission_expired():
            if (self.coverage_max_passes > 0 and
                    pass_index >= self.coverage_max_passes):
                break
            pass_index += 1
            covered_bins = set()
            for bin_id, entry in model["bins"].items():
                if entry["wall"] in resolved_walls:
                    covered_bins.add(bin_id)
            failures = {}
            attempted_views = set()
            reached_in_pass = 0
            priority_hint = None
            preferred_wall = None
            self.publish_state(
                "MAP_COVERAGE_PASS_START", pass_index=pass_index,
                resolved_walls=sorted(resolved_walls))

            while not rospy.is_shutdown() and not self._mission_expired():
                candidate = self._select_view(
                    covered_bins, resolved_walls, failures, attempted_views,
                    priority_hint=priority_hint,
                    preferred_wall=preferred_wall,
                    reached_in_pass=reached_in_pass)
                if candidate is None:
                    break
                active_hint = (
                    priority_hint
                    if self._selection_used_priority_hint else None)
                used_priority_hint = active_hint is not None
                priority_hint = None
                attempted_views.add(candidate["id"])
                score, new_count, distance = self._candidate_score(
                    candidate, covered_bins, failures)
                self.publish_state(
                    "MAP_COVERAGE_VIEW_SELECT", view=candidate["id"],
                    wall=candidate["wall"], x=candidate["x"],
                    y=candidate["y"], yaw=candidate["yaw"],
                    new_bins=new_count, score=score, distance=distance)
                rospy.logwarn(
                    "MAP_COVERAGE_VIEW_SELECT pass=%d view=%s wall=%s "
                    "new=%d distance=%.2f score=%.2f hinted=%s",
                    pass_index, candidate["id"], candidate["wall"],
                    new_count, distance, score,
                    str(used_priority_hint).lower())

                self.ocr_control("disable")
                self.target_found.clear()
                reached = self._navigate_view(candidate)
                if not reached:
                    failures[candidate["id"]] = (
                        failures.get(candidate["id"], 0) + 1)
                    rospy.logwarn(
                        "MAP_COVERAGE_VIEW_UNREACHABLE view=%s wall=%s; "
                        "selecting another coverage view",
                        candidate["id"], candidate["wall"])
                    priority_hint = active_hint
                    continue
                reached_in_pass += 1
                if not self._face_wall_safely(candidate):
                    failures[candidate["id"]] = (
                        failures.get(candidate["id"], 0) + 1)
                    rospy.logwarn(
                        "MAP_COVERAGE_WALL_TURN_BLOCKED view=%s; "
                        "selecting another view", candidate["id"])
                    priority_hint = active_hint
                    continue

                self.publish_state(
                    "MAP_COVERAGE_VIEW_REACHED", view=candidate["id"],
                    wall=candidate["wall"], x=candidate["x"],
                    y=candidate["y"])
                result, label = self._observe_view(candidate)
                covered_bins.update(candidate["visible_bins"])
                if result == "target":
                    return True
                if result == "target_hint":
                    priority_hint = label
                    preferred_wall = label["wall"]
                    self.publish_state(
                        "MAP_COVERAGE_TARGET_REPOSITION",
                        wall=label["wall"], tangent=label["tangent"],
                        error_px=label["error_px"])
                    continue
                if result == "non_target":
                    # Unlike the QR room, the large-room specification does
                    # not guarantee one workshop sign per wall. Only the
                    # camera-visible bins are covered here; another sign on
                    # the same physical wall must remain searchable.
                    preferred_wall = candidate["wall"]
                    self.publish_state(
                        "MAP_COVERAGE_NON_TARGET_OBSERVED",
                        wall=candidate["wall"], label=label,
                        target=False,
                        covered_view=candidate["id"])
                elif result == "empty":
                    preferred_wall = candidate["wall"]

            unresolved_bins = [
                bin_id for bin_id in all_bins if bin_id not in covered_bins
            ]
            self.publish_state(
                "MAP_COVERAGE_PASS_COMPLETE", pass_index=pass_index,
                reached_views=reached_in_pass,
                covered_bins=len(all_bins) - len(unresolved_bins),
                unresolved_bins=len(unresolved_bins),
                resolved_walls=sorted(resolved_walls))
            rospy.logwarn(
                "MAP_COVERAGE_PASS_COMPLETE pass=%d reached=%d "
                "covered=%d/%d resolved_walls=%s",
                pass_index, reached_in_pass,
                len(all_bins) - len(unresolved_bins), len(all_bins),
                sorted(resolved_walls))
            self.clear_navigation_costmaps()
            rospy.sleep(self.coverage_retry_pause_s)
        return False

    def approach_centered_wall(self):
        self.publish_state(
            "CENTERLINE_WALL_APPROACH",
            wall_target=self.final_wall_distance,
            nose_gap_target=self.final_wall_distance - 0.061,
            bidirectional_recovery=True)
        start_xy = self.snapshot_center()["odom_xy"]
        progress_xy = start_xy
        reverse_start_xy = None
        last_progress = time.time()
        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.time() - start < self.approach_timeout):
            snapshot = self.snapshot_center()
            wall = snapshot["wall"]
            if (wall is None or
                    time.time() - snapshot["scan_time"] > 1.0):
                self.publish_center_command()
                stable = 0
                rate.sleep()
                continue

            odom_xy = snapshot["odom_xy"]
            travelled = self.distance_between(start_xy, odom_xy)
            if self.distance_between(progress_xy, odom_xy) >= 0.010:
                progress_xy = odom_xy
                last_progress = time.time()
            if travelled > self.approach_max_travel:
                self.publish_zero(30)
                rospy.logerr(
                    "CENTERLINE_APPROACH_TRAVEL_LIMIT %.3fm", travelled)
                return False

            wall_distance = float(wall["distance"])
            heading_error = float(wall["heading_error"])
            mode, command_x = choose_parking_wall_motion(
                wall_distance, self.final_wall_distance,
                self.final_wall_tolerance, self.approach_slow_error,
                self.approach_slow_speed, self.approach_fast_speed,
                self.coverage_parking_reverse_speed)
            rear_clear = float(snapshot.get("rear_clear", float("inf")))

            if (snapshot["front_min"] < self.coverage_parking_hard_front and
                    mode != "reverse"):
                self.publish_zero(30)
                rospy.logerr(
                    "CENTERLINE_HARD_FRONT_BLOCKED front=%.3f wall=%.3f",
                    snapshot["front_min"], wall_distance)
                return False

            if mode == "reverse":
                stable = 0
                if reverse_start_xy is None:
                    reverse_start_xy = odom_xy
                reverse_travel = self.distance_between(
                    reverse_start_xy, odom_xy)
                if (rear_clear < self.coverage_parking_reverse_rear_clear or
                        reverse_travel >= self.coverage_parking_reverse_max):
                    self.publish_zero(30)
                    rospy.logerr(
                        "CENTERLINE_REVERSE_RECOVERY_BLOCKED rear=%.3f "
                        "travel=%.3f/%.3f wall=%.3f",
                        rear_clear, reverse_travel,
                        self.coverage_parking_reverse_max, wall_distance)
                    return False
                command_wz = 0.0
            else:
                reverse_start_xy = None
                if abs(heading_error) > self.approach_heading_limit:
                    self.publish_zero(8)
                    if wall_distance <= 0.30:
                        rospy.logerr(
                            "CENTERLINE_CLOSE_HEADING_REJECTED wall=%.3f "
                            "error=%.2fdeg", wall_distance,
                            math.degrees(heading_error))
                        return False
                    if not self.align_real_wall(
                            "CENTERLINE_APPROACH_HEADING_RECOVERY"):
                        return False
                    stable = 0
                    progress_xy = self.snapshot_center()["odom_xy"]
                    last_progress = time.time()
                    continue
                if mode == "hold":
                    stable += 1
                    command_wz = 0.0
                else:
                    stable = 0
                    command_wz = max(
                        -0.045, min(0.045,
                                    self.heading_kp * heading_error))

            self.publish_center_command(x=command_x, wz=command_wz)
            rospy.logwarn_throttle(
                0.30,
                "CENTERLINE_APPROACH_V5 mode=%s wall=%.3f target=%.3f "
                "heading=%.2fdeg front=%.3f rear=%.3f travel=%.3f "
                "stable=%d/%d cmd=(%.3f,%.3f)",
                mode, wall_distance, self.final_wall_distance,
                math.degrees(heading_error), snapshot["front_min"],
                rear_clear, travelled, stable, self.final_stable_frames,
                command_x, command_wz)
            if stable >= self.final_stable_frames:
                self.publish_zero(30)
                self.publish_state(
                    "CENTERLINE_PARKING_SUCCESS",
                    wall_distance=wall_distance,
                    nose_gap_estimate=wall_distance - 0.061,
                    heading_error_deg=math.degrees(heading_error),
                    forward_motion=travelled)
                rospy.logwarn(
                    "CENTERLINE_PARKED wall=%.3fm nose_gap=%.3fm "
                    "heading=%.2fdeg", wall_distance,
                    wall_distance - 0.061,
                    math.degrees(heading_error))
                return True
            if (abs(command_x) > 0.0 and
                    time.time() - last_progress > 3.0):
                self.publish_zero(30)
                rospy.logerr("CENTERLINE_APPROACH_NO_PROGRESS mode=%s", mode)
                return False
            rate.sleep()
        self.publish_zero(30)
        rospy.logerr("CENTERLINE_APPROACH_TIMEOUT")
        return False

    def park_inside_square(self):
        if super(MapCoverageFactoryDeliveryManager,
                 self).park_inside_square():
            return True
        for attempt in range(max(0, self.coverage_parking_reacquire_attempts)):
            if rospy.is_shutdown() or self._mission_expired():
                return False
            self.publish_zero(15)
            self.set_parking_mode(False)
            self.target_found.clear()
            self.ocr_control("disable")
            self.publish_state(
                "MAP_COVERAGE_PARKING_REACQUIRE",
                attempt=attempt + 1,
                max_attempts=self.coverage_parking_reacquire_attempts)
            rospy.logwarn(
                "MAP_COVERAGE_PARKING_REACQUIRE attempt=%d/%d; "
                "returning to wall coverage instead of failing the task",
                attempt + 1, self.coverage_parking_reacquire_attempts)
            if not self.find_target_workshop():
                continue
            if not self.approach_target_wall_with_navigation():
                continue
            if super(MapCoverageFactoryDeliveryManager,
                     self).park_inside_square():
                return True
        return False

    def shutdown(self):
        self._stop_coverage_entry_navigation()
        super(MapCoverageFactoryDeliveryManager, self).shutdown()


if __name__ == "__main__":
    try:
        MapCoverageFactoryDeliveryManager().run()
    except CoverageMapError as exc:
        rospy.logfatal("MAP_COVERAGE_FATAL %s", exc)
