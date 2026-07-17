#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""MoveBase-compatible front-first room navigation with stable cone tracks."""

import math

import rospy
from visualization_msgs.msg import Marker, MarkerArray

from factory_room_global_first_action_bridge import GlobalFirstActionBridge
from front_first_smooth_navigation_v1 import FrontFirstSmoothNavigator
from front_first_stable_dynamic_route_test_v1 import (
    StableObstacleTrack,
    point_segment_distance,
)
from room_lidar_semantics_v1 import classify_scan


class FactoryRoomFrontFirstStableActionBridge(
        GlobalFirstActionBridge, FrontFirstSmoothNavigator):
    """Preserve front-first motion while adding a persistent room obstacle map."""

    def __init__(self):
        self.dynamic_ready = False
        self.indoor_active = False
        self.obstacle_tracks = []
        self.next_track_id = 1
        self.dynamic_layer_dirty = False
        self.dynamic_layer_signature = ()
        self.last_layer_update = rospy.Time(0)
        self.last_dynamic_replan = rospy.Time(0)
        self.static_distance_cells = None
        super(FactoryRoomFrontFirstStableActionBridge, self).__init__()

        self.indoor_trigger_y = float(rospy.get_param(
            "~indoor_trigger_y", -1.75))
        self.indoor_trigger_less_than = bool(rospy.get_param(
            "~indoor_trigger_less_than", True))
        self.first_stage_max_lateral = float(rospy.get_param(
            "~first_stage_max_lateral_mps", 0.055))
        self.room_max_lateral = float(rospy.get_param(
            "~room_max_lateral_mps", 0.0))

        self.cone_base_pad = float(rospy.get_param(
            "~dynamic_cone_base_pad_m", 0.04))
        self.observation_range = float(rospy.get_param(
            "~dynamic_observation_range_m", 1.35))
        self.observation_half_angle = float(rospy.get_param(
            "~dynamic_observation_half_angle_deg", 85.0))
        self.minimum_cluster_points = int(rospy.get_param(
            "~dynamic_min_cluster_points", 3))
        self.maximum_cluster_span = float(rospy.get_param(
            "~dynamic_max_cluster_span_m", 0.22))
        self.wall_rejection_clearance = float(rospy.get_param(
            "~dynamic_wall_rejection_clearance_m", 0.11))
        self.dynamic_candidate_path_gate = float(rospy.get_param(
            "~dynamic_candidate_path_gate_m", 0.34))
        self.association_radius = float(rospy.get_param(
            "~dynamic_association_radius_m", 0.18))
        self.track_smoothing_alpha = float(rospy.get_param(
            "~dynamic_track_smoothing_alpha", 0.30))
        self.confirm_hits = int(rospy.get_param(
            "~dynamic_confirm_hits", 3))
        self.confirm_age_s = float(rospy.get_param(
            "~dynamic_confirm_age_s", 0.12))
        self.pending_ttl_s = float(rospy.get_param(
            "~dynamic_pending_ttl_s", 0.75))
        self.confirmed_ttl_s = float(rospy.get_param(
            "~dynamic_confirmed_ttl_s", 6.0))

        # This is the complete centre exclusion radius and is not inflated
        # again by the static-map distance field.
        self.dynamic_exclusion_radius = float(rospy.get_param(
            "~dynamic_exclusion_radius_m", 0.26))
        self.path_trigger_radius = float(rospy.get_param(
            "~dynamic_path_trigger_radius_m", 0.26))
        self.layer_update_min_interval = float(rospy.get_param(
            "~dynamic_layer_update_min_interval_s", 0.40))
        self.replan_min_dynamic_interval = float(rospy.get_param(
            "~dynamic_replan_min_interval_s", 0.85))
        self.dynamic_stuck_replan_s = float(rospy.get_param(
            "~dynamic_stuck_replan_s", 2.60))
        self.signature_resolution = float(rospy.get_param(
            "~dynamic_signature_resolution_m", 0.05))

        self.marker_pub = rospy.Publisher(
            "~dynamic_obstacles", MarkerArray, queue_size=1, latch=True)
        self.static_distance_cells = list(self.grid.dist_cells)
        self.dynamic_obstacles_enabled = False
        self.max_lateral_cmd = self.first_stage_max_lateral
        self.requested_vy = 0.0
        self.last_vy = 0.0
        self.dynamic_ready = True
        rospy.logwarn(
            "ROOM_FRONT_FIRST_STABLE_READY trigger_y=%.2f exclusion=%.2fm "
            "confirm=%d ttl=%.1fs first_stage_unchanged=true",
            self.indoor_trigger_y, self.dynamic_exclusion_radius,
            self.confirm_hits, self.confirmed_ttl_s)

    def _inside_room(self):
        if self.pose is None:
            return False
        if self.indoor_trigger_less_than:
            return self.pose[1] <= self.indoor_trigger_y
        return self.pose[1] >= self.indoor_trigger_y

    def _update_room_activation(self):
        if self.indoor_active or not self._inside_room():
            return
        self.indoor_active = True
        self.max_lateral_cmd = self.room_max_lateral
        self.requested_vy = 0.0
        self.last_vy = 0.0
        self.last_progress_time = rospy.Time.now()
        rospy.logwarn(
            "ROOM_STABLE_DYNAMIC_ACTIVATED y=%.3f radius=%.2fm lateral=%.3f",
            self.pose[1], self.dynamic_exclusion_radius,
            self.room_max_lateral)

    def reset_for_goal(self, goal):
        super(FactoryRoomFrontFirstStableActionBridge, self).reset_for_goal(goal)
        self.requested_vy = 0.0
        self.last_vy = 0.0
        self.max_lateral_cmd = (self.room_max_lateral if self.indoor_active
                                else self.first_stage_max_lateral)

    def _scan_origin_in_map(self, msg):
        frame_id = msg.header.frame_id or self.base_frame
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.map_frame, frame_id, rospy.Time(0))
            yaw = self.tf.transformations.euler_from_quaternion(rot)[2]
            return trans[0], trans[1], yaw
        except Exception:
            return self.pose

    def _static_clearance(self, wx, wy):
        cell = self.grid.world_to_map(wx, wy)
        if cell is None or self.static_distance_cells is None:
            return 0.0
        value = self.static_distance_cells[self.grid.index(cell[0], cell[1])]
        if value >= 1.0e17:
            return 999.0
        return value * self.grid.resolution

    def _point_near_remaining_path(self, point, gate):
        if self.pose is None or not self.path_world:
            return False
        points = [(self.pose[0], self.pose[1])]
        points.extend(self.path_world[max(0, self.path_index):])
        return any(point_segment_distance(
            point, points[index], points[index + 1]) <= gate
            for index in range(len(points) - 1))

    def _candidate_world_points(self, msg, semantic):
        origin = self._scan_origin_in_map(msg)
        if origin is None:
            return []
        ox, oy, oyaw = origin
        candidates = []
        for observation in semantic["cone_observations"]:
            if observation["count"] < self.minimum_cluster_points:
                continue
            if observation["span"] > self.maximum_cluster_span:
                continue
            if observation["range"] > self.observation_range:
                continue
            if abs(observation["angle_deg"]) > self.observation_half_angle:
                continue
            bearing = math.radians(observation["angle_deg"])
            distance = observation["range"] + self.cone_base_pad
            wx = ox + math.cos(oyaw + bearing) * distance
            wy = oy + math.sin(oyaw + bearing) * distance
            if self._static_clearance(wx, wy) < self.wall_rejection_clearance:
                continue
            if not self._point_near_remaining_path(
                    (wx, wy), self.dynamic_candidate_path_gate):
                continue
            candidates.append((wx, wy))
        return candidates

    def cb_scan(self, msg):
        super(FactoryRoomFrontFirstStableActionBridge, self).cb_scan(msg)
        if not self.dynamic_ready or not self.bridge_active:
            return
        self._update_room_activation()
        if not self.indoor_active or self.grid is None:
            return
        semantic = classify_scan(
            msg, cone_base_pad_m=0.0,
            max_cone_range_m=self.observation_range,
            front_angle_deg=self.front_angle_deg,
            side_min_deg=self.side_angle_min_deg,
            side_max_deg=self.side_angle_max_deg,
            wall_front_relax_m=0.0, wall_side_relax_m=0.0)
        self._update_tracks(
            self._candidate_world_points(msg, semantic), rospy.Time.now())

    def _update_tracks(self, candidates, stamp):
        matched = set()
        for wx, wy in candidates:
            best = None
            best_distance = self.association_radius
            for track in self.obstacle_tracks:
                if track.track_id in matched:
                    continue
                distance = math.hypot(wx - track.x, wy - track.y)
                if distance < best_distance:
                    best = track
                    best_distance = distance
            if best is None:
                best = StableObstacleTrack(
                    self.next_track_id, wx, wy, stamp)
                self.next_track_id += 1
                self.obstacle_tracks.append(best)
            else:
                best.update(wx, wy, stamp, self.track_smoothing_alpha)
            matched.add(best.track_id)
            if (not best.confirmed and best.hits >= self.confirm_hits and
                    (stamp - best.first_seen).to_sec() >= self.confirm_age_s):
                best.confirmed = True
                self.dynamic_layer_dirty = True
                rospy.logwarn(
                    "ROOM_TEMP_OBSTACLE_CONFIRMED id=%d map=(%.2f,%.2f)",
                    best.track_id, best.x, best.y)

        retained = []
        for track in self.obstacle_tracks:
            ttl = self.confirmed_ttl_s if track.confirmed else self.pending_ttl_s
            if (stamp - track.last_seen).to_sec() <= ttl:
                retained.append(track)
            elif track.confirmed:
                self.dynamic_layer_dirty = True
        self.obstacle_tracks = retained
        if any(track.confirmed and track.track_id in matched
               for track in self.obstacle_tracks):
            self.dynamic_layer_dirty = True

    def _confirmed_tracks(self):
        return [track for track in self.obstacle_tracks if track.confirmed]

    def _layer_signature(self, tracks):
        resolution = max(self.signature_resolution, 0.01)
        return tuple(sorted((int(round(track.x / resolution)),
                             int(round(track.y / resolution)))
                            for track in tracks))

    def _remaining_path_blocked(self, tracks):
        if not tracks or not self.path_world or self.pose is None:
            return False
        points = [(self.pose[0], self.pose[1])]
        points.extend(self.path_world[max(0, self.path_index):])
        for track in tracks:
            for index in range(len(points) - 1):
                if point_segment_distance(
                        (track.x, track.y), points[index],
                        points[index + 1]) <= self.path_trigger_radius:
                    return True
        return False

    def _publish_markers(self, tracks):
        message = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        message.markers.append(clear)
        for index, track in enumerate(tracks):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "room_stable_obstacles"
            marker.id = index
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            marker.pose.position.x = track.x
            marker.pose.position.y = track.y
            marker.pose.position.z = 0.10
            marker.pose.orientation.w = 1.0
            marker.scale.x = 2.0 * self.dynamic_exclusion_radius
            marker.scale.y = 2.0 * self.dynamic_exclusion_radius
            marker.scale.z = 0.20
            marker.color.r = 1.0
            marker.color.g = 0.45
            marker.color.b = 0.05
            marker.color.a = 0.48
            marker.lifetime = rospy.Duration(self.confirmed_ttl_s + 0.5)
            message.markers.append(marker)
        self.marker_pub.publish(message)

    def _apply_dynamic_layer(self):
        tracks = self._confirmed_tracks()
        signature = self._layer_signature(tracks)
        if signature == self.dynamic_layer_signature:
            self.dynamic_layer_dirty = False
            return False
        path_blocked = self._remaining_path_blocked(tracks)
        self.grid.clear_dynamic_blocks()
        cells = 0
        for track in tracks:
            cells += self.grid.add_dynamic_block(
                track.x, track.y, self.dynamic_exclusion_radius)
        self.grid.dist_cells = self.static_distance_cells
        self.planner.roadmaps.clear()
        self.dynamic_layer_signature = signature
        self.dynamic_layer_dirty = False
        self.last_layer_update = rospy.Time.now()
        self._publish_markers(tracks)
        rospy.logwarn(
            "ROOM_STABLE_DYNAMIC_LAYER obstacles=%d cells=%d blocked=%s "
            "radius=%.2f",
            len(tracks), cells, str(path_blocked).lower(),
            self.dynamic_exclusion_radius)
        return path_blocked

    def _dynamic_maintenance(self):
        if not self.indoor_active:
            return False
        now = rospy.Time.now()
        if (self.dynamic_layer_dirty and
                (now - self.last_layer_update).to_sec() >=
                self.layer_update_min_interval):
            blocked = self._apply_dynamic_layer()
            if (blocked and
                    (now - self.last_dynamic_replan).to_sec() >=
                    self.replan_min_dynamic_interval):
                self.last_dynamic_replan = now
                self.path_world = []
                self.path_index = 0
                self.last_progress_time = now
                rospy.logwarn("ROOM_STABLE_DYNAMIC_REPLAN path_intersection")
                self.plan_from_current_pose(
                    "stable cone intersects path", force=True)
                return True

        stuck_for = (now - self.last_progress_time).to_sec()
        if (self._confirmed_tracks() and
                stuck_for >= self.dynamic_stuck_replan_s and
                (now - self.last_dynamic_replan).to_sec() >=
                self.replan_min_dynamic_interval and
                self.distance_to_active_goal() >
                self.current_target_tolerance() + 0.12):
            self.last_dynamic_replan = now
            self.path_world = []
            self.path_index = 0
            self.last_progress_time = now
            rospy.logwarn(
                "ROOM_STABLE_DYNAMIC_REPLAN no_progress=%.2fs", stuck_for)
            self.plan_from_current_pose(
                "no progress near stable cones", force=True)
            return True
        return False

    def control_loop(self, event):
        if not self.dynamic_ready:
            return
        if not self.bridge_active:
            return
        self._update_room_activation()
        if self._dynamic_maintenance():
            return
        super(FactoryRoomFrontFirstStableActionBridge, self).control_loop(event)

    def publish_cmd(self, vx, wz):
        if self.dynamic_ready and self.indoor_active:
            self.max_lateral_cmd = self.room_max_lateral
            self.requested_vy = 0.0
            self.last_vy = 0.0
        else:
            self.max_lateral_cmd = self.first_stage_max_lateral
        super(FactoryRoomFrontFirstStableActionBridge, self).publish_cmd(vx, wz)

    def clear_dynamic_map(self, request):
        response = super(
            FactoryRoomFrontFirstStableActionBridge,
            self).clear_dynamic_map(request)
        self.obstacle_tracks = []
        self.dynamic_layer_signature = ()
        self.dynamic_layer_dirty = False
        self.next_track_id = 1
        if self.grid is not None and self.static_distance_cells is not None:
            self.grid.dist_cells = self.static_distance_cells
        self._publish_markers([])
        rospy.logwarn("ROOM_STABLE_TRACKS_CLEARED")
        return response


if __name__ == "__main__":
    FactoryRoomFrontFirstStableActionBridge().spin()
