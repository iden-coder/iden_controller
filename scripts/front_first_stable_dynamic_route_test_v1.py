#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Front-first route test with a persistent, lightweight obstacle layer.

Laser clusters are tracked in the map frame.  Only observations confirmed in
several scans and separated from static walls become temporary obstacles.
The global graph is replanned only when a confirmed obstacle intersects the
remaining path or when progress stalls.  Normal motion is vx/wz only.
"""

import math

import rospy
from visualization_msgs.msg import Marker, MarkerArray

from front_first_smooth_navigation_v1 import FrontFirstSmoothNavigator
from global_first_graph_nav_2249fcf import norm_angle
from room_lidar_semantics_v1 import classify_scan


DEFAULT_ROUTE = [
    {"name": "first_stage", "x": -1.48, "y": -0.45,
     "yaw": math.pi, "tolerance": 0.10, "require_yaw": False},
    {"name": "doorway_approach", "x": -0.85, "y": -1.41,
     "yaw": -math.pi / 2.0, "tolerance": 0.16, "require_yaw": False},
    {"name": "room_entry", "x": -0.85, "y": -1.90,
     "yaw": -math.pi / 2.0, "tolerance": 0.20, "require_yaw": False},
    {"name": "d1", "x": -1.43, "y": -2.57,
     "yaw": math.pi, "tolerance": 0.15, "require_yaw": False},
    {"name": "d2", "x": 0.41, "y": -2.00,
     "yaw": math.pi / 2.0, "tolerance": 0.15, "require_yaw": False},
    {"name": "d3", "x": 2.54, "y": -2.51,
     "yaw": -math.pi / 2.0, "tolerance": 0.15, "require_yaw": False},
]


def point_segment_distance(point, start, end):
    px, py = point
    ax, ay = start
    bx, by = end
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-10:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


class StableObstacleTrack(object):
    def __init__(self, track_id, x, y, stamp):
        self.track_id = track_id
        self.x = x
        self.y = y
        self.first_seen = stamp
        self.last_seen = stamp
        self.hits = 1
        self.confirmed = False

    def update(self, x, y, stamp, alpha):
        self.x = (1.0 - alpha) * self.x + alpha * x
        self.y = (1.0 - alpha) * self.y + alpha * y
        self.last_seen = stamp
        self.hits += 1


class FrontFirstStableDynamicRouteTest(FrontFirstSmoothNavigator):
    def __init__(self):
        self.route_ready = False
        self.route_points = [dict(point) for point in DEFAULT_ROUTE]
        self.route_index = 0
        self.route_enter_time = None

        self.obstacle_tracks = []
        self.next_track_id = 1
        self.dynamic_layer_dirty = False
        self.dynamic_layer_signature = ()
        self.last_layer_update = rospy.Time(0)
        self.last_dynamic_replan = rospy.Time(0)
        self.static_distance_cells = None
        super(FrontFirstStableDynamicRouteTest, self).__init__()

        self.route_points = self._validate_route(
            rospy.get_param("~route_points", DEFAULT_ROUTE))
        self.route_final_hold_s = float(rospy.get_param(
            "~route_final_hold_s", 0.60))
        self.route_yaw_tolerance = math.radians(float(rospy.get_param(
            "~route_yaw_tolerance_deg", 10.0)))
        self.route_yaw_speed = float(rospy.get_param(
            "~route_yaw_speed_rps", 0.26))

        self.cone_base_pad = float(rospy.get_param(
            "~dynamic_cone_base_pad_m", 0.04))
        self.observation_range = float(rospy.get_param(
            "~dynamic_observation_range_m", 1.35))
        self.observation_half_angle = float(rospy.get_param(
            "~dynamic_observation_half_angle_deg", 85.0))
        self.minimum_cluster_points = int(rospy.get_param(
            "~dynamic_min_cluster_points", 3))
        self.wall_rejection_clearance = float(rospy.get_param(
            "~dynamic_wall_rejection_clearance_m", 0.11))
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

        # This is the complete centre exclusion radius.  The static distance
        # field is deliberately not recomputed, so planner footprint clearance
        # is not accidentally added to this radius a second time.
        self.dynamic_exclusion_radius = float(rospy.get_param(
            "~dynamic_exclusion_radius_m", 0.25))
        self.path_trigger_radius = float(rospy.get_param(
            "~dynamic_path_trigger_radius_m", 0.25))
        self.layer_update_min_interval = float(rospy.get_param(
            "~dynamic_layer_update_min_interval_s", 0.40))
        self.replan_min_dynamic_interval = float(rospy.get_param(
            "~dynamic_replan_min_interval_s", 0.85))
        self.dynamic_stuck_replan_s = float(rospy.get_param(
            "~dynamic_stuck_replan_s", 2.6))
        self.signature_resolution = float(rospy.get_param(
            "~dynamic_signature_resolution_m", 0.05))
        self.dynamic_activation_point = str(rospy.get_param(
            "~dynamic_activation_route_point", "d1"))
        self.dynamic_candidate_path_gate = float(rospy.get_param(
            "~dynamic_candidate_path_gate_m", 0.34))
        self.maximum_cluster_span = float(rospy.get_param(
            "~dynamic_max_cluster_span_m", 0.22))
        self.first_stage_max_lateral = float(rospy.get_param(
            "~first_stage_max_lateral_mps", 0.055))
        self.dynamic_stage_max_lateral = float(rospy.get_param(
            "~dynamic_stage_max_lateral_mps", 0.0))
        activation_matches = [
            index for index, point in enumerate(self.route_points)
            if point["name"] == self.dynamic_activation_point]
        if not activation_matches:
            raise ValueError(
                "dynamic_activation_route_point %s is not in route_points" %
                self.dynamic_activation_point)
        self.dynamic_activation_index = activation_matches[0]

        self.marker_pub = rospy.Publisher(
            "~dynamic_obstacles", MarkerArray, queue_size=1, latch=True)
        self.static_distance_cells = list(self.grid.dist_cells)

        # Restore the proven light lateral wall correction before entering the
        # cone room.  Once the dynamic layer is active, cones are avoided by
        # global replanning and ordinary navigation remains vx/wz only.
        self.max_lateral_cmd = self.first_stage_max_lateral
        self.requested_vy = 0.0
        self.last_vy = 0.0
        self.dynamic_obstacles_enabled = False

        self.route_index = 0
        self.finished = False
        self.goal_stage = 1
        self.waypoint_enabled = False
        self._activate_route_point(clear_path=True)
        self.route_ready = True
        rospy.logwarn(
            "STABLE_DYNAMIC_ROUTE_READY points=%s exclusion=%.2fm "
            "confirm=%d scans ttl=%.1fs activation=%s lateral=(%.3f->%.3f)",
            " -> ".join(point["name"] for point in self.route_points),
            self.dynamic_exclusion_radius, self.confirm_hits,
            self.confirmed_ttl_s, self.dynamic_activation_point,
            self.first_stage_max_lateral, self.dynamic_stage_max_lateral)

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

    def _scan_origin_in_map(self, msg):
        frame_id = msg.header.frame_id or self.base_frame
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.map_frame, frame_id, rospy.Time(0))
            yaw = self.tf.transformations.euler_from_quaternion(rot)[2]
            return trans[0], trans[1], yaw
        except Exception:
            if self.pose is None:
                return None
            return self.pose

    def _static_clearance(self, wx, wy):
        cell = self.grid.world_to_map(wx, wy)
        if cell is None or self.static_distance_cells is None:
            return 0.0
        index = self.grid.index(cell[0], cell[1])
        value = self.static_distance_cells[index]
        if value >= 1.0e17:
            return 999.0
        return value * self.grid.resolution

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

    def _dynamic_layer_active(self):
        if not getattr(self, "route_ready", False):
            return False
        return self.route_index >= self.dynamic_activation_index

    def _point_near_remaining_path(self, point, gate):
        if self.pose is None or not self.path_world:
            return False
        points = [(self.pose[0], self.pose[1])]
        points.extend(self.path_world[max(0, self.path_index):])
        for index in range(len(points) - 1):
            if point_segment_distance(
                    point, points[index], points[index + 1]) <= gate:
                return True
        return False

    def cb_scan(self, msg):
        super(FrontFirstStableDynamicRouteTest, self).cb_scan(msg)
        if (not self.route_ready or self.grid is None or
                not self._dynamic_layer_active()):
            return
        semantic = classify_scan(
            msg, cone_base_pad_m=0.0,
            max_cone_range_m=self.observation_range,
            front_angle_deg=self.front_angle_deg,
            side_min_deg=self.side_angle_min_deg,
            side_max_deg=self.side_angle_max_deg,
            wall_front_relax_m=0.0, wall_side_relax_m=0.0)
        candidates = self._candidate_world_points(msg, semantic)
        self._update_tracks(candidates, rospy.Time.now())

    def _update_tracks(self, candidates, stamp):
        matched_tracks = set()
        for wx, wy in candidates:
            best = None
            best_dist = self.association_radius
            for track in self.obstacle_tracks:
                if track.track_id in matched_tracks:
                    continue
                distance = math.hypot(wx - track.x, wy - track.y)
                if distance < best_dist:
                    best = track
                    best_dist = distance
            if best is None:
                best = StableObstacleTrack(
                    self.next_track_id, wx, wy, stamp)
                self.next_track_id += 1
                self.obstacle_tracks.append(best)
            else:
                best.update(wx, wy, stamp, self.track_smoothing_alpha)
            matched_tracks.add(best.track_id)
            if (not best.confirmed and best.hits >= self.confirm_hits and
                    (stamp - best.first_seen).to_sec() >= self.confirm_age_s):
                best.confirmed = True
                self.dynamic_layer_dirty = True
                rospy.logwarn(
                    "TEMP_OBSTACLE_CONFIRMED id=%d map=(%.2f,%.2f) hits=%d",
                    best.track_id, best.x, best.y, best.hits)

        retained = []
        for track in self.obstacle_tracks:
            ttl = self.confirmed_ttl_s if track.confirmed else self.pending_ttl_s
            if (stamp - track.last_seen).to_sec() <= ttl:
                retained.append(track)
            elif track.confirmed:
                self.dynamic_layer_dirty = True
                rospy.logwarn(
                    "TEMP_OBSTACLE_EXPIRED id=%d map=(%.2f,%.2f)",
                    track.track_id, track.x, track.y)
        self.obstacle_tracks = retained

        if any(track.confirmed and track.track_id in matched_tracks
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
        if len(points) < 2:
            return False
        for track in tracks:
            for index in range(len(points) - 1):
                if point_segment_distance(
                        (track.x, track.y), points[index], points[index + 1]) <= \
                        self.path_trigger_radius:
                    return True
        return False

    def _publish_dynamic_markers(self, tracks):
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        for index, track in enumerate(tracks):
            marker = Marker()
            marker.header.frame_id = self.map_frame
            marker.header.stamp = rospy.Time.now()
            marker.ns = "stable_dynamic_obstacles"
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
            markers.markers.append(marker)
        self.marker_pub.publish(markers)

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

        # Keep the original static distance field.  Dynamic disks already
        # represent the complete centre exclusion radius.
        self.grid.dist_cells = self.static_distance_cells
        self.planner.roadmaps.clear()
        self.dynamic_layer_signature = signature
        self.dynamic_layer_dirty = False
        self.last_layer_update = rospy.Time.now()
        self._publish_dynamic_markers(tracks)
        rospy.logwarn(
            "STABLE_DYNAMIC_LAYER obstacles=%d cells=%d path_blocked=%s "
            "radius=%.2f no_double_inflation=true",
            len(tracks), cells, str(path_blocked).lower(),
            self.dynamic_exclusion_radius)
        return path_blocked

    def _dynamic_maintenance(self):
        if not self._dynamic_layer_active():
            return False
        now = rospy.Time.now()
        if (self.dynamic_layer_dirty and
                (now - self.last_layer_update).to_sec() >=
                self.layer_update_min_interval):
            path_blocked = self._apply_dynamic_layer()
            if (path_blocked and
                    (now - self.last_dynamic_replan).to_sec() >=
                    self.replan_min_dynamic_interval):
                self.last_dynamic_replan = now
                self.path_world = []
                self.path_index = 0
                rospy.logwarn("STABLE_DYNAMIC_REPLAN reason=path_intersection")
                planned = self.plan_from_current_pose(
                    "stable obstacle intersects path", force=True)
                self.last_progress_time = rospy.Time.now()
                return planned

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
                "STABLE_DYNAMIC_REPLAN reason=no_progress stuck=%.2fs",
                stuck_for)
            planned = self.plan_from_current_pose(
                "no progress with stable obstacles", force=True)
            self.last_progress_time = rospy.Time.now()
            return planned
        return False

    def control_loop(self, event):
        if not self.route_ready:
            self.publish_zero("STABLE_DYNAMIC_INITIALIZING")
            return
        if self._dynamic_maintenance():
            return
        super(FrontFirstStableDynamicRouteTest, self).control_loop(event)

    def publish_cmd(self, vx, wz):
        # The parent constructor starts the control timer before this class has
        # finished loading route-stage parameters.  Initial zero commands must
        # be publishable during that short window.
        if not getattr(self, "route_ready", False):
            super(FrontFirstStableDynamicRouteTest, self).publish_cmd(vx, wz)
            return
        if self._dynamic_layer_active():
            self.max_lateral_cmd = self.dynamic_stage_max_lateral
            self.requested_vy = 0.0
            self.last_vy = 0.0
        else:
            self.max_lateral_cmd = self.first_stage_max_lateral
        super(FrontFirstStableDynamicRouteTest, self).publish_cmd(vx, wz)

    def check_goal(self):
        point = self.current_route_point()
        if self.distance_to_active_goal() > point["tolerance"]:
            self.route_enter_time = None
            return False

        if point["require_yaw"]:
            yaw_error = norm_angle(point["yaw"] - self.pose[2])
            if abs(yaw_error) > self.route_yaw_tolerance:
                self.publish_cmd(
                    0.0,
                    max(-self.route_yaw_speed,
                        min(self.route_yaw_speed, yaw_error)))
                return True

        final_point = self.route_index == len(self.route_points) - 1
        if final_point:
            now = rospy.Time.now()
            if self.route_enter_time is None:
                self.route_enter_time = now
                self.publish_zero("FINAL_ROUTE_POINT_HOLD")
                rospy.logwarn(
                    "STABLE_ROUTE_POINT_REACHED index=%d/%d name=%s",
                    self.route_index + 1, len(self.route_points), point["name"])
                return True
            if (now - self.route_enter_time).to_sec() < self.route_final_hold_s:
                self.publish_zero("FINAL_ROUTE_POINT_HOLD")
                return True
            self.finished = True
            self.publish_zero("STABLE_DYNAMIC_ROUTE_COMPLETE")
            self.log_status("stable dynamic route complete; node remains stopped")
            rospy.logwarn("STABLE_DYNAMIC_ROUTE_COMPLETE all_points=%d",
                          len(self.route_points))
            return True

        previous = point["name"]
        self.route_index += 1
        self._activate_route_point(clear_path=True)
        following = self.current_route_point()["name"]
        rospy.logwarn(
            "STABLE_ROUTE_POINT_PASS %s -> %s no_hold=true",
            previous, following)
        if self.route_index == self.dynamic_activation_index:
            self.max_lateral_cmd = self.dynamic_stage_max_lateral
            self.requested_vy = 0.0
            self.last_vy = 0.0
            rospy.logwarn(
                "STABLE_DYNAMIC_ACTIVATED after=%s current=%s",
                previous, following)
        self.plan_from_current_pose("continuous route advance", force=True)
        self.last_progress_time = rospy.Time.now()
        return True


if __name__ == "__main__":
    FrontFirstStableDynamicRouteTest().spin()
