#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Indoor graph navigation with proactive cone mapping and room-only clearance."""

import math

import rospy

from factory_room_global_first_action_bridge_indoor_v1 import (
    IndoorInflationActionBridge,
)


class ProactiveConeActionBridge(IndoorInflationActionBridge):
    def __init__(self):
        self.proactive_centers = []
        self.last_proactive_scan = rospy.Time(0)
        super(ProactiveConeActionBridge, self).__init__()
        self.proactive_enabled = bool(rospy.get_param(
            "~indoor_proactive_obstacles", True))
        self.proactive_period = float(rospy.get_param(
            "~indoor_proactive_period_s", 0.30))
        self.proactive_angle = math.radians(float(rospy.get_param(
            "~indoor_proactive_angle_deg", 92.0)))
        self.proactive_min_range = float(rospy.get_param(
            "~indoor_proactive_min_range_m", 0.28))
        self.proactive_max_range = float(rospy.get_param(
            "~indoor_proactive_max_range_m", 0.95))
        self.proactive_cluster_gap = float(rospy.get_param(
            "~indoor_proactive_cluster_gap_m", 0.10))
        self.proactive_cluster_max_span = float(rospy.get_param(
            "~indoor_proactive_cluster_max_span_m", 0.34))
        self.proactive_static_clearance = float(rospy.get_param(
            "~indoor_proactive_static_clearance_m", 0.12))
        self.proactive_merge_distance = float(rospy.get_param(
            "~indoor_proactive_merge_distance_m", 0.18))
        self.proactive_radius = float(rospy.get_param(
            "~indoor_proactive_radius_m", 0.30))
        self.proactive_max_clusters = int(rospy.get_param(
            "~indoor_proactive_max_clusters", 5))
        self.indoor_max_linear = float(rospy.get_param(
            "~indoor_max_linear_vel", 0.28))
        self.indoor_lookahead = float(rospy.get_param(
            "~indoor_lookahead_dist_m", 0.34))
        rospy.logwarn(
            "ROOM_PROACTIVE_CONE_MAPPING_ARMED range=%.2f..%.2fm radius=%.2fm",
            self.proactive_min_range, self.proactive_max_range,
            self.proactive_radius)

    def activate_indoor_profile(self):
        super(ProactiveConeActionBridge, self).activate_indoor_profile()
        # React quickly if an object enters the guard before clustering finishes.
        self.front_replan_after_s = float(rospy.get_param(
            "~indoor_front_replan_after_s", 0.25))
        self.blocked_replan_s = float(rospy.get_param(
            "~indoor_blocked_replan_s", 0.45))
        self.replan_min_interval_s = float(rospy.get_param(
            "~indoor_replan_min_interval_s", 0.45))
        self.dynamic_obstacle_trigger_m = max(
            self.dynamic_obstacle_trigger_m, self.indoor_front_slow)
        self.dynamic_obstacle_record_range_m = max(
            self.dynamic_obstacle_record_range_m, self.proactive_max_range)
        self.max_linear = min(self.max_linear, self.indoor_max_linear)
        self.lookahead_dist = min(self.lookahead_dist, self.indoor_lookahead)

    def is_new_center(self, center):
        return all(math.hypot(center[0] - old[0], center[1] - old[1]) >=
                   self.proactive_merge_distance
                   for old in self.proactive_centers)

    def point_is_separate_from_static_wall(self, wx, wy):
        cell = self.grid.world_to_map(wx, wy)
        if cell is None:
            return False
        return self.grid.clearance_m(cell[0], cell[1]) > \
            self.proactive_static_clearance

    def scan_clusters(self):
        points = []
        for index, distance in enumerate(self.scan.ranges):
            if not math.isfinite(distance):
                continue
            if (distance < max(self.scan.range_min, self.proactive_min_range) or
                    distance > min(self.scan.range_max,
                                   self.proactive_max_range)):
                continue
            angle = self.scan.angle_min + index * self.scan.angle_increment
            if abs(angle) > self.proactive_angle:
                continue
            points.append((index, distance * math.cos(angle),
                           distance * math.sin(angle), angle))

        clusters = []
        current = []
        for point in points:
            if current:
                previous = current[-1]
                gap = math.hypot(point[1] - previous[1],
                                 point[2] - previous[2])
                if point[0] != previous[0] + 1 or gap > self.proactive_cluster_gap:
                    clusters.append(current)
                    current = []
            current.append(point)
        if current:
            clusters.append(current)
        return clusters

    def mark_visible_cones(self):
        if (not self.proactive_enabled or not self.indoor_profile_active or
                not self.bridge_active or self.pose is None or
                self.scan is None or self.grid is None or self.planner is None or
                not self.scan_fresh()):
            return 0
        now = rospy.Time.now()
        if (now - self.last_proactive_scan).to_sec() < self.proactive_period:
            return 0
        self.last_proactive_scan = now

        x, y, yaw = self.pose
        candidates = []
        for cluster in self.scan_clusters():
            if len(cluster) < 2:
                continue
            span = math.hypot(cluster[-1][1] - cluster[0][1],
                              cluster[-1][2] - cluster[0][2])
            if span > self.proactive_cluster_max_span:
                continue
            nearest = min(cluster, key=lambda value: value[1] * value[1] +
                          value[2] * value[2])
            distance = math.hypot(nearest[1], nearest[2])
            angle = nearest[3]
            wx = x + math.cos(yaw + angle) * distance
            wy = y + math.sin(yaw + angle) * distance
            if not self.point_is_separate_from_static_wall(wx, wy):
                continue
            center = (wx, wy)
            if self.is_new_center(center):
                candidates.append((distance, center, len(cluster), span))

        candidates.sort(key=lambda value: value[0])
        added_cells = 0
        accepted = []
        for _, center, count, span in candidates[:self.proactive_max_clusters]:
            if not self.is_new_center(center):
                continue
            self.proactive_centers.append(center)
            accepted.append((center, count, span))
            added_cells += self.grid.add_dynamic_block(
                center[0], center[1], self.proactive_radius)

        if added_cells <= 0:
            return 0
        self.planner.roadmaps.clear()
        self.path_world = []
        self.path_index = 0
        self.last_plan_time = rospy.Time(0)
        rospy.logwarn(
            "ROOM_CONES_MAPPED new=%d cells=%d total_cones=%d; replanning",
            len(accepted), added_cells, len(self.proactive_centers))
        return len(accepted)

    def clear_dynamic_map(self, request):
        self.proactive_centers = []
        return super(ProactiveConeActionBridge, self).clear_dynamic_map(request)

    def control_loop(self, event):
        if (self.bridge_active and not self.indoor_profile_active and
                self.crossed_indoor_trigger()):
            self.activate_indoor_profile()
        self.mark_visible_cones()
        # Call the bridge parent directly because V1 would repeat activation.
        super(IndoorInflationActionBridge, self).control_loop(event)


if __name__ == "__main__":
    ProactiveConeActionBridge().spin()
