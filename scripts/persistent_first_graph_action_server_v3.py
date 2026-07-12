#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Relaxed room observation navigation with 20 cm cone exclusion."""

import math
import rospy

from global_first_graph_nav_2249fcf import RosGlobalFirstGraphNavigator
from persistent_first_graph_action_server_v2 import RecoveringConeActionServer


class CompactConeExclusionActionServer(RecoveringConeActionServer):
    def __init__(self):
        # Defaults are needed before the inherited constructor starts its timer.
        self.entry_transit_active = False
        self.entry_target_x = -0.85
        self.entry_target_y = -1.90
        self.entry_target_match_radius = 0.32
        self.entry_accept_y = -1.78
        self.cone_mapping_activation_y = -1.95
        self.room_observation_goal_active = False
        self.observation_accept_radius = 0.26
        self.indoor_wall_front_stop = 0.24
        self.indoor_wall_front_slow = 0.40
        self.indoor_wall_side_stop = 0.16
        self.indoor_wall_side_slow = 0.25
        self.indoor_wall_proximity_weight = 2.8
        super(CompactConeExclusionActionServer, self).__init__()
        self.cone_exclusion_radius = float(rospy.get_param(
            "~indoor_cone_exclusion_radius_m", 0.20))
        self.entry_target_x = float(rospy.get_param(
            "~entry_target_x", -0.85))
        self.entry_target_y = float(rospy.get_param(
            "~entry_target_y", -1.90))
        self.entry_target_match_radius = float(rospy.get_param(
            "~entry_target_match_radius_m", 0.32))
        self.entry_accept_y = float(rospy.get_param(
            "~entry_accept_y", -1.78))
        self.cone_mapping_activation_y = float(rospy.get_param(
            "~cone_mapping_activation_y", -1.95))
        self.observation_accept_radius = float(rospy.get_param(
            "~observation_accept_radius_m", 0.26))
        self.indoor_wall_front_stop = float(rospy.get_param(
            "~indoor_wall_front_stop_m", 0.24))
        self.indoor_wall_front_slow = float(rospy.get_param(
            "~indoor_wall_front_slow_m", 0.40))
        self.indoor_wall_side_stop = float(rospy.get_param(
            "~indoor_wall_side_stop_m", 0.16))
        self.indoor_wall_side_slow = float(rospy.get_param(
            "~indoor_wall_side_slow_m", 0.25))
        self.indoor_wall_proximity_weight = float(rospy.get_param(
            "~indoor_wall_proximity_weight", 2.8))
        self.indoor_dynamic_radius = self.cone_exclusion_radius
        if self.indoor_profile_active:
            self.dynamic_obstacle_radius_m = self.cone_exclusion_radius
            self.apply_relaxed_wall_profile()
        rospy.logwarn(
            "CONE_EXCLUSION_OVERRIDE physical_radius=%.3fm exclusion=%.3fm "
            "entry_accept_y=%.3f cone_mapping_y=%.3f",
            self.cone_radius, self.cone_exclusion_radius,
            self.entry_accept_y, self.cone_mapping_activation_y)

    def reset_for_goal(self, goal):
        super(CompactConeExclusionActionServer, self).reset_for_goal(goal)
        distance_to_entry = math.hypot(
            self.goal_x - self.entry_target_x,
            self.goal_y - self.entry_target_y)
        self.entry_transit_active = (
            distance_to_entry <= self.entry_target_match_radius)
        self.room_observation_goal_active = not self.entry_transit_active
        if not self.entry_transit_active:
            # d1/d2/d3 are room goals, not doorway goals. Never send them back
            # through the entrance waypoint even if the indoor pose update is
            # delayed by one callback cycle.
            self.waypoint_enabled = False
            self.goal_stage = 1
            self.active_goal = (self.goal_x, self.goal_y)
            self.path_world = []
            self.path_index = 0
            self.last_plan_time = rospy.Time(0)
            rospy.logwarn(
                "ROOM_GOAL_DIRECT no_entry_waypoint goal=(%.3f,%.3f) "
                "accept_radius=%.3f yaw_not_required=true",
                self.goal_x, self.goal_y, self.observation_accept_radius)
        else:
            rospy.logwarn(
                "ENTRY_TRANSIT_GOAL relaxed_accept_y=%.3f yaw_not_required=true",
                self.entry_accept_y)

    def check_goal(self):
        if (self.entry_transit_active and self.pose is not None and
                self.pose[1] <= self.entry_accept_y):
            self.finished = True
            self.publish_zero("ENTRY_TRANSIT_ACCEPTED")
            self.log_status(
                "entry transit accepted at (%.2f, %.2f); exact pose/yaw skipped" %
                (self.pose[0], self.pose[1]))
            rospy.logwarn(
                "ENTRY_TRANSIT_ACCEPTED pose=(%.3f,%.3f) threshold_y=%.3f",
                self.pose[0], self.pose[1], self.entry_accept_y)
            return True
        if (self.room_observation_goal_active and self.pose is not None and
                math.hypot(self.pose[0] - self.goal_x,
                           self.pose[1] - self.goal_y) <=
                self.observation_accept_radius):
            self.finished = True
            self.publish_zero("ROOM_OBSERVATION_REGION_ACCEPTED")
            rospy.logwarn(
                "ROOM_OBSERVATION_REGION_ACCEPTED pose=(%.3f,%.3f) "
                "goal=(%.3f,%.3f) radius=%.3f yaw_skipped=true",
                self.pose[0], self.pose[1], self.goal_x, self.goal_y,
                self.observation_accept_radius)
            return True
        return super(CompactConeExclusionActionServer, self).check_goal()

    def apply_scan_guard(self, cmd):
        if (self.pose is not None and
                self.pose[1] > self.cone_mapping_activation_y):
            # At the doorway, keep the proven reactive laser guard but do not
            # permanently record the door frame as a circular cone.
            previous = self.dynamic_obstacles_enabled
            self.dynamic_obstacles_enabled = False
            try:
                return RosGlobalFirstGraphNavigator.apply_scan_guard(self, cmd)
            finally:
                self.dynamic_obstacles_enabled = previous
        return super(CompactConeExclusionActionServer, self).apply_scan_guard(cmd)

    def activate_indoor_profile(self):
        requested = float(rospy.get_param(
            "~indoor_cone_exclusion_radius_m", 0.20))
        self.cone_exclusion_radius = requested
        self.indoor_dynamic_radius = requested
        super(CompactConeExclusionActionServer, self).activate_indoor_profile()
        self.dynamic_obstacle_radius_m = requested
        self.apply_relaxed_wall_profile()
        rospy.logwarn("CONE_EXCLUSION_ACTIVE exact_radius=%.3fm", requested)

    def apply_relaxed_wall_profile(self):
        self.params["proximity_weight"] = self.indoor_wall_proximity_weight
        self.front_stop_m = self.indoor_wall_front_stop
        self.front_slow_m = self.indoor_wall_front_slow
        self.side_stop_m = self.indoor_wall_side_stop
        self.side_slow_m = self.indoor_wall_side_slow
        if self.planner is not None:
            self.planner.roadmaps.clear()
        rospy.logwarn(
            "ROOM_WALL_GUARD_RELAXED front=(%.2f,%.2f) side=(%.2f,%.2f) "
            "proximity_weight=%.2f",
            self.front_stop_m, self.front_slow_m,
            self.side_stop_m, self.side_slow_m,
            self.indoor_wall_proximity_weight)


if __name__ == "__main__":
    CompactConeExclusionActionServer().spin()
