#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V4 dynamic obstacle navigation with position-only coverage goals."""

import rospy

from factory_room_front_first_stable_action_v4 import (
    ConsolidatingStableRoomActionBridge,
)


class MapCoverageRoomActionBridge(ConsolidatingStableRoomActionBridge):
    def __init__(self):
        super(MapCoverageRoomActionBridge, self).__init__()
        self.normal_require_goal_yaw = False
        self.require_goal_yaw = False
        self.coverage_normal_goal_tolerance = float(self.goal_tolerance)
        self.coverage_d3_min_x = float(rospy.get_param(
            "~coverage_d3_tight_min_x", 2.45))
        self.coverage_d3_max_y = float(rospy.get_param(
            "~coverage_d3_tight_max_y", -2.38))
        self.coverage_d3_goal_tolerance = float(rospy.get_param(
            "~coverage_d3_goal_tolerance_m", 0.10))
        rospy.logwarn(
            "MAP_COVERAGE_ACTION_V5 position_only_all_views=true "
            "guarded_wall_turn_owned_by_manager=true")

    def reset_for_goal(self, goal):
        pose = goal.target_pose.pose
        d3_anchor = (
            float(pose.position.x) >= self.coverage_d3_min_x and
            float(pose.position.y) <= self.coverage_d3_max_y)
        self.goal_tolerance = (
            self.coverage_d3_goal_tolerance if d3_anchor
            else self.coverage_normal_goal_tolerance)
        self.normal_require_goal_yaw = False
        self.require_goal_yaw = False
        super(MapCoverageRoomActionBridge, self).reset_for_goal(goal)
        self.require_goal_yaw = False
        rospy.logwarn(
            "MAP_COVERAGE_GOAL_ACCEPTANCE position_only=true x=%.3f y=%.3f "
            "tolerance=%.2fm d3_anchor=%s",
            self.goal_x, self.goal_y, self.goal_tolerance,
            str(d3_anchor).lower())


if __name__ == "__main__":
    MapCoverageRoomActionBridge().spin()
