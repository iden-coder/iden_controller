#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Persistent first-graph navigation with cone geometry and active recovery."""

import math

import rospy

from persistent_first_graph_action_server_v1 import (
    PersistentFirstGraphActionServer,
)


class RecoveringConeActionServer(PersistentFirstGraphActionServer):
    def __init__(self):
        # Defaults must exist before the parent starts its ROS timer.
        self.cone_radius = 0.15
        self.cone_robot_clearance = 0.239
        self.cone_exclusion_radius = 0.389
        self.recovery_enabled = True
        self.recovery_anchor = None
        self.recovery_anchor_time = rospy.Time(0)
        self.recovery_until = rospy.Time(0)
        self.recovery_turn = 0.0
        self.recovery_replan_pending = False
        self.recovery_count = 0
        super(RecoveringConeActionServer, self).__init__()

        self.cone_radius = float(rospy.get_param(
            "~indoor_cone_radius_m", 0.15))
        robot_half_diag = float(rospy.get_param(
            "~robot_half_diag", 0.214))
        footprint_margin = float(rospy.get_param(
            "~indoor_cone_footprint_margin_m", 0.025))
        self.cone_robot_clearance = robot_half_diag + footprint_margin
        self.cone_exclusion_radius = (
            self.cone_radius + self.cone_robot_clearance)
        self.indoor_dynamic_radius = self.cone_exclusion_radius

        self.recovery_enabled = bool(rospy.get_param(
            "~navigation_recovery_enabled", True))
        self.recovery_stuck_time = float(rospy.get_param(
            "~navigation_recovery_stuck_s", 9.0))
        self.recovery_progress = float(rospy.get_param(
            "~navigation_recovery_progress_m", 0.045))
        self.recovery_goal_guard = float(rospy.get_param(
            "~navigation_recovery_goal_guard_m", 0.12))
        self.recovery_turn_speed = float(rospy.get_param(
            "~navigation_recovery_turn_speed", 0.24))
        self.recovery_turn_time = float(rospy.get_param(
            "~navigation_recovery_turn_time_s", 0.90))

        if self.indoor_profile_active:
            self.dynamic_obstacle_radius_m = self.cone_exclusion_radius
            self.dynamic_obstacle_range_pad_m = self.cone_radius
        rospy.logwarn(
            "CONE_MODEL_READY physical_radius=%.3fm robot_clearance=%.3fm "
            "center_exclusion=%.3fm recovery_after=%.1fs",
            self.cone_radius, self.cone_robot_clearance,
            self.cone_exclusion_radius, self.recovery_stuck_time)

    def activate_indoor_profile(self):
        self.indoor_dynamic_radius = self.cone_exclusion_radius
        super(RecoveringConeActionServer, self).activate_indoor_profile()
        # Laser measures the cone surface. Shift by the physical radius to
        # estimate its center, then keep the robot center outside the combined
        # cone-plus-footprint exclusion circle.
        self.dynamic_obstacle_range_pad_m = self.cone_radius
        self.dynamic_obstacle_radius_m = self.cone_exclusion_radius
        rospy.logwarn(
            "CONE_MODEL_ACTIVE cone=%.3fm exclusion=%.3fm",
            self.cone_radius, self.cone_exclusion_radius)

    def reset_recovery_watchdog(self):
        if self.pose is None:
            self.recovery_anchor = None
        else:
            self.recovery_anchor = (self.pose[0], self.pose[1])
        self.recovery_anchor_time = rospy.Time.now()

    def reset_for_goal(self, goal):
        super(RecoveringConeActionServer, self).reset_for_goal(goal)
        self.recovery_until = rospy.Time(0)
        self.recovery_replan_pending = False
        self.recovery_count = 0
        self.reset_recovery_watchdog()

    def near_special_fixed_point(self):
        if self.pose is None or not self.action_active or self.finished:
            return True
        guard = self.current_target_tolerance() + self.recovery_goal_guard
        return self.distance_to_active_goal() <= guard

    def start_recovery(self):
        remembered = self.remember_front_dynamic_obstacles(
            "long-stationary recovery")
        turn_sign = 1.0 if self.left >= self.right else -1.0
        if self.recovery_count % 2 == 1:
            # Alternate on repeated attempts so a misleading single scan does
            # not make the robot choose the same trapped heading forever.
            turn_sign *= -1.0
        self.recovery_turn = turn_sign * self.recovery_turn_speed
        self.recovery_until = (
            rospy.Time.now() + rospy.Duration(self.recovery_turn_time))
        self.recovery_replan_pending = True
        self.recovery_count += 1
        self.path_world = []
        self.path_index = 0
        if self.planner is not None:
            self.planner.roadmaps.clear()
        self.publish_zero("ACTIVE_STUCK_RECOVERY_START")
        rospy.logwarn(
            "ACTIVE_STUCK_RECOVERY_START count=%d pose=(%.3f,%.3f) "
            "front=%.3f left=%.3f right=%.3f dynamic_cells=%d turn=%.3f",
            self.recovery_count, self.pose[0], self.pose[1], self.front,
            self.left, self.right, remembered, self.recovery_turn)

    def update_recovery_watchdog(self):
        if (not self.recovery_enabled or not self.action_active or
                self.pose is None or self.finished):
            self.reset_recovery_watchdog()
            return
        if self.near_special_fixed_point():
            self.reset_recovery_watchdog()
            return
        if self.recovery_anchor is None:
            self.reset_recovery_watchdog()
            return
        moved = math.hypot(
            self.pose[0] - self.recovery_anchor[0],
            self.pose[1] - self.recovery_anchor[1])
        if moved >= self.recovery_progress:
            self.reset_recovery_watchdog()
            return
        stationary = (
            rospy.Time.now() - self.recovery_anchor_time).to_sec()
        if stationary >= self.recovery_stuck_time:
            self.start_recovery()
            self.reset_recovery_watchdog()

    def control_loop(self, event):
        now = rospy.Time.now()
        if self.recovery_until > now and self.action_active:
            self.publish_cmd(0.0, self.recovery_turn)
            return
        if self.recovery_replan_pending and self.action_active:
            self.recovery_replan_pending = False
            self.publish_zero("ACTIVE_STUCK_RECOVERY_REPLAN")
            self.last_plan_time = rospy.Time(0)
            self.plan_from_current_pose("active stuck recovery", force=True)
            rospy.logwarn("ACTIVE_STUCK_RECOVERY_REPLAN count=%d",
                          self.recovery_count)
            return
        self.update_recovery_watchdog()
        if self.recovery_until > rospy.Time.now() and self.action_active:
            self.publish_cmd(0.0, self.recovery_turn)
            return
        super(RecoveringConeActionServer, self).control_loop(event)


if __name__ == "__main__":
    RecoveringConeActionServer().spin()
