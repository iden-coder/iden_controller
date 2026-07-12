#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Persistent action server built directly on the proven first-stage navigator."""

import math
import subprocess

import actionlib
import rospy
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseFeedback, MoveBaseResult
from std_srvs.srv import Empty, EmptyResponse

from global_first_graph_nav_2249fcf import (
    RosGlobalFirstGraphNavigator,
    quat_from_yaw,
    yaw_from_quat,
)


class PersistentFirstGraphActionServer(RosGlobalFirstGraphNavigator):
    def __init__(self):
        # These defaults are required while the parent creates subscribers and
        # its timer. Velocity publication remains muted until an action is live.
        self.action_active = False
        self.indoor_profile_active = False
        self.indoor_trigger_y = -1.75
        self.indoor_front_mark_range = 0.72
        super(PersistentFirstGraphActionServer, self).__init__()

        self.finished = True
        self.start_pose_accepted = True
        self.waypoint_enabled = False
        self.action_name = rospy.get_param(
            "~action_name", "/factory_room/move_base")
        self.stale_first_nav_node = rospy.get_param(
            "~stale_first_nav_node", "/global_first_graph_nav")
        self.clear_service_name = rospy.get_param(
            "~clear_service_name", "/factory_room/move_base/clear_costmaps")
        self.indoor_trigger_y = float(rospy.get_param(
            "~indoor_trigger_y", -1.75))
        self.entry_waypoint_x = float(rospy.get_param(
            "~entry_waypoint_x", -0.85))
        self.entry_waypoint_y = float(rospy.get_param(
            "~entry_waypoint_y", -1.41))
        self.entry_goal_y_threshold = float(rospy.get_param(
            "~entry_goal_y_threshold", -1.75))
        self.indoor_front_mark_range = float(rospy.get_param(
            "~indoor_front_mark_range_m", 0.72))
        self.indoor_dynamic_radius = float(rospy.get_param(
            "~indoor_dynamic_obstacle_radius_m", 0.30))
        self.indoor_hard_clearance = float(rospy.get_param(
            "~indoor_hard_clearance_m", 0.21))
        self.indoor_preferred_clearance = float(rospy.get_param(
            "~indoor_preferred_clearance_m", 0.42))
        self.indoor_smooth_clearance = float(rospy.get_param(
            "~indoor_smooth_clearance_m", 0.27))
        self.indoor_emergency_clearance = float(rospy.get_param(
            "~indoor_emergency_clearance_m", 0.18))
        self.indoor_max_linear = float(rospy.get_param(
            "~indoor_max_linear_vel", 0.28))
        self.indoor_lookahead = float(rospy.get_param(
            "~indoor_lookahead_dist_m", 0.34))

        self.action_server = actionlib.SimpleActionServer(
            self.action_name, MoveBaseAction,
            execute_cb=self.execute_goal, auto_start=False)
        self.clear_service = rospy.Service(
            self.clear_service_name, Empty, self.clear_dynamic_map)
        self.action_server.start()
        rospy.on_shutdown(self.server_shutdown)
        rospy.logwarn(
            "PERSISTENT_FIRST_GRAPH_READY action=%s map_loaded_once=true",
            self.action_name)

    def publish_cmd(self, vx, wz):
        if self.action_active:
            super(PersistentFirstGraphActionServer, self).publish_cmd(vx, wz)

    def publish_zero(self, reason):
        self.last_cmd = (0.0, 0.0)
        if self.action_active:
            super(PersistentFirstGraphActionServer, self).publish_zero(reason)

    def cb_pose(self, msg):
        super(PersistentFirstGraphActionServer, self).cb_pose(msg)
        if (not self.indoor_profile_active and self.pose is not None and
                self.pose[1] <= self.indoor_trigger_y):
            self.activate_indoor_profile()

    def activate_indoor_profile(self):
        if self.indoor_profile_active:
            return
        self.indoor_profile_active = True
        self.params["hard_clearance_m"] = self.indoor_hard_clearance
        self.params["preferred_clearance_m"] = self.indoor_preferred_clearance
        self.params["smooth_clearance_m"] = self.indoor_smooth_clearance
        self.params["emergency_min_clearance_m"] = self.indoor_emergency_clearance
        self.params["proximity_weight"] = 4.4
        self.dynamic_obstacle_radius_m = self.indoor_dynamic_radius
        self.dynamic_obstacle_trigger_m = self.indoor_front_mark_range
        self.dynamic_obstacle_record_range_m = max(
            self.dynamic_obstacle_record_range_m,
            self.indoor_front_mark_range + 0.08)
        self.dynamic_obstacle_min_interval_s = 0.65
        self.front_angle_deg = max(self.front_angle_deg, 40.0)
        self.front_stop_m = 0.30
        self.front_slow_m = 0.52
        self.side_stop_m = 0.19
        self.side_slow_m = 0.34
        self.front_replan_after_s = 0.30
        self.blocked_replan_s = 0.55
        self.replan_min_interval_s = 0.55
        self.max_linear = min(self.max_linear, self.indoor_max_linear)
        self.lookahead_dist = min(self.lookahead_dist, self.indoor_lookahead)
        if self.planner is not None:
            self.planner.roadmaps.clear()
        self.path_world = []
        self.path_index = 0
        self.last_plan_time = rospy.Time(0)
        rospy.logwarn(
            "FIRST_NAV_INDOOR_PROFILE_ACTIVE pose=(%.3f,%.3f) "
            "hard=%.2f dynamic_radius=%.2f mark_range=%.2f",
            self.pose[0], self.pose[1], self.indoor_hard_clearance,
            self.indoor_dynamic_radius, self.indoor_front_mark_range)

    def apply_scan_guard(self, cmd):
        if (self.action_active and self.indoor_profile_active and
                self.front < self.indoor_front_mark_range):
            added = self.remember_front_dynamic_obstacles(
                "first-nav proactive indoor obstacle")
            if added > 0:
                self.path_world = []
                self.path_index = 0
                self.last_plan_time = rospy.Time(0)
                rospy.logwarn(
                    "FIRST_NAV_DYNAMIC_OBSTACLE_REPLAN front=%.3f cells=%d",
                    self.front, added)
                self.plan_from_current_pose(
                    "proactive indoor obstacle", force=True)
                return 0.0, 0.0
        return super(PersistentFirstGraphActionServer, self).apply_scan_guard(cmd)

    def reset_for_goal(self, goal):
        pose = goal.target_pose.pose
        self.goal_x = float(pose.position.x)
        self.goal_y = float(pose.position.y)
        self.goal_yaw = float(yaw_from_quat(pose.orientation))
        use_entry_waypoint = (
            not self.indoor_profile_active and
            self.goal_y <= self.entry_goal_y_threshold)
        self.waypoint_enabled = use_entry_waypoint
        if use_entry_waypoint:
            self.waypoint_x = self.entry_waypoint_x
            self.waypoint_y = self.entry_waypoint_y
            self.goal_stage = 0
            self.active_goal = (self.waypoint_x, self.waypoint_y)
            rospy.logwarn(
                "FIRST_NAV_ENTRY_WAYPOINT enabled=(%.3f,%.3f)",
                self.waypoint_x, self.waypoint_y)
        else:
            self.goal_stage = 1
            self.active_goal = (self.goal_x, self.goal_y)
        self.path_world = []
        self.path_index = 0
        self.finished = False
        self.goal_enter_time = None
        self.waypoint_enter_time = None
        self.front_block_start = None
        self.replan_fail_count = 0
        self.last_plan_time = rospy.Time(0)
        self.last_blocked_replan = rospy.Time(0)
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pose = None
        self.last_progress_yaw = None
        self.last_goal_dist = float("inf")
        self.last_cmd = (0.0, 0.0)
        self.last_control_time = rospy.Time.now()
        self.start_pose_accepted = True

    def execute_goal(self, goal):
        frame = (goal.target_pose.header.frame_id or self.map_frame).lstrip("/")
        if frame != self.map_frame.lstrip("/"):
            self.action_server.set_aborted(
                MoveBaseResult(), "only map-frame goals are supported")
            return
        # The QR navigator deliberately stays alive after scanning and keeps
        # publishing zero velocity. Remove only that stale node before this
        # already-loaded server takes ownership of cmd_vel_raw.
        try:
            subprocess.call(
                ["rosnode", "kill", self.stale_first_nav_node],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        self.action_active = True
        self.reset_for_goal(goal)
        rospy.logwarn(
            "PERSISTENT_FIRST_GRAPH_GOAL x=%.3f y=%.3f yaw=%.1fdeg",
            self.goal_x, self.goal_y, math.degrees(self.goal_yaw))

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.action_server.is_preempt_requested():
                self.finished = True
                self.publish_zero("ACTION_PREEMPTED")
                self.action_active = False
                self.action_server.set_preempted(
                    MoveBaseResult(), "goal preempted")
                return
            if self.finished:
                self.publish_zero("ACTION_REACHED")
                self.action_active = False
                self.action_server.set_succeeded(
                    MoveBaseResult(), "persistent first-graph goal reached")
                rospy.logwarn(
                    "PERSISTENT_FIRST_GRAPH_REACHED x=%.3f y=%.3f",
                    self.goal_x, self.goal_y)
                return
            self.publish_feedback()
            rate.sleep()

    def publish_feedback(self):
        if self.pose is None:
            return
        feedback = MoveBaseFeedback()
        feedback.base_position = PoseStamped()
        feedback.base_position.header.stamp = rospy.Time.now()
        feedback.base_position.header.frame_id = self.map_frame
        feedback.base_position.pose.position.x = self.pose[0]
        feedback.base_position.pose.position.y = self.pose[1]
        qx, qy, qz, qw = quat_from_yaw(self.pose[2])
        feedback.base_position.pose.orientation.x = qx
        feedback.base_position.pose.orientation.y = qy
        feedback.base_position.pose.orientation.z = qz
        feedback.base_position.pose.orientation.w = qw
        self.action_server.publish_feedback(feedback)

    def clear_dynamic_map(self, _request):
        if self.grid is not None:
            self.grid.clear_dynamic_blocks()
        if self.planner is not None:
            self.planner.roadmaps.clear()
        self.path_world = []
        self.path_index = 0
        self.last_plan_time = rospy.Time(0)
        rospy.logwarn("PERSISTENT_FIRST_GRAPH_DYNAMIC_MAP_CLEARED")
        return EmptyResponse()

    def server_shutdown(self):
        self.finished = True
        if self.action_active:
            self.publish_zero("SERVER_SHUTDOWN")
        self.action_active = False


if __name__ == "__main__":
    PersistentFirstGraphActionServer().spin()
