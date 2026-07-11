#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Expose the existing global-first navigator as a reusable MoveBase action."""

import math
import threading

import actionlib
import rospy
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import (
    MoveBaseAction, MoveBaseFeedback, MoveBaseResult)
from std_srvs.srv import Empty, EmptyResponse

from global_first_graph_nav_2249fcf import (
    RosGlobalFirstGraphNavigator, quat_from_yaw, yaw_from_quat)


class GlobalFirstActionBridge(RosGlobalFirstGraphNavigator):
    def __init__(self):
        # The parent starts its timer during construction. Keep it silent until
        # the room manager sends a goal so QR scanning retains cmd_vel control.
        self.bridge_active = False
        self.goal_lock = threading.RLock()
        super(GlobalFirstActionBridge, self).__init__()

        self.finished = True
        self.waypoint_enabled = False
        self.start_pose_accepted = True
        self.action_name = rospy.get_param(
            "~action_name", "/factory_room/move_base")
        self.clear_service_name = rospy.get_param(
            "~clear_service_name",
            "/factory_room/move_base/clear_costmaps")

        self.action_server = actionlib.SimpleActionServer(
            self.action_name, MoveBaseAction,
            execute_cb=self.execute_goal, auto_start=False)
        self.clear_service = rospy.Service(
            self.clear_service_name, Empty, self.clear_dynamic_map)
        self.action_server.start()
        rospy.on_shutdown(self.bridge_shutdown)
        rospy.logwarn(
            "ROOM_GLOBAL_FIRST_BRIDGE_READY action=%s clear=%s",
            self.action_name, self.clear_service_name)

    def control_loop(self, event):
        if not self.bridge_active:
            return
        super(GlobalFirstActionBridge, self).control_loop(event)

    def reset_for_goal(self, goal):
        pose = goal.target_pose.pose
        self.goal_x = float(pose.position.x)
        self.goal_y = float(pose.position.y)
        self.goal_yaw = float(yaw_from_quat(pose.orientation))
        self.waypoint_enabled = False
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
            rospy.logerr("ROOM_GLOBAL_FIRST_REJECT frame=%s", frame)
            self.action_server.set_aborted(
                MoveBaseResult(), "only map-frame goals are supported")
            return

        with self.goal_lock:
            self.reset_for_goal(goal)
            self.bridge_active = True
        rospy.logwarn(
            "ROOM_GLOBAL_FIRST_GOAL x=%.3f y=%.3f yaw=%.1fdeg",
            self.goal_x, self.goal_y, math.degrees(self.goal_yaw))

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self.action_server.is_preempt_requested():
                with self.goal_lock:
                    self.bridge_active = False
                    self.finished = True
                self.publish_zero("ROOM_GOAL_PREEMPTED")
                self.action_server.set_preempted(
                    MoveBaseResult(), "room goal preempted")
                rospy.logwarn("ROOM_GLOBAL_FIRST_PREEMPTED")
                return

            if self.finished:
                with self.goal_lock:
                    self.bridge_active = False
                self.publish_zero("ROOM_GOAL_REACHED")
                self.action_server.set_succeeded(
                    MoveBaseResult(), "global-first goal reached")
                rospy.logwarn(
                    "ROOM_GLOBAL_FIRST_REACHED x=%.3f y=%.3f",
                    self.goal_x, self.goal_y)
                return

            self.publish_action_feedback()
            rate.sleep()

        with self.goal_lock:
            self.bridge_active = False
            self.finished = True
        self.publish_zero("ROOM_BRIDGE_SHUTDOWN")
        if self.action_server.is_active():
            self.action_server.set_aborted(
                MoveBaseResult(), "bridge shutting down")

    def publish_action_feedback(self):
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
        with self.goal_lock:
            if self.grid is not None:
                self.grid.clear_dynamic_blocks()
            if self.planner is not None:
                self.planner.roadmaps.clear()
            self.path_world = []
            self.path_index = 0
            self.last_plan_time = rospy.Time(0)
        rospy.logwarn("ROOM_GLOBAL_FIRST_DYNAMIC_MAP_CLEARED")
        return EmptyResponse()

    def bridge_shutdown(self):
        self.bridge_active = False
        self.finished = True
        self.publish_zero("ROOM_BRIDGE_SHUTDOWN")


if __name__ == "__main__":
    node = GlobalFirstActionBridge()
    node.spin()

