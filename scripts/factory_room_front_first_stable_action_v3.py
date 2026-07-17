#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V2 stable room navigation plus rear-clearance-gated cone recovery."""

import math

import rospy
from geometry_msgs.msg import Twist

from factory_room_front_first_stable_action_v2 import (
    FactoryRoomFrontFirstStableActionBridgeV2,
)


class RecoveringStableRoomActionBridge(
        FactoryRoomFrontFirstStableActionBridgeV2):
    def __init__(self):
        self.recovery_ready = False
        self.recovery_active = False
        self.recovery_until = rospy.Time(0)
        self.recovery_start_pose = None
        self.translation_progress_pose = None
        self.translation_progress_time = rospy.Time(0)
        self.last_recovery_end = rospy.Time(0)
        super(RecoveringStableRoomActionBridge, self).__init__()
        self.recovery_trigger_s = float(rospy.get_param(
            "~room_recovery_trigger_s", 3.4))
        self.recovery_back_speed = float(rospy.get_param(
            "~room_recovery_back_speed_mps", 0.060))
        self.recovery_back_distance = float(rospy.get_param(
            "~room_recovery_back_distance_m", 0.11))
        self.recovery_timeout_s = float(rospy.get_param(
            "~room_recovery_timeout_s", 2.8))
        self.recovery_rear_clear = float(rospy.get_param(
            "~room_recovery_rear_clear_m", 0.31))
        self.recovery_track_near = float(rospy.get_param(
            "~room_recovery_track_near_m", 0.46))
        self.recovery_progress_m = float(rospy.get_param(
            "~room_recovery_progress_m", 0.035))
        self.recovery_cooldown_s = float(rospy.get_param(
            "~room_recovery_cooldown_s", 2.5))
        self.recovery_ready = True
        self._reset_translation_watch()
        rospy.logwarn(
            "ROOM_RECOVERY_V3_READY trigger=%.1fs back=%.2fm@%.3fmps "
            "rear_need=%.2fm",
            self.recovery_trigger_s, self.recovery_back_distance,
            self.recovery_back_speed, self.recovery_rear_clear)

    def _reset_translation_watch(self):
        self.translation_progress_pose = (
            None if self.pose is None else (self.pose[0], self.pose[1]))
        self.translation_progress_time = rospy.Time.now()

    def reset_for_goal(self, goal):
        super(RecoveringStableRoomActionBridge, self).reset_for_goal(goal)
        self.recovery_active = False
        self._reset_translation_watch()

    def _update_translation_watch(self):
        if self.pose is None:
            return
        current = (self.pose[0], self.pose[1])
        if self.translation_progress_pose is None:
            self.translation_progress_pose = current
            self.translation_progress_time = rospy.Time.now()
            return
        if math.hypot(
                current[0] - self.translation_progress_pose[0],
                current[1] - self.translation_progress_pose[1]) >= \
                self.recovery_progress_m:
            self.translation_progress_pose = current
            self.translation_progress_time = rospy.Time.now()

    def _nearest_confirmed_track(self):
        if self.pose is None:
            return float("inf")
        return min((math.hypot(track.x - self.pose[0],
                               track.y - self.pose[1])
                    for track in self._confirmed_tracks()),
                   default=float("inf"))

    def _direct_recovery_command(self, speed):
        command = Twist()
        command.linear.x = float(speed)
        self.cmd_pub.publish(command)

    def _start_recovery(self, now, nearest):
        if self.rear < self.recovery_rear_clear:
            rospy.logwarn_throttle(
                1.0,
                "ROOM_RECOVERY_REJECTED rear=%.3f need=%.3f nearest=%.3f",
                self.rear, self.recovery_rear_clear, nearest)
            return False
        self.recovery_active = True
        self.recovery_until = now + rospy.Duration(self.recovery_timeout_s)
        self.recovery_start_pose = (
            None if self.pose is None else (self.pose[0], self.pose[1]))
        self.requested_vy = 0.0
        self.last_vy = 0.0
        self.last_cmd = (0.0, 0.0)
        rospy.logwarn(
            "ROOM_RECOVERY_START nearest_cone=%.3f rear=%.3f target=%.3fm",
            nearest, self.rear, self.recovery_back_distance)
        self._direct_recovery_command(-self.recovery_back_speed)
        return True

    def _finish_recovery(self, reason):
        self._direct_recovery_command(0.0)
        self.recovery_active = False
        self.last_recovery_end = rospy.Time.now()
        self.path_world = []
        self.path_index = 0
        self.last_progress_time = rospy.Time.now()
        self._reset_translation_watch()
        rospy.logwarn("ROOM_RECOVERY_END reason=%s; replanning", reason)
        self.plan_from_current_pose("cone clearance recovery", force=True)

    def _run_recovery(self, now):
        if (not self.bridge_active or not self.pose_fresh() or
                not self.scan_fresh()):
            self._finish_recovery("input_unavailable")
            return
        moved = 0.0
        if self.recovery_start_pose is not None and self.pose is not None:
            moved = math.hypot(
                self.pose[0] - self.recovery_start_pose[0],
                self.pose[1] - self.recovery_start_pose[1])
        if self.rear < self.recovery_rear_clear:
            self._finish_recovery("rear_clearance")
            return
        if moved >= self.recovery_back_distance:
            self._finish_recovery("distance")
            return
        if now >= self.recovery_until:
            self._finish_recovery("timeout")
            return
        self._direct_recovery_command(-self.recovery_back_speed)
        rospy.logwarn_throttle(
            0.35, "ROOM_RECOVERY_BACKING moved=%.3f/%.3f rear=%.3f",
            moved, self.recovery_back_distance, self.rear)

    def control_loop(self, event):
        if not self.recovery_ready:
            super(RecoveringStableRoomActionBridge, self).control_loop(event)
            return
        now = rospy.Time.now()
        if self.recovery_active:
            self._run_recovery(now)
            return

        self._update_translation_watch()
        stalled = (now - self.translation_progress_time).to_sec()
        nearest = self._nearest_confirmed_track()
        goal_far = False
        if self.bridge_active and self.pose is not None:
            goal_far = (self.distance_to_active_goal() >
                        self.current_target_tolerance() + 0.16)
        cooldown_done = (
            (now - self.last_recovery_end).to_sec() >=
            self.recovery_cooldown_s)
        if (self.bridge_active and self.indoor_active and goal_far and
                cooldown_done and stalled >= self.recovery_trigger_s and
                nearest <= self.recovery_track_near):
            if self._start_recovery(now, nearest):
                return
        super(RecoveringStableRoomActionBridge, self).control_loop(event)


if __name__ == "__main__":
    RecoveringStableRoomActionBridge().spin()
