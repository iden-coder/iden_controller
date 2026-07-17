#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V3 room navigation with duplicate-cone track consolidation."""

import math

import rospy

from factory_room_front_first_stable_action_v3 import (
    RecoveringStableRoomActionBridge,
)


class ConsolidatingStableRoomActionBridge(
        RecoveringStableRoomActionBridge):
    def __init__(self):
        self.track_merge_ready = False
        super(ConsolidatingStableRoomActionBridge, self).__init__()
        self.track_merge_radius = float(rospy.get_param(
            "~dynamic_track_merge_radius_m", 0.14))
        self.normal_require_goal_yaw = bool(self.require_goal_yaw)
        self.d3_position_only_min_x = float(rospy.get_param(
            "~d3_position_only_min_x", 2.15))
        self.d3_position_only_max_y = float(rospy.get_param(
            "~d3_position_only_max_y", -2.05))
        self.track_merge_ready = True
        rospy.logwarn(
            "ROOM_CONE_TRACK_V4 merge_radius=%.2fm wall_reject=%.2fm "
            "ttl=%.1fs",
            self.track_merge_radius, self.wall_rejection_clearance,
            self.confirmed_ttl_s)

    def reset_for_goal(self, goal):
        pose = goal.target_pose.pose
        d3_region = (
            float(pose.position.x) >= self.d3_position_only_min_x and
            float(pose.position.y) <= self.d3_position_only_max_y)
        self.require_goal_yaw = (
            False if d3_region else self.normal_require_goal_yaw)
        super(ConsolidatingStableRoomActionBridge, self).reset_for_goal(goal)
        rospy.logwarn(
            "ROOM_GOAL_ACCEPTANCE position_only=%s x=%.3f y=%.3f",
            str(d3_region).lower(), self.goal_x, self.goal_y)

    @staticmethod
    def _earlier(first, second):
        return first if first.to_sec() <= second.to_sec() else second

    @staticmethod
    def _later(first, second):
        return first if first.to_sec() >= second.to_sec() else second

    def _coalesce_tracks(self):
        if not self.track_merge_ready or len(self.obstacle_tracks) < 2:
            return
        ordered = sorted(
            self.obstacle_tracks,
            key=lambda track: (not track.confirmed, -track.hits,
                               track.track_id))
        retained = []
        merged_count = 0
        for track in ordered:
            keeper = None
            keeper_distance = self.track_merge_radius
            for candidate in retained:
                distance = math.hypot(
                    track.x - candidate.x, track.y - candidate.y)
                if distance <= keeper_distance:
                    keeper = candidate
                    keeper_distance = distance
            if keeper is None:
                retained.append(track)
                continue
            old_weight = max(1, keeper.hits)
            new_weight = max(1, track.hits)
            total = float(old_weight + new_weight)
            keeper.x = (keeper.x * old_weight + track.x * new_weight) / total
            keeper.y = (keeper.y * old_weight + track.y * new_weight) / total
            keeper.hits = min(10000, old_weight + new_weight)
            keeper.confirmed = keeper.confirmed or track.confirmed
            keeper.first_seen = self._earlier(
                keeper.first_seen, track.first_seen)
            keeper.last_seen = self._later(
                keeper.last_seen, track.last_seen)
            merged_count += 1
        if merged_count:
            self.obstacle_tracks = retained
            self.dynamic_layer_dirty = True
            rospy.logwarn_throttle(
                0.8,
                "ROOM_CONE_TRACKS_MERGED duplicates=%d retained=%d",
                merged_count, len(retained))

    def _update_tracks(self, candidates, stamp):
        super(ConsolidatingStableRoomActionBridge, self)._update_tracks(
            candidates, stamp)
        self._coalesce_tracks()


if __name__ == "__main__":
    ConsolidatingStableRoomActionBridge().spin()
