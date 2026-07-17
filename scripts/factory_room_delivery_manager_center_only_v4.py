#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V3 delivery manager with reachable nearby views for wall-side d3."""

import math

import rospy

from factory_room_delivery_manager_center_only_v3 import (
    FastClearanceCenterlineManager,
)


class NearbyViewFactoryDeliveryManager(FastClearanceCenterlineManager):
    def __init__(self):
        super(NearbyViewFactoryDeliveryManager, self).__init__()
        raw_offsets = rospy.get_param(
            "~d3_nearby_offsets_m",
            [[0.0, 0.14], [0.0, 0.26], [-0.16, 0.20],
             [-0.28, 0.24], [-0.26, 0.38]])
        self.d3_nearby_offsets = []
        for offset in raw_offsets:
            if isinstance(offset, (list, tuple)) and len(offset) == 2:
                self.d3_nearby_offsets.append(
                    (float(offset[0]), float(offset[1])))
        self.d3_nearby_offsets = self.d3_nearby_offsets or [(0.0, 0.20)]
        rospy.logwarn(
            "D3_NEARBY_VIEW_V4 position_only=true candidates=%s",
            self.d3_nearby_offsets)

    def navigate_once(self, name, x, y, yaw):
        previous_retries = self.nav_goal_retries
        self.nav_goal_retries = 1
        try:
            return self.navigate(name, x, y, yaw, allow_offsets=False)
        finally:
            self.nav_goal_retries = previous_retries

    def navigate_d3_or_nearby(self, point):
        candidates = [("d3", point["x"], point["y"])]
        candidates.extend((
            "d3_near_%02d" % (index + 1),
            point["x"] + offset_x,
            point["y"] + offset_y)
            for index, (offset_x, offset_y) in enumerate(
                self.d3_nearby_offsets))
        for index, (name, x, y) in enumerate(candidates):
            if rospy.is_shutdown() or self._mission_expired():
                return None
            self.publish_state(
                "D3_VIEW_NAVIGATING", candidate=name,
                candidate_index=index + 1, candidate_count=len(candidates),
                x=x, y=y)
            if self.navigate_once(name, x, y, point["yaw"]):
                self.publish_state(
                    "D3_VIEW_REACHED", candidate=name, x=x, y=y,
                    exact=(index == 0))
                rospy.logwarn(
                    "D3_VIEW_ACCEPTED name=%s exact=%s x=%.3f y=%.3f",
                    name, str(index == 0).lower(), x, y)
                return name
            rospy.logwarn(
                "D3_VIEW_UNREACHABLE name=%s; trying a nearby view",
                name)
        return None

    def find_target_workshop(self):
        for cycle in range(max(1, self.search_cycles)):
            points = (self.observation_points if cycle % 2 == 0
                      else list(reversed(self.observation_points)))
            for point in points:
                if rospy.is_shutdown() or self._mission_expired():
                    return False
                self.ocr_control("disable")
                if point["name"] == "d3":
                    view_name = self.navigate_d3_or_nearby(point)
                    if view_name is None:
                        continue
                    reached = True
                    scan_name = view_name
                else:
                    reached = self.navigate(
                        point["name"], point["x"], point["y"],
                        point["yaw"], allow_offsets=True)
                    scan_name = point["name"]
                if not reached:
                    continue
                if self.scan_for_target_at_current_pose(scan_name):
                    return True
            rospy.logwarn(
                "WORKSHOP_SEARCH_CYCLE_COMPLETE cycle=%d/%d target=%s",
                cycle + 1, self.search_cycles, self.target_warehouse)
            self.clear_navigation_costmaps()
        return False


if __name__ == "__main__":
    NearbyViewFactoryDeliveryManager().run()
