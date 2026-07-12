#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Room action bridge with an inflation-equivalent indoor safety profile."""

import rospy
from std_msgs.msg import String

from factory_room_global_first_action_bridge import GlobalFirstActionBridge


class IndoorInflationActionBridge(GlobalFirstActionBridge):
    def __init__(self):
        # The parent starts its control timer during construction.
        self.indoor_profile_active = False
        self.indoor_trigger_y = -1.75
        self.indoor_trigger_less = True
        super(IndoorInflationActionBridge, self).__init__()
        self.indoor_trigger_y = float(rospy.get_param(
            "~indoor_trigger_y", -1.75))
        self.indoor_trigger_less = bool(rospy.get_param(
            "~indoor_trigger_less_than", True))
        self.indoor_hard_clearance = float(rospy.get_param(
            "~indoor_hard_clearance_m", 0.21))
        self.indoor_preferred_clearance = float(rospy.get_param(
            "~indoor_preferred_clearance_m", 0.44))
        self.indoor_emergency_clearance = float(rospy.get_param(
            "~indoor_emergency_clearance_m", 0.18))
        self.indoor_smooth_clearance = float(rospy.get_param(
            "~indoor_smooth_clearance_m", 0.28))
        self.indoor_dynamic_radius = float(rospy.get_param(
            "~indoor_dynamic_obstacle_radius_m", 0.24))
        self.indoor_proximity_weight = float(rospy.get_param(
            "~indoor_proximity_weight", 4.2))
        self.indoor_front_stop = float(rospy.get_param(
            "~indoor_front_stop_m", 0.27))
        self.indoor_front_slow = float(rospy.get_param(
            "~indoor_front_slow_m", 0.45))
        self.indoor_side_stop = float(rospy.get_param(
            "~indoor_side_stop_m", 0.18))
        self.indoor_side_slow = float(rospy.get_param(
            "~indoor_side_slow_m", 0.31))
        self.profile_pub = rospy.Publisher(
            "/factory_room/clearance_profile", String,
            queue_size=1, latch=True)
        self.profile_pub.publish(String(data="DOOR_BASELINE"))
        rospy.logwarn(
            "ROOM_INDOOR_PROFILE_ARMED trigger_y=%.3f baseline_hard=%.3f",
            self.indoor_trigger_y,
            float(self.params.get("hard_clearance_m", 0.0)))

    def crossed_indoor_trigger(self):
        if self.pose is None:
            return False
        if self.indoor_trigger_less:
            return self.pose[1] <= self.indoor_trigger_y
        return self.pose[1] >= self.indoor_trigger_y

    def activate_indoor_profile(self):
        if self.indoor_profile_active:
            return
        with self.goal_lock:
            self.indoor_profile_active = True
            self.params["hard_clearance_m"] = self.indoor_hard_clearance
            self.params["preferred_clearance_m"] = self.indoor_preferred_clearance
            self.params["emergency_min_clearance_m"] = self.indoor_emergency_clearance
            self.params["smooth_clearance_m"] = self.indoor_smooth_clearance
            self.params["proximity_weight"] = self.indoor_proximity_weight
            self.dynamic_obstacle_radius_m = self.indoor_dynamic_radius
            self.front_stop_m = self.indoor_front_stop
            self.front_slow_m = self.indoor_front_slow
            self.side_stop_m = self.indoor_side_stop
            self.side_slow_m = self.indoor_side_slow
            # Doorway observations were recorded with the smaller profile.
            # Clear them only after crossing the trigger, then force a new plan
            # using the larger room-only clearance.
            if self.grid is not None:
                self.grid.clear_dynamic_blocks()
            if self.planner is not None:
                self.planner.roadmaps.clear()
            self.path_world = []
            self.path_index = 0
            self.last_plan_time = rospy.Time(0)
            self.last_blocked_replan = rospy.Time(0)
        self.profile_pub.publish(String(data="INDOOR_INFLATED"))
        rospy.logwarn(
            "ROOM_INDOOR_PROFILE_ACTIVE pose=(%.3f,%.3f) hard=%.3f "
            "preferred=%.3f emergency=%.3f smooth=%.3f dynamic_radius=%.3f",
            self.pose[0], self.pose[1], self.indoor_hard_clearance,
            self.indoor_preferred_clearance,
            self.indoor_emergency_clearance,
            self.indoor_smooth_clearance, self.indoor_dynamic_radius)

    def control_loop(self, event):
        if (self.bridge_active and not self.indoor_profile_active and
                self.crossed_indoor_trigger()):
            self.activate_indoor_profile()
        super(IndoorInflationActionBridge, self).control_loop(event)


if __name__ == "__main__":
    IndoorInflationActionBridge().spin()

