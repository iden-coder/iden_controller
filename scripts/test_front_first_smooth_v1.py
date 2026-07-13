#!/usr/bin/env python3
"""Offline behavior checks for front-first continuous tracking."""

import math

import rospy

from front_first_smooth_navigation_v1 import FrontFirstSmoothNavigator


def make_node():
    node = FrontFirstSmoothNavigator.__new__(FrontFirstSmoothNavigator)
    node.pose = (0.0, 0.0, 0.0)
    node.active_goal = (2.0, 0.0)
    node.goal_stage = 1
    node.waypoint_enabled = False
    node.cruise_speed = 0.48
    node.minimum_curve_speed = 0.065
    node.max_forward_cmd = 0.48
    node.max_lateral_cmd = 0.055
    node.max_turn_cmd = 0.72
    node.heading_gain = 1.45
    node.cross_track_gain = 0.55
    node.curve_slow_angle = math.radians(52.0)
    node.final_slow_radius = 0.48
    node.final_creep_speed = 0.075
    node.final_slow_speed = 0.22
    node.avoid_influence = 0.52
    node.avoid_stop = 0.20
    node.lateral_trigger = 0.32
    node.lateral_side_need = 0.18
    node.avoid_turn_gain = 0.42
    node.wall_center_gain = 0.22
    node.unstick_until = rospy.Time(0)
    node.unstick_sign = 1.0
    node.unstick_forward = 0.055
    node.unstick_lateral = 0.035
    node.unstick_turn = 0.26
    node.front = 2.0
    node.left = 2.0
    node.right = 2.0
    node.requested_vy = 0.0
    return node


def main():
    rospy.rostime.set_rostime_initialized(True)
    node = make_node()
    vx, wz = node.compute_cmd((0.8, 0.25))
    assert vx > 0.0 and wz > 0.0
    assert abs(node.requested_vy) < 1e-9

    node.front = 0.27
    node.left = 0.60
    node.right = 0.22
    vx, wz = node.compute_cmd((0.8, 0.0))
    assert vx > 0.0 and wz > 0.0
    assert 0.0 < node.requested_vy <= 0.055
    print("front-first smooth checks passed")


if __name__ == "__main__":
    main()
