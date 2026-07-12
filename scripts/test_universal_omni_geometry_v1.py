#!/usr/bin/env python3
"""Offline checks for unknown-size surface-based obstacle avoidance."""

from universal_omni_navigation_v1 import UniversalOmniNavigator


def make_node():
    node = UniversalOmniNavigator.__new__(UniversalOmniNavigator)
    node.horizon_s = 1.25
    node.sim_dt = 0.10
    node.wall_hard = 0.215
    node.wall_preferred = 0.255
    node.dynamic_hard = 0.225
    node.dynamic_preferred = 0.295
    node.map_hard = 0.16
    node.robot_half_length = 0.171
    node.robot_half_width = 0.128
    node.footprint_margin = 0.010
    node.wall_preferred_gap = 0.025
    node.dynamic_preferred_gap = 0.050
    node.score_path = 4.6
    node.score_goal = 2.8
    node.score_dynamic = 3.0
    node.score_wall = 0.85
    node.score_lateral = 0.18
    node.score_turn = 0.14
    node.score_switch = 0.35
    node.score_speed_reward = 0.90
    node.last_selected_side = 0
    node.grid = None
    node.pose = None
    return node


def main():
    node = make_node()
    target = (0.9, 0.0)
    path = [(0.1 * i, 0.0) for i in range(1, 10)]
    # A wide unknown obstacle is represented only by its measured surface.
    wide_surface = [(0.48, y * 0.04) for y in range(-4, 5)]
    straight = node.evaluate(
        (0.50, 0.0, 0.0), target, path, [], wide_surface)
    side = node.evaluate(
        (0.08, 0.22, 0.0), target, path, [], wide_surface)
    clear = node.evaluate(
        (0.50, 0.0, 0.0), target, path, [], [])
    assert straight is None
    assert side is not None
    assert clear is not None

    # If localization starts already inside the nominal clearance band, an
    # escape command that increases distance must remain legal.
    close_left_surface = [(0.0, 0.20)]
    escape_right = node.evaluate(
        (0.05, -0.12, 0.0), target, path, [], close_left_surface)
    push_left = node.evaluate(
        (0.05, 0.12, 0.0), target, path, [], close_left_surface)
    assert escape_right is not None
    assert push_left is None

    # A 36 cm corridor leaves about 4 cm per side for this 27.6 cm hard
    # footprint.  It is narrow but physically traversable when heading is
    # aligned, and must not be rejected by a diagonal-radius circle.
    corridor_walls = []
    for index in range(1, 10):
        corridor_walls.append((0.1 * index, 0.18))
        corridor_walls.append((0.1 * index, -0.18))
    narrow_straight = node.evaluate(
        (0.30, 0.0, 0.0), target, path, corridor_walls, [])
    assert narrow_straight is not None
    print("universal omni geometry checks passed")


if __name__ == "__main__":
    main()
