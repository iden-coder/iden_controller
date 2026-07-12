#!/usr/bin/env python3
"""Offline geometry checks for the hybrid omni trajectory evaluator."""

from persistent_hybrid_omni_action_server_v1 import HybridOmniActionServer


def make_selector():
    node = HybridOmniActionServer.__new__(HybridOmniActionServer)
    node.omni_control_horizon = 1.10
    node.omni_sim_dt = 0.11
    node.wall_hard_clearance = 0.18
    node.wall_preferred_clearance = 0.23
    node.dynamic_hard_clearance = 0.22
    node.dynamic_preferred_clearance = 0.32
    node.static_map_hard_clearance = 0.18
    node.score_path_weight = 4.2
    node.score_goal_weight = 3.0
    node.score_dynamic_weight = 2.8
    node.score_wall_weight = 0.75
    node.score_lateral_weight = 0.25
    node.score_turn_weight = 0.18
    node.score_switch_weight = 0.32
    node.last_selected_side = 0
    node.grid = None
    node.pose = None
    return node


def main():
    node = make_selector()
    target = (0.8, 0.0)
    path = [(0.1 * i, 0.0) for i in range(1, 9)]
    cone_surface = [(0.34, 0.0), (0.35, 0.04), (0.35, -0.04)]

    straight = node.evaluate_candidate(
        (0.27, 0.0, 0.0), target, path, [], cone_surface)
    side_step = node.evaluate_candidate(
        (0.06, 0.16, 0.0), target, path, [], cone_surface)
    assert straight is None, "straight trajectory must collide with the cone"
    assert side_step is not None, "safe lateral trajectory must remain available"

    clear = node.evaluate_candidate(
        (0.27, 0.0, 0.0), target, path, [], [])
    assert clear is not None, "straight trajectory must remain valid in free space"
    print("hybrid omni geometry checks passed")


if __name__ == "__main__":
    main()
