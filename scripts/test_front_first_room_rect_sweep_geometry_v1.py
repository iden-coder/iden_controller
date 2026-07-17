#!/usr/bin/env python3

"""Deterministic geometry checks for the room rectangular sweep guard."""

from front_first_room_rect_sweep_route_test_v1 import (
    FrontFirstRoomRectSweepRouteTest,
)


def make_guard(points):
    guard = FrontFirstRoomRectSweepRouteTest.__new__(
        FrontFirstRoomRectSweepRouteTest)
    guard.robot_half_length = 0.171
    guard.robot_half_width = 0.128
    guard.footprint_margin = 0.015
    guard.guard_horizon = 0.80
    guard.guard_dt = 0.05
    guard.guard_worsen_allowance = 0.003
    guard.rect_scan_points = points
    return guard


def main():
    # A surface point outside the current rectangle is swept by a left turn,
    # while a straight forward command moves away from it safely.
    guard = make_guard([(0.148, 0.176)])
    assert guard._trajectory_result(0.20, 0.0)[0]
    assert not guard._trajectory_result(0.0, 0.50)[0]

    # Parallel walls 32 cm apart admit the 25.6 cm wide aligned rectangle.
    # This verifies that the guard does not replace it with a large circle.
    guard = make_guard([(0.30, 0.16), (0.30, -0.16)])
    assert guard._trajectory_result(0.20, 0.0)[0]
    print("RECT_SWEEP_GEOMETRY_TEST_PASS")


if __name__ == "__main__":
    main()
