#!/usr/bin/env python3

"""Deterministic geometry checks for the integrated room command guard."""

from factory_room_rect_sweep_command_guard_v1 import (
    FactoryRoomRectSweepCommandGuard,
)


def make_guard():
    guard = FactoryRoomRectSweepCommandGuard.__new__(
        FactoryRoomRectSweepCommandGuard)
    guard.half_length = 0.171
    guard.half_width = 0.128
    guard.margin = 0.015
    guard.horizon = 0.80
    guard.dt = 0.05
    guard.worsen_allowance = 0.003
    return guard


def main():
    guard = make_guard()

    obstacle = [(0.148, 0.176)]
    assert guard._trajectory_result(0.20, 0.0, 0.0, obstacle)[0]
    assert not guard._trajectory_result(0.0, 0.0, 0.50, obstacle)[0]

    narrow_corridor = [(0.30, 0.16), (0.30, -0.16)]
    assert guard._trajectory_result(
        0.20, 0.0, 0.0, narrow_corridor)[0]
    print("INTEGRATED_RECT_COMMAND_GUARD_TEST_PASS")


if __name__ == "__main__":
    main()
