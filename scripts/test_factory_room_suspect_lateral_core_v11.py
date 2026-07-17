#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_suspect_lateral_core_v11 import (
    lateral_acquisition_exit,
    near_scan_micro_command,
    orthogonal_alignment_command,
    orthogonal_handoff_ready,
    physical_view_ignore_decision,
    suspect_lateral_command,
)


class SuspectLateralCoreV11Test(unittest.TestCase):
    def decide(self, bbox):
        return suspect_lateral_command(
            bbox, 800, 70, -1.0, 0.10, 0.025, 0.055)

    def test_left_edge_moves_robot_left_for_installed_camera(self):
        state, command, error = self.decide([0, 80, 100, 160])
        self.assertEqual(state, "move")
        self.assertGreater(command, 0.0)
        self.assertLess(error, 0.0)
        self.assertLessEqual(command, 0.055)

    def test_right_edge_moves_robot_right_for_installed_camera(self):
        state, command, error = self.decide([690, 80, 800, 160])
        self.assertEqual(state, "move")
        self.assertLess(command, 0.0)
        self.assertGreater(error, 0.0)

    def test_centered_sign_stops(self):
        self.assertEqual(
            self.decide([320, 80, 460, 160]),
            ("centered", 0.0, -10.0))

    def test_invalid_bbox_is_rejected(self):
        self.assertEqual(self.decide([1, 2, 3]), ("invalid", 0.0, 0.0))

    def test_exit_reasons_are_deterministic(self):
        self.assertEqual(
            lateral_acquisition_exit(0.18, 0.18, 0.1, 0.7, 0.2, 1.2),
            "shift_limit")
        self.assertEqual(
            lateral_acquisition_exit(0.05, 0.18, 0.8, 0.7, 0.2, 1.2),
            "ocr_stale")
        self.assertEqual(
            lateral_acquisition_exit(0.05, 0.18, 0.1, 0.7, 1.3, 1.2),
            "blocked")
        self.assertEqual(
            lateral_acquisition_exit(0.05, 0.18, 0.1, 0.7, 0.2, 1.2),
            "")

    def test_uninitialized_clock_is_a_safe_exit_not_invalid(self):
        self.assertEqual(
            lateral_acquisition_exit(
                0.0, 0.18, float("inf"), 0.7, 0.0, 1.2),
            "ocr_stale")
        self.assertEqual(
            lateral_acquisition_exit(
                0.0, 0.18, 0.1, 0.7, float("inf"), 1.2),
            "blocked")

    def test_non_target_view_needs_time_and_yaw_separation(self):
        blocked, _ = physical_view_ignore_decision(
            1.0, 2.8, 8.0, 1.0, 1.0, True, 0.38)
        self.assertTrue(blocked)
        blocked, _ = physical_view_ignore_decision(
            3.0, 2.8, 8.0, 1.1, 1.0, True, 0.38)
        self.assertTrue(blocked)
        blocked, delta = physical_view_ignore_decision(
            3.0, 2.8, 8.0, 1.5, 1.0, True, 0.38)
        self.assertFalse(blocked)
        self.assertAlmostEqual(delta, 0.5)

    def test_non_target_view_has_route_and_hard_timeout_release(self):
        self.assertFalse(physical_view_ignore_decision(
            1.0, 2.8, 8.0, 1.0, 1.0, False, 0.38)[0])
        self.assertFalse(physical_view_ignore_decision(
            8.0, 2.8, 8.0, 1.0, 1.0, True, 0.38)[0])

    def test_wall_alignment_prefers_lidar_and_limits_rotation(self):
        state, command, error, source = orthogonal_alignment_command(
            0.40, -0.80, 0.05, 1.35, 0.10, 0.28)
        self.assertEqual((state, source), ("rotate", "lidar"))
        self.assertAlmostEqual(error, 0.40)
        self.assertAlmostEqual(command, 0.28)

    def test_wall_alignment_uses_cardinal_fallback_and_stops(self):
        state, command, error, source = orthogonal_alignment_command(
            None, 0.03, 0.05, 1.35, 0.10, 0.28)
        self.assertEqual((state, source), ("aligned", "cardinal_fallback"))
        self.assertEqual(command, 0.0)
        self.assertAlmostEqual(error, 0.03)

    def test_handoff_requires_wall_target_and_image_center(self):
        self.assertTrue(orthogonal_handoff_ready(
            0.03, 0.06, "食品加工车间", "食品加工车间", 30.0, 45.0))
        self.assertFalse(orthogonal_handoff_ready(
            0.12, 0.06, "食品加工车间", "食品加工车间", 30.0, 45.0))
        self.assertFalse(orthogonal_handoff_ready(
            0.03, 0.06, "日用品加工车间", "食品加工车间", 10.0, 45.0))
        self.assertFalse(orthogonal_handoff_ready(
            0.03, 0.06, "食品加工车间", "食品加工车间", 80.0, 45.0))

    def test_near_scan_micro_uses_small_forward_lateral_and_turn(self):
        state, vx, vy, wz = near_scan_micro_command(
            0.20, -0.06, 0.12, 0.12, 0.35,
            0.8, 0.8, 0.9, 0.10, 0.045, 0.20)
        self.assertEqual(state, "move")
        self.assertGreater(vx, 0.0)
        self.assertLess(vy, 0.0)
        self.assertGreater(wz, 0.0)
        self.assertLessEqual(vx, 0.10)
        self.assertGreaterEqual(vy, -0.045)

    def test_near_scan_micro_never_requests_unsafe_reverse(self):
        state, vx, vy, _ = near_scan_micro_command(
            -0.12, 0.08, 0.5, 0.10, 0.35,
            0.8, 0.8, 0.9, 0.10, 0.045, 0.20)
        self.assertEqual(state, "move")
        self.assertEqual(vx, 0.0)
        self.assertGreater(vy, 0.0)

    def test_near_scan_micro_stops_inside_position_tolerance(self):
        self.assertEqual(near_scan_micro_command(
            0.05, 0.04, 1.0, 0.10, 0.35,
            0.8, 0.8, 0.9, 0.10, 0.045, 0.20),
            ("reached", 0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
