#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import unittest

from factory_room_wall_handoff_core_v6 import (
    estimated_sign_heading,
    fit_oriented_front_wall,
    nearest_orthogonal_heading,
    ocr_heading_command,
)


class WallHandoffCoreV6Test(unittest.TestCase):
    def test_front_wall_wins_even_when_side_wall_has_more_points(self):
        front = [(0.62, -0.32 + i * 0.025) for i in range(27)]
        side = [(0.08 + i * 0.022, 0.48) for i in range(52)]
        model = fit_oriented_front_wall(
            front + side, 0.03, 12, 0.20, math.radians(48.0))
        self.assertIsNotNone(model)
        self.assertAlmostEqual(model["distance"], 0.62, delta=0.04)
        self.assertLess(abs(math.degrees(model["heading_error"])), 5.0)

    def test_side_wall_alone_is_not_a_front_wall(self):
        side = [(0.08 + i * 0.022, 0.48) for i in range(52)]
        model = fit_oriented_front_wall(
            side, 0.03, 12, 0.20, math.radians(48.0))
        self.assertIsNone(model)

    def test_sign_on_left_commands_positive_yaw(self):
        self.assertGreater(
            ocr_heading_command(-280.0, 800.0, 45.0, 0.40, 0.09, 0.24),
            0.0)

    def test_sign_on_right_commands_negative_yaw(self):
        self.assertLess(
            ocr_heading_command(240.0, 800.0, 45.0, 0.40, 0.09, 0.24),
            0.0)

    def test_centered_sign_does_not_rotate(self):
        self.assertEqual(
            ocr_heading_command(20.0, 800.0, 45.0, 0.40, 0.09, 0.24),
            0.0)

    def test_left_sign_guides_cardinal_fallback_toward_ninety_degrees(self):
        estimated = estimated_sign_heading(
            math.radians(43.0), -282.0, 800.0, math.radians(70.0))
        target = nearest_orthogonal_heading(estimated)
        self.assertAlmostEqual(math.degrees(target), 90.0, delta=0.01)

    def test_centered_sign_snaps_current_heading_to_nearest_cardinal(self):
        estimated = estimated_sign_heading(
            math.radians(102.0), 0.0, 800.0, math.radians(70.0))
        target = nearest_orthogonal_heading(estimated)
        self.assertAlmostEqual(math.degrees(target), 90.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
