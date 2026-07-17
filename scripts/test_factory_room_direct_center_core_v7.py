#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_direct_center_core_v7 import (
    direct_lateral_decision,
    progress_state,
)


class DirectCenterCoreV7Test(unittest.TestCase):
    def test_old_twenty_pixel_tolerance_finishes(self):
        result = direct_lateral_decision(
            -17.2, 20.0, 0.02, 0.40, 0.075, 0.22, 0.026, 0.032)
        self.assertEqual(result["action"], "centered")

    def test_narrow_clearance_keeps_effective_nonzero_speed(self):
        result = direct_lateral_decision(
            -47.0, 20.0, 0.049, 0.078, 0.075, 0.22, 0.026, 0.032)
        self.assertEqual(result["action"], "move")
        self.assertAlmostEqual(result["command"], 0.026, delta=1e-6)
        self.assertEqual(result["guard"], "narrow")

    def test_hard_clearance_still_blocks_motion(self):
        result = direct_lateral_decision(
            50.0, 20.0, -0.05, 0.074, 0.075, 0.22, 0.026, 0.032)
        self.assertEqual(result["action"], "blocked")
        self.assertEqual(result["command"], 0.0)

    def test_clear_space_keeps_requested_direction(self):
        result = direct_lateral_decision(
            60.0, 20.0, -0.055, 0.50, 0.075, 0.22, 0.026, 0.032)
        self.assertEqual(result["action"], "move")
        self.assertLess(result["command"], 0.0)

    def test_progress_resets_stall_clock(self):
        result = progress_state(47.0, 40.0, 1.0, 3.0, 2.0, 1.2, 3.6)
        self.assertEqual(result["best_error"], 40.0)
        self.assertEqual(result["last_progress_time"], 3.0)
        self.assertFalse(result["boost"])

    def test_no_progress_boosts_then_fails(self):
        boosted = progress_state(47.0, 46.5, 1.0, 2.3, 2.0, 1.2, 3.6)
        failed = progress_state(47.0, 46.5, 1.0, 4.7, 2.0, 1.2, 3.6)
        self.assertTrue(boosted["boost"])
        self.assertFalse(boosted["failed"])
        self.assertTrue(failed["failed"])


if __name__ == "__main__":
    unittest.main()

