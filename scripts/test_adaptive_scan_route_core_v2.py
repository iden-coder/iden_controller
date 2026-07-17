#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import unittest

from adaptive_scan_route_core_v2 import (
    approach_point,
    arrival_heading_correction,
    merge_groups,
)


class AdaptiveScanRouteCoreV2Test(unittest.TestCase):
    def test_approach_point_is_behind_goal(self):
        x, y = approach_point(1.0, 2.0, math.pi / 2.0, 0.4)
        self.assertAlmostEqual(x, 1.0, places=6)
        self.assertAlmostEqual(y, 1.6, places=6)

    def test_duplicate_tracks_merge_without_chaining_cones(self):
        groups = merge_groups([(0.0, 0.0), (0.18, 0.0),
                               (0.36, 0.0), (1.0, 0.0)], 0.20)
        self.assertEqual(sorted(len(group) for group in groups), [1, 1, 2])

    def test_separate_cones_stay_separate(self):
        groups = merge_groups([(0.0, 0.0), (0.26, 0.0)], 0.23)
        self.assertEqual(len(groups), 2)

    def test_heading_correction_is_zero_outside_blend(self):
        self.assertEqual(arrival_heading_correction(
            1.0, 0.5, 0.4, 0.8, 0.35), 0.0)

    def test_heading_correction_increases_near_goal(self):
        far = arrival_heading_correction(0.4, 0.30, 0.40, 0.8, 0.35)
        near = arrival_heading_correction(0.4, 0.10, 0.40, 0.8, 0.35)
        self.assertGreater(near, far)

    def test_heading_correction_is_limited(self):
        value = arrival_heading_correction(2.0, 0.0, 0.4, 1.0, 0.35)
        self.assertAlmostEqual(value, 0.35)


if __name__ == "__main__":
    unittest.main()
