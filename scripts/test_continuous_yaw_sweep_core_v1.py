#!/usr/bin/env python3

import math
import unittest

from continuous_yaw_sweep_core_v1 import (
    braking_limited_speed,
    directed_yaw_increment,
    slew_rate,
    wrap_angle,
)


class ContinuousYawSweepCoreTest(unittest.TestCase):
    def test_counterclockwise_progress_crosses_positive_pi(self):
        progress = directed_yaw_increment(
            math.radians(179.0), math.radians(-178.0), 1)
        self.assertAlmostEqual(progress, math.radians(3.0), places=6)

    def test_clockwise_progress_crosses_negative_pi(self):
        progress = directed_yaw_increment(
            math.radians(-179.0), math.radians(178.0), -1)
        self.assertAlmostEqual(progress, math.radians(3.0), places=6)

    def test_opposite_motion_is_not_counted(self):
        self.assertEqual(directed_yaw_increment(0.0, -0.1, 1), 0.0)

    def test_speed_brakes_near_endpoint(self):
        far = braking_limited_speed(1.0, 0.16, 0.28, 0.05, 0.01)
        near = braking_limited_speed(0.02, 0.16, 0.28, 0.05, 0.01)
        stopped = braking_limited_speed(0.005, 0.16, 0.28, 0.05, 0.01)
        self.assertAlmostEqual(far, 0.16)
        self.assertLess(near, far)
        self.assertEqual(stopped, 0.0)

    def test_slew_rate_limits_acceleration(self):
        self.assertAlmostEqual(slew_rate(0.0, 0.16, 0.28, 0.1), 0.028)
        self.assertAlmostEqual(slew_rate(0.10, 0.0, 0.28, 0.1), 0.072)

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(math.radians(450.0)), math.pi / 2.0)


if __name__ == "__main__":
    unittest.main()

