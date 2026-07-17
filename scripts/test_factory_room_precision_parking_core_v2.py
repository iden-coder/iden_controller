#!/usr/bin/env python3

import unittest

from factory_room_precision_parking_core_v2 import longitudinal_decision


class PrecisionParkingDecisionTest(unittest.TestCase):
    def decide(self, error, settled=False):
        return longitudinal_decision(
            error, tolerance=0.008, stop_lead=0.025, settled=settled,
            near_band=0.080, near_min_speed=0.018,
            near_max_speed=0.035, far_full_speed_error=0.120,
            far_max_speed=0.150, reverse_speed=0.020)

    def test_far_approach_keeps_full_speed(self):
        self.assertEqual(("far", 0.150), self.decide(0.150))

    def test_near_target_is_slow(self):
        phase, speed = self.decide(0.028)
        self.assertEqual("near", phase)
        self.assertLessEqual(speed, 0.025)

    def test_stop_and_settle_before_crawl(self):
        self.assertEqual(("settle", 0.0), self.decide(0.019))
        self.assertEqual(("crawl", 0.018), self.decide(0.015, settled=True))

    def test_target_band_holds_zero(self):
        self.assertEqual(("hold", 0.0), self.decide(0.006, settled=True))
        self.assertEqual(("hold", 0.0), self.decide(-0.008, settled=True))

    def test_small_overshoot_reverses(self):
        phase, speed = self.decide(-0.013, settled=True)
        self.assertEqual("reverse", phase)
        self.assertEqual(-0.020, speed)


if __name__ == "__main__":
    unittest.main()
