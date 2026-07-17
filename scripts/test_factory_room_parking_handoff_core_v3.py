#!/usr/bin/env python3

import unittest

from factory_room_parking_handoff_core_v3 import (
    is_fresh_target_payload,
    precenter_action,
)


class ParkingHandoffCoreTest(unittest.TestCase):
    def payload(self, stamp=11.0, label="target"):
        return {
            "stable": True,
            "label": label,
            "stamp": stamp,
            "bbox": [200, 50, 600, 170],
            "image_width": 800,
        }

    def test_stale_ocr_is_rejected(self):
        self.assertFalse(is_fresh_target_payload(
            self.payload(stamp=9.9), "target", 10.0))

    def test_wrong_label_is_rejected(self):
        self.assertFalse(is_fresh_target_payload(
            self.payload(label="other"), "target", 10.0))

    def test_fresh_target_is_accepted(self):
        self.assertTrue(is_fresh_target_payload(
            self.payload(), "target", 10.0))

    def test_centered_sign_skips_clearance_motion(self):
        self.assertEqual(
            "centered", precenter_action(8, 10, 0.02, 0.075, 0.045, True))

    def test_blocked_lateral_uses_bounded_forward_clearance(self):
        self.assertEqual(
            "advance", precenter_action(40, 10, 0.065, 0.075, 0.045, True))
        self.assertEqual(
            "reobserve", precenter_action(40, 10, 0.03, 0.075, 0.045, True))


if __name__ == "__main__":
    unittest.main()
