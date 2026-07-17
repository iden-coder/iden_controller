#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_edge_recenter_core_v10 import (
    edge_target_recenter_decision,
)


class EdgeTargetRecenterCoreV10Test(unittest.TestCase):
    def decide(self, bbox):
        return edge_target_recenter_decision(
            bbox, 800, 12, 55, 0.28, 0.08, 0.20)

    def test_left_edge_turns_left_toward_image_center(self):
        state, command = self.decide([0, 87, 154, 172])
        self.assertEqual(state, "recenter")
        self.assertGreater(command, 0.0)
        self.assertLessEqual(command, 0.20)

    def test_right_edge_turns_right_toward_image_center(self):
        state, command = self.decide([700, 87, 800, 172])
        self.assertEqual(state, "recenter")
        self.assertLess(command, 0.0)

    def test_centered_target_is_usable(self):
        self.assertEqual(self.decide([250, 80, 550, 170]),
                         ("usable", 0.0))

    def test_tiny_partial_target_is_not_chased(self):
        self.assertEqual(self.decide([0, 80, 40, 170]),
                         ("partial", 0.0))


if __name__ == "__main__":
    unittest.main()

