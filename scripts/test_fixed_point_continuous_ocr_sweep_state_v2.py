#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import unittest
from unittest.mock import Mock, patch

import fixed_point_continuous_ocr_sweep_route_test_v2 as controller_module
from fixed_point_continuous_ocr_sweep_route_test_v2 import (
    FixedPointContinuousOcrSweepRouteTestV2,
)


class ContinuousSweepStateV2Test(unittest.TestCase):
    @staticmethod
    def make_node(route_index):
        node = object.__new__(FixedPointContinuousOcrSweepRouteTestV2)
        node.route_points = [
            {"name": "d1", "x": 0.0, "y": 0.0},
            {"name": "d2_approach", "x": 1.0, "y": 0.0},
            {"name": "d2", "x": 1.4, "y": 0.0},
            {"name": "d3", "x": 2.0, "y": 0.0},
        ]
        node.route_index = route_index
        node.scan_specs = {"d1": {}, "d2": {}, "d3": {}}
        node.scan_progress = math.radians(90.0)
        node.scan_active = True
        node.scan_phase = "sweep"
        node.scan_command_wz = 0.2
        node.scanned_route_indexes = set()
        node.finished = False
        node.route_transition_observe_s = 2.40
        node.route_transition_hold_until = None
        node.scan_safety_mode_active = False
        node.scan_safety_mode_pub = None
        node.publish_zero = Mock()
        node._set_ocr = Mock()
        node.log_status = Mock()
        node._activate_route_point = Mock()
        node._reset_approach_alignment = Mock()
        node._reset_translation_progress = Mock()
        node.plan_from_current_pose = Mock(return_value=True)
        return node

    def test_completed_scan_advances_without_reacquiring_old_goal(self):
        node = self.make_node(0)
        with patch.object(controller_module.rospy.Time, "now",
                          return_value=controller_module.rospy.Time(10)):
            node._finish_scan()
        self.assertEqual(node.route_index, 1)
        self.assertIn(0, node.scanned_route_indexes)
        node._activate_route_point.assert_called_once_with(clear_path=True)
        node.plan_from_current_pose.assert_called_once_with(
            "atomic advance after continuous scan", force=True)
        self.assertIsNotNone(node.route_transition_hold_until)
        self.assertAlmostEqual(
            node.route_transition_hold_until.to_sec(), 12.40, places=5)

    def test_last_scan_finishes_and_does_not_plan_again(self):
        node = self.make_node(3)
        node._finish_scan()
        self.assertTrue(node.finished)
        self.assertEqual(node.route_index, 3)
        node.plan_from_current_pose.assert_not_called()
        self.assertIsNone(node.route_transition_hold_until)


if __name__ == "__main__":
    unittest.main()
