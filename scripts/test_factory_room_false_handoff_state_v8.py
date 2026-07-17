#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import unittest
from unittest.mock import patch

import factory_room_continuous_scan_search_v2 as search_module


class FalseHandoffStateV8Test(unittest.TestCase):
    def make_node(self, route_index):
        node = search_module.RecoverableContinuousScanSearchV2.__new__(
            search_module.RecoverableContinuousScanSearchV2)
        node.integration_lock = threading.RLock()
        node.integration_phase = node.HANDOFF
        node.route_index = route_index
        node.route_points = [{"name": "p%d" % i} for i in range(9)]
        node.ocr_confirmed_non_targets = set()
        node.ocr_candidate_hold_until = 1.0
        node.ocr_candidate_label = "old"
        node.ocr_ignore_label = ""
        node.ocr_ignore_until = None
        node.ocr_non_target_ignore_s = 1.0
        node.finished = True
        node.search_completion_reported = False
        node.current_route_point = lambda: node.route_points[node.route_index]
        node._set_scan_safety_mode = lambda _enabled: None
        node._set_ocr = lambda _enabled: None
        node.publish_zero = lambda _reason: None
        node.statuses = []
        node.publish_integration_status = (
            lambda state, **extra: node.statuses.append((state, extra)))
        return node

    @staticmethod
    def fake_finish(node):
        if node.route_index < len(node.route_points) - 1:
            node.route_index += 1

    def run_resume(self, node):
        with patch.object(search_module.rospy.Time, "now",
                          return_value=10.0), \
                patch.object(search_module.rospy, "Duration",
                             side_effect=lambda value: value), \
                patch.object(
                    search_module.FixedPointContinuousOcrSweepRouteTestV2,
                    "_finish_scan", new=self.fake_finish):
            return node._resume_after_false_handoff("日用品加工车间")

    def test_nonfinal_false_handoff_advances_and_resumes(self):
        node = self.make_node(4)
        self.assertTrue(self.run_resume(node))
        self.assertEqual(node.route_index, 5)
        self.assertEqual(node.integration_phase, node.ACTIVE)
        self.assertFalse(node.finished)
        self.assertIn("日用品加工车间",
                      node.ocr_confirmed_non_targets)
        self.assertEqual(node.statuses[-1][0],
                         "SEARCH_RESUMED_AFTER_FALSE_HANDOFF")

    def test_final_false_handoff_completes_without_resume_motion(self):
        node = self.make_node(8)
        self.assertTrue(self.run_resume(node))
        self.assertEqual(node.route_index, 8)
        self.assertEqual(node.integration_phase, node.HOLD)
        self.assertTrue(node.search_completion_reported)
        self.assertEqual(node.statuses[-1][0], "SEARCH_COMPLETE")


if __name__ == "__main__":
    unittest.main()

