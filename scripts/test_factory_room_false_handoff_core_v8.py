#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_false_handoff_core_v8 import (
    handoff_bbox_is_usable,
    parking_recheck_result,
    resume_transition,
)


class FalseHandoffCoreV8Test(unittest.TestCase):
    def test_rejects_logged_left_edge_false_target(self):
        self.assertFalse(handoff_bbox_is_usable(
            [0.0, 94.0, 135.0, 172.0], 800, 12, 55))

    def test_accepts_complete_target_inside_image(self):
        self.assertTrue(handoff_bbox_is_usable(
            [80.0, 94.0, 240.0, 172.0], 800, 12, 55))

    def test_rejects_tiny_or_invalid_box(self):
        self.assertFalse(handoff_bbox_is_usable(
            [80.0, 94.0, 120.0, 172.0], 800, 12, 55))
        self.assertFalse(handoff_bbox_is_usable(None, 800, 12, 55))

    def test_stable_fresh_other_workshop_requests_resume(self):
        payload = {
            "stable": True, "label": "日用品加工车间",
            "votes": 12, "stamp": 10.0,
        }
        self.assertEqual(parking_recheck_result(
            payload, "食品加工车间", 10.2, 9.0, 1.8, 8),
            ("non_target", "日用品加工车间"))

    def test_matching_target_is_not_rejected(self):
        payload = {
            "stable": True, "label": "食品加工车间",
            "votes": 8, "stamp": 10.0,
        }
        self.assertEqual(parking_recheck_result(
            payload, "食品加工车间", 10.2, 9.0, 1.8, 8),
            ("target", "食品加工车间"))

    def test_stale_or_low_vote_result_is_inconclusive(self):
        stale = {
            "stable": True, "label": "日用品加工车间",
            "votes": 12, "stamp": 5.0,
        }
        low_votes = dict(stale, votes=7, stamp=10.0)
        self.assertEqual(parking_recheck_result(
            stale, "食品加工车间", 10.2, 9.0, 1.8, 8)[0],
            "inconclusive")
        self.assertEqual(parking_recheck_result(
            low_votes, "食品加工车间", 10.2, 9.0, 1.8, 8)[0],
            "inconclusive")

    def test_resume_advances_or_completes_route(self):
        self.assertEqual(resume_transition(4, 9), "advance")
        self.assertEqual(resume_transition(8, 9), "complete")


if __name__ == "__main__":
    unittest.main()

