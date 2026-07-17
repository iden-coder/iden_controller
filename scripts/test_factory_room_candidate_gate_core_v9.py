#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_candidate_gate_core_v9 import candidate_gate_decision


class CandidateGateCoreV9Test(unittest.TestCase):
    def test_paired_candidate_cannot_skip_approach(self):
        self.assertEqual(
            candidate_gate_decision(
                "to_approach", 0.42, 0.01, 0.0, 0.14, 0.055, 0.0),
            "keep_navigating")

    def test_paired_candidate_advances_after_approach(self):
        self.assertEqual(
            candidate_gate_decision(
                "to_approach", 0.13, 0.40, 0.30, 0.14, 0.055, 0.0),
            "advance_to_candidate")

    def test_logged_d2_near_miss_releases_with_relaxed_approach(self):
        self.assertEqual(
            candidate_gate_decision(
                "to_approach", 0.148, 0.30, 0.45,
                0.20, 0.10, 0.0),
            "advance_to_candidate")

    def test_final_leg_requires_precise_candidate_arrival(self):
        self.assertEqual(
            candidate_gate_decision(
                "to_candidate", None, 0.06, 0.50, 0.14, 0.055, 0.0),
            "keep_navigating")
        self.assertEqual(
            candidate_gate_decision(
                "to_candidate", None, 0.05, 0.50, 0.14, 0.055, 0.0),
            "candidate_reached")

    def test_direct_candidate_requires_real_motion(self):
        self.assertEqual(
            candidate_gate_decision(
                "to_candidate", None, 0.02, 0.01, 0.14, 0.055, 0.03),
            "keep_navigating")
        self.assertEqual(
            candidate_gate_decision(
                "to_candidate", None, 0.02, 0.04, 0.14, 0.055, 0.03),
            "candidate_reached")


if __name__ == "__main__":
    unittest.main()
