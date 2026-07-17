#!/usr/bin/env python3

import math
import unittest

import numpy as np

from factory_room_map_coverage_core_v1 import (
    OccupancyMap,
    build_bottom_right_anchor,
    build_coverage_model,
    choose_parking_wall_motion,
    estimate_wall_tangent_hint,
    evaluate_target_observation,
    target_center_gate_px,
    update_entry_crossing_stability,
)


def make_shifted_room(left, right, bottom, top, seed):
    resolution = 0.03
    origin_x = -5.0
    origin_y = -5.0
    width = 400
    height = 250
    data = np.zeros((height, width), dtype=np.int16)

    def cell(x, y):
        return (int((x - origin_x) / resolution),
                int((y - origin_y) / resolution))

    left_col, bottom_row = cell(left, bottom)
    right_col, top_row = cell(right, top)
    data[bottom_row:top_row + 1, left_col:left_col + 2] = 100
    data[bottom_row:top_row + 1, right_col:right_col + 2] = 100
    data[bottom_row:bottom_row + 2, left_col:right_col + 1] = 100
    data[top_row:top_row + 2, left_col:right_col + 1] = 100

    # A door opening and a short interior obstacle must not become room walls.
    door_center, _ = cell(seed[0] - 0.8, top)
    data[top_row:top_row + 2, door_center - 7:door_center + 7] = 0
    cone_col, cone_row = cell(seed[0] + 0.55, seed[1] - 0.15)
    data[cone_row - 2:cone_row + 3, cone_col - 2:cone_col + 3] = 100
    return OccupancyMap(
        width, height, resolution, origin_x, origin_y, data.ravel())


class MapCoverageCoreTest(unittest.TestCase):
    def test_final_parking_recovers_small_distance_overshoot(self):
        mode, speed = choose_parking_wall_motion(
            0.150, 0.171, 0.006, 0.15, 0.024, 0.10, 0.035)
        self.assertEqual(mode, "reverse")
        self.assertAlmostEqual(speed, -0.035)

        mode, speed = choose_parking_wall_motion(
            0.171, 0.171, 0.006, 0.15, 0.024, 0.10, 0.035)
        self.assertEqual(mode, "hold")
        self.assertEqual(speed, 0.0)

        mode, speed = choose_parking_wall_motion(
            0.260, 0.171, 0.006, 0.15, 0.024, 0.10, 0.035)
        self.assertEqual(mode, "forward")
        self.assertGreater(speed, 0.024)

    def test_room_entry_gate_requires_stable_line_crossing(self):
        count = 0
        for pose_y in (-1.70, -1.78, -1.79):
            count, ready = update_entry_crossing_stability(
                count, pose_y, -1.75, 3)
        self.assertFalse(ready)
        self.assertEqual(count, 2)

        count, ready = update_entry_crossing_stability(
            count, -1.81, -1.75, 3)
        self.assertTrue(ready)

        count, ready = update_entry_crossing_stability(
            count, -1.72, -1.75, 3)
        self.assertFalse(ready)
        self.assertEqual(count, 0)

    def check_room(self, expected, seed):
        grid = make_shifted_room(*expected, seed)
        model = build_coverage_model(
            grid, seed_x=seed[0], seed_y=seed[1],
            search_radius_x=3.4, search_radius_y=1.8,
            standoff=0.72, candidate_spacing=0.72,
            bin_spacing=0.24, half_fov_deg=35.0,
            corner_trim=0.30, min_candidate_clearance=0.24)
        bounds = model["bounds"]
        for key, value in zip(
                ("left", "right", "bottom", "top"), expected):
            self.assertAlmostEqual(bounds[key], value, delta=0.12)
        covered = set()
        for candidate in model["candidates"]:
            self.assertGreaterEqual(candidate["clearance"], 0.24)
            covered.update(candidate["visible_bins"])
        self.assertEqual(covered, set(model["bins"]))
        self.assertEqual(
            {candidate["wall"] for candidate in model["candidates"]},
            {"left", "top", "right", "bottom"})

    def test_nominal_room_with_doorway(self):
        self.check_room((-2.2, 2.9, -3.35, -1.28), (0.35, -2.31))

    def test_similar_map_shifted_twelve_centimeters(self):
        self.check_room((-2.08, 3.02, -3.23, -1.16), (0.47, -2.19))

    def test_peripheral_target_becomes_reposition_hint(self):
        self.assertAlmostEqual(target_center_gate_px(800, 0.18), 144.0)
        left_hint = estimate_wall_tangent_hint(
            "left", -2.18, -352.5, 800, 0.72, 35.0,
            -3.31, -1.29, 0.30)
        right_hint = estimate_wall_tangent_hint(
            "right", -2.18, -352.5, 800, 0.72, 35.0,
            -3.31, -1.29, 0.30)
        self.assertLess(left_hint, -2.18)
        self.assertGreater(right_hint, -2.18)
        self.assertGreaterEqual(left_hint, -3.01)
        self.assertLessEqual(right_hint, -1.59)

    def test_log_replay_uses_best_complete_target_view(self):
        edge_view = evaluate_target_observation(
            [5.0, 69.0, 195.0, 163.0], 800, -300.0,
            0.62, 35.0)
        best_view = evaluate_target_observation(
            [64.0, 42.0, 387.0, 152.0], 800, -174.0,
            0.62, 35.0)
        reversed_view = evaluate_target_observation(
            [0.0, 40.0, 308.0, 160.0], 800, -246.0,
            0.62, 35.0)
        self.assertFalse(edge_view["accepted"])
        self.assertTrue(edge_view["clipped"])
        self.assertTrue(best_view["accepted"])
        self.assertFalse(best_view["clipped"])
        self.assertLess(best_view["lateral_shift"], 0.20)
        self.assertFalse(reversed_view["accepted"])

    def test_map_derived_d3_anchor_faces_bottom_wall(self):
        grid = make_shifted_room(
            -2.2, 2.9, -3.35, -1.28, (0.35, -2.31))
        model = build_coverage_model(
            grid, seed_x=0.35, seed_y=-2.31,
            search_radius_x=3.4, search_radius_y=1.8)
        anchor = build_bottom_right_anchor(grid, model)
        self.assertAlmostEqual(anchor["x"], 2.48, delta=0.12)
        self.assertAlmostEqual(anchor["y"], -2.55, delta=0.12)
        self.assertAlmostEqual(anchor["yaw"], -math.pi / 2.0)
        self.assertGreaterEqual(anchor["clearance"], 0.24)
        self.assertTrue(anchor["visible_bins"])


if __name__ == "__main__":
    unittest.main()
