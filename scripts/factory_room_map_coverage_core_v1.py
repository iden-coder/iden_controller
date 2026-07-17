#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Map-driven wall coverage geometry for the factory room.

The module deliberately has no ROS dependency so the geometry can be tested
against a saved occupancy map before it is used by the real robot.
"""

import math

import numpy as np


class CoverageMapError(RuntimeError):
    pass


def update_entry_crossing_stability(current_count, pose_y, trigger_y,
                                    required_samples, less_than=True):
    """Update a debounced map-Y room-entry gate."""
    required = max(1, int(required_samples))
    if pose_y is None:
        return 0, False
    crossed = (pose_y <= trigger_y if less_than
               else pose_y >= trigger_y)
    count = int(current_count) + 1 if crossed else 0
    return count, count >= required


def choose_parking_wall_motion(wall_distance, target_distance, tolerance,
                               slow_error, slow_speed, fast_speed,
                               reverse_speed):
    """Choose a signed wall-normal speed for final parking.

    A small overshoot is recoverable by backing straight away from the wall;
    it must not turn into a full workshop-search retry.
    """
    error = float(wall_distance) - float(target_distance)
    tolerance = max(0.0, float(tolerance))
    if abs(error) <= tolerance:
        return "hold", 0.0
    if error < 0.0:
        return "reverse", -abs(float(reverse_speed))
    span = max(tolerance, float(slow_error), 1e-6)
    ratio = min(1.0, max(0.0, error / span))
    speed = float(slow_speed) + ratio * (
        float(fast_speed) - float(slow_speed))
    return "forward", max(0.0, speed)


class OccupancyMap(object):
    def __init__(self, width, height, resolution, origin_x, origin_y, data):
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        values = np.asarray(data, dtype=np.int16)
        if values.size != self.width * self.height:
            raise CoverageMapError("occupancy data size does not match map")
        self.values = values.reshape((self.height, self.width))
        self.blocked = np.logical_or(self.values < 0, self.values >= 50)

    def grid_to_world(self, col, row):
        return (
            self.origin_x + (float(col) + 0.5) * self.resolution,
            self.origin_y + (float(row) + 0.5) * self.resolution,
        )

    def world_to_grid(self, x, y):
        col = int(math.floor((float(x) - self.origin_x) / self.resolution))
        row = int(math.floor((float(y) - self.origin_y) / self.resolution))
        return col, row

    def inside(self, col, row):
        return 0 <= col < self.width and 0 <= row < self.height

    def is_blocked_world(self, x, y):
        col, row = self.world_to_grid(x, y)
        return not self.inside(col, row) or bool(self.blocked[row, col])

    def clearance(self, x, y, max_radius=1.0):
        col, row = self.world_to_grid(x, y)
        if not self.inside(col, row) or self.blocked[row, col]:
            return 0.0
        cells = max(1, int(math.ceil(max_radius / self.resolution)))
        c0 = max(0, col - cells)
        c1 = min(self.width, col + cells + 1)
        r0 = max(0, row - cells)
        r1 = min(self.height, row + cells + 1)
        local = self.blocked[r0:r1, c0:c1]
        rows, cols = np.nonzero(local)
        if not len(rows):
            return float(max_radius)
        dx = (cols + c0 - col) * self.resolution
        dy = (rows + r0 - row) * self.resolution
        return float(np.sqrt(dx * dx + dy * dy).min())

    def line_is_free(self, x0, y0, x1, y1, terminal_margin=0.20):
        length = math.hypot(x1 - x0, y1 - y0)
        usable = max(0.0, length - terminal_margin)
        steps = max(2, int(math.ceil(usable / (0.5 * self.resolution))))
        for index in range(steps + 1):
            distance = usable * float(index) / float(steps)
            ratio = 0.0 if length <= 1e-6 else distance / length
            if self.is_blocked_world(
                    x0 + ratio * (x1 - x0),
                    y0 + ratio * (y1 - y0)):
                return False
        return True


def target_center_gate_px(image_width, gate_ratio=0.18, minimum_px=60.0):
    return max(float(minimum_px), float(gate_ratio) * float(image_width))


def evaluate_target_observation(bbox, image_width, pixel_error,
                                wall_distance, half_fov_deg,
                                gate_ratio=0.18, edge_margin_px=20.0,
                                max_lateral_shift=0.28):
    width = float(image_width)
    if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
            width <= 1.0):
        return {
            "accepted": False, "clipped": True,
            "lateral_shift": float("inf"), "gate_px": 0.0,
        }
    x0 = float(bbox[0])
    x1 = float(bbox[2])
    clipped = (
        x0 <= float(edge_margin_px) or
        x1 >= width - float(edge_margin_px))
    half_fov = math.radians(max(15.0, float(half_fov_deg)))
    focal_px = width / (2.0 * math.tan(half_fov))
    lateral_shift = (
        abs(float(pixel_error)) / max(1.0, focal_px) *
        max(0.0, float(wall_distance)))
    gate_px = target_center_gate_px(width, gate_ratio)
    accepted = (
        not clipped and lateral_shift <= float(max_lateral_shift))
    return {
        "accepted": accepted,
        "clipped": clipped,
        "lateral_shift": lateral_shift,
        "gate_px": gate_px,
        "focal_px": focal_px,
    }


def estimate_wall_tangent_hint(wall_name, current_tangent, pixel_error,
                               image_width, standoff, half_fov_deg,
                               tangent_start, tangent_end, corner_trim):
    half_fov = math.radians(max(15.0, float(half_fov_deg)))
    focal_px = float(image_width) / (2.0 * math.tan(half_fov))
    bearing = math.atan2(float(pixel_error), max(1.0, focal_px))
    tangent_sign = {
        "left": 1.0,
        "right": -1.0,
        "top": 1.0,
        "bottom": -1.0,
    }[wall_name]
    tangent = (
        float(current_tangent) + tangent_sign * float(standoff) *
        math.tan(bearing))
    return max(float(tangent_start) + float(corner_trim),
               min(float(tangent_end) - float(corner_trim), tangent))


def _smoothed_projection(values, half_window=2):
    values = np.asarray(values, dtype=np.float64)
    width = 2 * int(half_window) + 1
    if values.size < width:
        return values
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(values, kernel, mode="same")


def _refine_peak(grid, scores, peak_index):
    peak_score = float(scores[peak_index])
    radius = max(1, int(round(0.10 / grid.resolution)))
    nearby = np.arange(
        max(0, peak_index - radius), min(len(scores), peak_index + radius + 1))
    strong = nearby[scores[nearby] >= 0.68 * peak_score]
    if not strong.size:
        strong = np.asarray([peak_index])
    weights = np.maximum(scores[strong], 1e-6)
    return float(np.sum((strong + 0.5) * weights) / np.sum(weights))


def _local_peaks(scores, indexes):
    indexes = np.asarray(indexes, dtype=np.int32)
    if not indexes.size:
        raise CoverageMapError("empty wall search interval")
    local_max = float(np.max(scores[indexes]))
    threshold = max(8.0, 0.35 * local_max)
    peaks = []
    for index in indexes:
        if index <= 0 or index >= len(scores) - 1:
            continue
        if (scores[index] >= threshold and
                scores[index] >= scores[index - 1] and
                scores[index] >= scores[index + 1]):
            if peaks and index - peaks[-1] <= 2:
                if scores[index] > scores[peaks[-1]]:
                    peaks[-1] = int(index)
            else:
                peaks.append(int(index))
    if not peaks:
        peaks = [int(indexes[int(np.argmax(scores[indexes]))])]
    return peaks


def _balanced_wall_pair(scores, lower_indexes, upper_indexes, seed_index):
    lower = _local_peaks(scores, lower_indexes)
    upper = _local_peaks(scores, upper_indexes)
    maximum = max(1.0, float(np.max(scores)))
    best = None
    for first in lower:
        for second in upper:
            first_distance = float(seed_index - first)
            second_distance = float(second - seed_index)
            total_distance = max(1.0, first_distance + second_distance)
            symmetry = abs(first_distance - second_distance) / total_distance
            strength = (scores[first] + scores[second]) / (2.0 * maximum)
            # A rectangular room normally places its seed near the middle.
            # Strength still matters, so short interior map strokes do not win.
            score = strength - 1.25 * symmetry
            if best is None or score > best[0]:
                best = (score, first, second)
    if best is None:
        raise CoverageMapError("could not pair opposite room walls")
    return best[1], best[2]


def detect_room_walls(grid, seed_x=0.40, seed_y=-2.30,
                      search_radius_x=3.50, search_radius_y=2.00,
                      min_half_extent=0.65):
    seed_col, seed_row = grid.world_to_grid(seed_x, seed_y)
    if not grid.inside(seed_col, seed_row):
        raise CoverageMapError("room seed is outside occupancy map")

    c0 = max(0, int(seed_col - search_radius_x / grid.resolution))
    c1 = min(grid.width, int(seed_col + search_radius_x / grid.resolution) + 1)
    r0 = max(0, int(seed_row - search_radius_y / grid.resolution))
    r1 = min(grid.height, int(seed_row + search_radius_y / grid.resolution) + 1)
    roi = grid.blocked[r0:r1, c0:c1]
    vertical_scores = _smoothed_projection(roi.sum(axis=0))
    horizontal_scores = _smoothed_projection(roi.sum(axis=1))

    min_x_cells = int(round(min_half_extent / grid.resolution))
    min_y_cells = int(round(min_half_extent / grid.resolution))
    local_seed_col = seed_col - c0
    local_seed_row = seed_row - r0
    left_indexes = np.arange(0, max(0, local_seed_col - min_x_cells))
    right_indexes = np.arange(
        min(len(vertical_scores), local_seed_col + min_x_cells),
        len(vertical_scores))
    bottom_indexes = np.arange(0, max(0, local_seed_row - min_y_cells))
    top_indexes = np.arange(
        min(len(horizontal_scores), local_seed_row + min_y_cells),
        len(horizontal_scores))

    left_peak, right_peak = _balanced_wall_pair(
        vertical_scores, left_indexes, right_indexes, local_seed_col)
    bottom_peak, top_peak = _balanced_wall_pair(
        horizontal_scores, bottom_indexes, top_indexes, local_seed_row)
    left_grid = c0 + _refine_peak(grid, vertical_scores, left_peak)
    right_grid = c0 + _refine_peak(grid, vertical_scores, right_peak)
    bottom_grid = r0 + _refine_peak(grid, horizontal_scores, bottom_peak)
    top_grid = r0 + _refine_peak(grid, horizontal_scores, top_peak)
    left = grid.origin_x + left_grid * grid.resolution
    right = grid.origin_x + right_grid * grid.resolution
    bottom = grid.origin_y + bottom_grid * grid.resolution
    top = grid.origin_y + top_grid * grid.resolution

    width = right - left
    height = top - bottom
    if not (1.6 <= width <= 7.5 and 1.2 <= height <= 4.5):
        raise CoverageMapError(
            "implausible room bounds width=%.2f height=%.2f" %
            (width, height))
    if not (left < seed_x < right and bottom < seed_y < top):
        raise CoverageMapError("detected walls do not contain room seed")
    return {
        "left": left,
        "right": right,
        "bottom": bottom,
        "top": top,
        "width": width,
        "height": height,
    }


def _wall_description(name, bounds):
    if name == "left":
        return {
            "fixed": bounds["left"], "start": bounds["bottom"],
            "end": bounds["top"], "axis": "y", "normal": (1.0, 0.0),
            "yaw": math.pi,
        }
    if name == "right":
        return {
            "fixed": bounds["right"], "start": bounds["bottom"],
            "end": bounds["top"], "axis": "y", "normal": (-1.0, 0.0),
            "yaw": 0.0,
        }
    if name == "top":
        return {
            "fixed": bounds["top"], "start": bounds["left"],
            "end": bounds["right"], "axis": "x", "normal": (0.0, -1.0),
            "yaw": math.pi / 2.0,
        }
    if name == "bottom":
        return {
            "fixed": bounds["bottom"], "start": bounds["left"],
            "end": bounds["right"], "axis": "x", "normal": (0.0, 1.0),
            "yaw": -math.pi / 2.0,
        }
    raise CoverageMapError("unknown wall %s" % name)


def _wall_point(wall, tangent, inward_distance=0.0):
    nx, ny = wall["normal"]
    if wall["axis"] == "x":
        return tangent + nx * inward_distance, wall["fixed"] + ny * inward_distance
    return wall["fixed"] + nx * inward_distance, tangent + ny * inward_distance


def _wall_present(grid, wall, tangent, probe_depth=0.09):
    wx, wy = _wall_point(wall, tangent)
    nx, ny = wall["normal"]
    samples = max(2, int(math.ceil(2.0 * probe_depth / grid.resolution)))
    for index in range(samples + 1):
        offset = -probe_depth + 2.0 * probe_depth * index / float(samples)
        if grid.is_blocked_world(wx + nx * offset, wy + ny * offset):
            return True
    return False


def build_wall_coverage(grid, bounds, standoff=0.72,
                        candidate_spacing=0.72, bin_spacing=0.24,
                        half_fov_deg=35.0, corner_trim=0.30,
                        min_candidate_clearance=0.24):
    walls = {}
    bins = {}
    candidates = []
    candidate_index = 0
    for wall_name in ("left", "top", "right", "bottom"):
        wall = _wall_description(wall_name, bounds)
        walls[wall_name] = wall
        usable_start = wall["start"] + corner_trim
        usable_end = wall["end"] - corner_trim
        if usable_end <= usable_start:
            continue

        bin_count = max(1, int(math.ceil(
            (usable_end - usable_start) / float(bin_spacing))))
        wall_bins = []
        for index in range(bin_count):
            tangent = usable_start + (
                index + 0.5) * (usable_end - usable_start) / bin_count
            if not _wall_present(grid, wall, tangent):
                continue
            x, y = _wall_point(wall, tangent)
            bin_id = "%s_b%02d" % (wall_name, index + 1)
            entry = {
                "id": bin_id, "wall": wall_name, "tangent": tangent,
                "x": x, "y": y,
            }
            wall_bins.append(entry)
            bins[bin_id] = entry

        if not wall_bins:
            continue
        # Keep a dense pool of possible poses. The runtime selector still
        # needs only a sparse set, but a dense pool lets it route around a
        # cone or a map detail without leaving a blind strip on that wall.
        tangent_values = [entry["tangent"] for entry in wall_bins]
        half_span = standoff * math.tan(math.radians(half_fov_deg))
        for tangent in tangent_values:
            x, y = _wall_point(wall, float(tangent), standoff)
            clearance = grid.clearance(x, y, max_radius=0.90)
            if clearance < min_candidate_clearance:
                continue
            visible = []
            for entry in wall_bins:
                if abs(entry["tangent"] - tangent) > half_span:
                    continue
                if grid.line_is_free(x, y, entry["x"], entry["y"]):
                    visible.append(entry["id"])
            if not visible:
                continue
            candidate_index += 1
            candidates.append({
                "id": "map_view_%02d" % candidate_index,
                "wall": wall_name,
                "x": x,
                "y": y,
                "yaw": wall["yaw"],
                "tangent": float(tangent),
                "clearance": clearance,
                "visible_bins": visible,
            })

    if not candidates:
        raise CoverageMapError("map produced no safe OCR observation views")
    covered = set()
    for candidate in candidates:
        covered.update(candidate["visible_bins"])
    uncovered = set(bins) - covered
    if uncovered:
        raise CoverageMapError(
            "%d wall bins have no safe camera view" % len(uncovered))
    return {
        "bounds": bounds,
        "walls": walls,
        "bins": bins,
        "candidates": candidates,
    }


def build_bottom_right_anchor(grid, coverage_model,
                              right_inset=0.42, bottom_inset=0.80,
                              half_fov_deg=35.0,
                              min_candidate_clearance=0.24):
    bounds = coverage_model["bounds"]
    wall = coverage_model["walls"]["bottom"]
    x = bounds["right"] - float(right_inset)
    y = bounds["bottom"] + float(bottom_inset)
    clearance = grid.clearance(x, y, max_radius=0.90)
    if clearance < float(min_candidate_clearance):
        raise CoverageMapError(
            "bottom-right anchor clearance %.3f is unsafe" % clearance)
    wall_distance = y - bounds["bottom"]
    half_span = wall_distance * math.tan(math.radians(half_fov_deg))
    visible = []
    for bin_id, entry in coverage_model["bins"].items():
        if entry["wall"] != "bottom":
            continue
        if abs(entry["tangent"] - x) > half_span:
            continue
        if grid.line_is_free(x, y, entry["x"], entry["y"]):
            visible.append(bin_id)
    if not visible:
        raise CoverageMapError(
            "bottom-right anchor has no visible bottom-wall bins")
    return {
        "id": "map_corner_d3",
        "wall": "bottom",
        "x": x,
        "y": y,
        "yaw": -math.pi / 2.0,
        "tangent": x,
        "clearance": clearance,
        "visible_bins": visible,
        "mandatory": True,
    }


def build_coverage_model(grid, **kwargs):
    detection_keys = {
        "seed_x", "seed_y", "search_radius_x", "search_radius_y",
        "min_half_extent",
    }
    detection_args = {
        key: value for key, value in kwargs.items() if key in detection_keys
    }
    coverage_args = {
        key: value for key, value in kwargs.items() if key not in detection_keys
    }
    bounds = detect_room_walls(grid, **detection_args)
    return build_wall_coverage(grid, bounds, **coverage_args)
