#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Aspect-ratio-safe detector for the 50 cm floor parking frame."""

import math

import cv2
import numpy as np


def clamp(value, low, high):
    return max(low, min(high, value))


class PerspectiveParkingDetector(object):
    """Detect two perspective side rails and the entrance crossbar.

    The camera driver can silently fall back from 1280x720 to 800x600.  This
    detector works in normalized raw-image coordinates and therefore never
    stretches a 4:3 image through a 16:9 calibration.
    """

    def __init__(self, max_width=800):
        self.max_width = int(max_width)
        self.filtered_lateral = None
        self.filtered_heading = None

    @staticmethod
    def _line_x_at_y(line, y_value):
        x1, y1, x2, y2 = line["line"]
        if abs(y2 - y1) < 1.0:
            return 0.5 * (x1 + x2)
        return x1 + (float(y_value) - y1) * (x2 - x1) / float(y2 - y1)

    @staticmethod
    def _empty():
        return {
            "found": False, "confidence": 0.0,
            "lateral_error_m": 0.0, "raw_lateral_error_m": 0.0,
            "heading_error_rad": 0.0, "near_edge_distance_m": None,
            "rail_width_m": None, "rail_center_px": None,
            "near_edge_y_px": None, "vertical_count": 0,
            "horizontal_count": 0, "bev": None, "mask": None,
            "debug": None,
        }

    def detect(self, bgr_image):
        result = self._empty()
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            return result

        source_h, source_w = bgr_image.shape[:2]
        scale = min(1.0, float(self.max_width) / max(float(source_w), 1.0))
        if scale < 0.999:
            frame = cv2.resize(
                bgr_image, (int(round(source_w * scale)),
                            int(round(source_h * scale))),
                interpolation=cv2.INTER_AREA)
        else:
            frame = bgr_image.copy()
        height, width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv, np.array([0, 0, 185], dtype=np.uint8),
            np.array([180, 85, 255], dtype=np.uint8))
        white[:int(0.11 * height), :] = 0
        white[int(0.72 * height):, :] = 0
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8),
            iterations=1)
        white = cv2.morphologyEx(
            white, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8),
            iterations=1)

        lines = cv2.HoughLinesP(
            white, 1.0, np.pi / 360.0, threshold=max(28, width // 24),
            minLineLength=max(55, int(0.10 * width)),
            maxLineGap=max(24, int(0.055 * width)))
        left_lines = []
        right_lines = []
        horizontal = []
        y_reference = 0.25 * height
        if lines is not None:
            for packed in lines:
                x1, y1, x2, y2 = [float(v) for v in packed[0]]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                angle = abs(math.degrees(math.atan2(dy, dx)))
                angle = min(angle, 180.0 - angle)
                item = {
                    "line": (int(x1), int(y1), int(x2), int(y2)),
                    "length": length,
                }
                if angle <= 13.0 and length >= 0.34 * width:
                    y_mid = 0.5 * (y1 + y2)
                    if 0.30 * height <= y_mid <= 0.66 * height:
                        item.update({
                            "y": y_mid, "x_lo": min(x1, x2),
                            "x_hi": max(x1, x2),
                        })
                        horizontal.append(item)
                    continue
                if not (24.0 <= angle <= 74.0 and
                        length >= 0.12 * width and abs(dy) >= 1.0):
                    continue
                slope = dx / dy
                x_ref = x1 + (y_reference - y1) * slope
                item.update({"slope": slope, "x_ref": x_ref})
                if slope < -0.22 and x_ref < 0.52 * width:
                    left_lines.append(item)
                elif slope > 0.22 and x_ref > 0.48 * width:
                    right_lines.append(item)

        y_far = 0.16 * height
        y_near = 0.35 * height
        candidates = []
        for left in left_lines:
            for right in right_lines:
                left_far = self._line_x_at_y(left, y_far)
                right_far = self._line_x_at_y(right, y_far)
                left_near = self._line_x_at_y(left, y_near)
                right_near = self._line_x_at_y(right, y_near)
                far_sep = right_far - left_far
                near_sep = right_near - left_near
                if not (0.30 * width <= far_sep <= 0.86 * width):
                    continue
                if not (0.55 * width <= near_sep <= 1.55 * width):
                    continue
                if near_sep < 1.10 * far_sep:
                    continue
                center_far = 0.5 * (left_far + right_far)
                center_near = 0.5 * (left_near + right_near)
                left_b = left_near - left["slope"] * y_near
                right_b = right_near - right["slope"] * y_near
                slope_delta = left["slope"] - right["slope"]
                if abs(slope_delta) < 0.10:
                    continue
                vanish_y = (right_b - left_b) / slope_delta
                vanish_x = left["slope"] * vanish_y + left_b
                if not (-0.45 * height <= vanish_y <= 0.28 * height and
                        0.20 * width <= vanish_x <= 0.80 * width):
                    continue
                crossing = []
                for bar in horizontal:
                    span = bar["x_hi"] - bar["x_lo"]
                    bar_center = 0.5 * (bar["x_lo"] + bar["x_hi"])
                    if (span >= 0.48 * width and
                            abs(bar_center - 0.5 * width) <= 0.20 * width):
                        crossing.append(bar)
                if not crossing:
                    continue
                center_score = max(
                    0.0, 1.0 - abs(center_near - 0.5 * width) /
                    (0.38 * width))
                far_center_score = max(
                    0.0, 1.0 - abs(center_far - 0.5 * width) /
                    (0.38 * width))
                perspective_score = clamp(
                    (near_sep / max(far_sep, 1.0) - 1.0) / 1.15, 0.0, 1.0)
                heading_score = max(
                    0.0, 1.0 - abs(vanish_x - 0.5 * width) /
                    (0.26 * width))
                line_score = clamp(
                    (left["length"] + right["length"]) /
                    (0.65 * width), 0.0, 1.0)
                cross_score = clamp(
                    max(item["length"] for item in crossing) /
                    (0.78 * width), 0.0, 1.0)
                score = (0.23 * center_score + 0.13 * far_center_score +
                         0.22 * heading_score + 0.12 * perspective_score +
                         0.15 * line_score + 0.15 * cross_score)
                candidates.append({
                    "left": left, "right": right, "bars": crossing,
                    "center_far": center_far, "center_near": center_near,
                    "vanish_x": vanish_x, "vanish_y": vanish_y,
                    "far_sep": far_sep, "near_sep": near_sep,
                    "score": score,
                })

        debug = frame.copy()
        for item in left_lines:
            cv2.line(debug, item["line"][:2], item["line"][2:],
                     (255, 160, 0), 2)
        for item in right_lines:
            cv2.line(debug, item["line"][:2], item["line"][2:],
                     (255, 0, 180), 2)
        for item in horizontal:
            cv2.line(debug, item["line"][:2], item["line"][2:],
                     (0, 220, 220), 2)
        result.update({
            "bev": frame, "mask": white, "debug": debug,
            "vertical_count": len(left_lines) + len(right_lines),
            "horizontal_count": len(horizontal),
            "left_count": len(left_lines), "right_count": len(right_lines),
            "source_width": source_w, "source_height": source_h,
        })
        if not candidates:
            return result

        best = max(candidates, key=lambda item: item["score"])
        near_bar = max(best["bars"], key=lambda item: item["y"])
        frame_center = 0.5 * width
        lateral = ((best["center_near"] - frame_center) /
                   max(best["near_sep"], 1.0) * 0.50)
        # Vanishing-point displacement represents heading; unlike comparing
        # two finite rail centers it is insensitive to lateral displacement.
        heading = math.atan2(
            best["vanish_x"] - frame_center, 0.78 * width)
        if self.filtered_lateral is None:
            self.filtered_lateral = lateral
            self.filtered_heading = heading
        else:
            alpha = 0.28
            self.filtered_lateral += alpha * (lateral - self.filtered_lateral)
            self.filtered_heading += alpha * (heading - self.filtered_heading)
        lateral_filtered = self.filtered_lateral
        heading_filtered = self.filtered_heading
        near_distance = clamp(
            0.62 * (1.0 - near_bar["y"] / float(height)), 0.06, 0.50)
        confidence = clamp(best["score"], 0.0, 1.0)
        result.update({
            "found": confidence >= 0.54,
            "confidence": confidence,
            "lateral_error_m": lateral_filtered,
            "raw_lateral_error_m": lateral_filtered,
            "heading_error_rad": heading_filtered,
            "lateral_unfiltered_m": lateral,
            "heading_unfiltered_rad": heading,
            "near_edge_distance_m": near_distance,
            "rail_width_m": 0.50,
            "rail_center_px": best["center_near"],
            "center_near_px": best["center_near"],
            "center_far_px": best["center_far"],
            "near_edge_y_px": near_bar["y"],
        })
        cv2.line(debug, best["left"]["line"][:2],
                 best["left"]["line"][2:], (0, 0, 255), 4)
        cv2.line(debug, best["right"]["line"][:2],
                 best["right"]["line"][2:], (0, 0, 255), 4)
        cv2.line(debug, near_bar["line"][:2], near_bar["line"][2:],
                 (0, 255, 0), 4)
        cv2.putText(
            debug, "frame conf=%.2f off=%.3fm head=%.1fdeg" %
            (confidence, lateral_filtered, math.degrees(heading_filtered)),
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2)
        return result
