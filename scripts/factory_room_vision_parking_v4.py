#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Strict BEV parking-frame detector based on the working line follower."""

import math

import cv2
import numpy as np

from factory_room_vision_core import GroundSquareDetector, clamp


class CenterlineParkingDetector(GroundSquareDetector):
    """Require two side rails and a crossbar in bird's-eye view.

    The parent detector's BEV calibration is copied from ucar_followline.  Its
    permissive raw-image quadrilateral shortcut is intentionally disabled: a
    wall sign or a short bright contour must never authorize vehicle motion.
    """

    def detect(self, bgr_image):
        empty = {
            "found": False, "confidence": 0.0,
            "lateral_error_m": 0.0, "near_edge_distance_m": None,
            "rail_width_m": None, "rail_center_px": None,
            "near_edge_y_px": None, "vertical_count": 0,
            "horizontal_count": 0, "bev": None, "mask": None,
            "debug": None,
        }
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            return empty

        frame = cv2.resize(bgr_image, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        bev = cv2.warpPerspective(
            frame, self.homography, (self.width, self.height),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv, np.array([0, 0, 150], dtype=np.uint8),
            np.array([180, 115, 255], dtype=np.uint8))
        border = max(5, int(self.width * 0.025))
        white[:, :border] = 0
        white[:, self.width - border:] = 0
        white[:int(self.height * 0.24), :] = 0
        white[self.height - 4:, :] = 0
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8),
            iterations=1)
        white = cv2.morphologyEx(
            white, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8),
            iterations=1)

        lines = cv2.HoughLinesP(
            white, 1.0, np.pi / 180.0, threshold=32,
            minLineLength=42, maxLineGap=32)
        vertical = []
        horizontal = []
        if lines is not None:
            for packed in lines:
                x1, y1, x2, y2 = [int(v) for v in packed[0]]
                dx, dy = x2 - x1, y2 - y1
                length = math.hypot(dx, dy)
                angle = abs(math.degrees(math.atan2(dy, dx)))
                angle = min(angle, 180.0 - angle)
                if angle >= 64.0 and length >= 50.0:
                    y_lo, y_hi = sorted((y1, y2))
                    if y_hi >= self.height * 0.40:
                        vertical.append({
                            "x": 0.5 * (x1 + x2), "y_lo": y_lo,
                            "y_hi": y_hi, "length": length,
                            "line": (x1, y1, x2, y2),
                        })
                elif angle <= 20.0 and length >= 75.0:
                    x_lo, x_hi = sorted((x1, x2))
                    horizontal.append({
                        "y": 0.5 * (y1 + y2), "x_lo": x_lo,
                        "x_hi": x_hi, "length": length,
                        "line": (x1, y1, x2, y2),
                    })

        min_width = self.pixels_per_meter_x * 0.30
        max_width = self.pixels_per_meter_x * 0.68
        expected_width = self.pixels_per_meter_x * 0.50
        candidates = []
        for left in vertical:
            for right in vertical:
                separation = right["x"] - left["x"]
                if separation < min_width or separation > max_width:
                    continue
                overlap_lo = max(left["y_lo"], right["y_lo"])
                overlap_hi = min(left["y_hi"], right["y_hi"])
                overlap = overlap_hi - overlap_lo
                if overlap < 40.0:
                    continue
                center = 0.5 * (left["x"] + right["x"])
                crossing = []
                for bar in horizontal:
                    covered = max(
                        0.0, min(bar["x_hi"], right["x"]) -
                        max(bar["x_lo"], left["x"]))
                    bar_center = 0.5 * (bar["x_lo"] + bar["x_hi"])
                    if (covered >= 0.35 * separation and
                            abs(bar_center - center) <= 0.18 * separation and
                            overlap_lo - 28.0 <= bar["y"] <= overlap_hi + 120.0):
                        crossing.append(bar)
                if not crossing:
                    continue
                width_score = max(
                    0.0, 1.0 - abs(separation - expected_width) /
                    max(0.42 * expected_width, 1.0))
                center_score = max(
                    0.0, 1.0 - abs(center - self.width * 0.5) /
                    (self.width * 0.60))
                rail_score = min(1.0, overlap / (self.height * 0.45))
                cross_score = min(1.0, max(
                    bar["length"] for bar in crossing) / separation)
                score = (0.34 * width_score + 0.22 * center_score +
                         0.22 * rail_score + 0.22 * cross_score)
                candidates.append({
                    "left": left, "right": right, "bars": crossing,
                    "center": center, "width": separation,
                    "score": score, "far_y": overlap_lo,
                })

        debug = bev.copy()
        for line in vertical:
            cv2.line(debug, line["line"][:2], line["line"][2:],
                     (255, 180, 0), 2)
        for line in horizontal:
            cv2.line(debug, line["line"][:2], line["line"][2:],
                     (0, 220, 220), 2)
        result = dict(empty)
        result.update({
            "bev": bev, "mask": white, "debug": debug,
            "vertical_count": len(vertical),
            "horizontal_count": len(horizontal),
        })
        if not candidates:
            return result

        best = max(candidates, key=lambda item: item["score"])
        near_bar = max(best["bars"], key=lambda item: item["y"])
        near_y = float(near_bar["y"])
        near_distance = self.ground_depth_m * (
            (self.height - 1.0 - near_y) / (self.height - 1.0))
        near_distance = clamp(near_distance, 0.0, self.ground_depth_m)
        confidence = clamp(best["score"], 0.0, 1.0)

        def line_x_at_y(line, y_value):
            x1, y1, x2, y2 = line["line"]
            if abs(y2 - y1) < 1.0:
                return float(line["x"])
            return x1 + (float(y_value) - y1) * (x2 - x1) / float(y2 - y1)

        far_y = float(best["far_y"])
        left_near = line_x_at_y(best["left"], near_y)
        right_near = line_x_at_y(best["right"], near_y)
        left_far = line_x_at_y(best["left"], far_y)
        right_far = line_x_at_y(best["right"], far_y)
        center_near = 0.5 * (left_near + right_near)
        center_far = 0.5 * (left_far + right_far)
        # Aim at a point along the centerline, as the working line follower
        # does, so lateral and heading errors are corrected together.
        control_center = 0.55 * center_near + 0.45 * center_far
        raw_lateral = ((center_near - self.width * 0.5) /
                       self.pixels_per_meter_x)
        forward_depth = max(
            0.02, self.ground_depth_m * abs(near_y - far_y) /
            float(self.height - 1))
        heading_error = math.atan2(
            (center_far - center_near) / self.pixels_per_meter_x,
            forward_depth)
        lookahead_lateral = ((control_center - self.width * 0.5) /
                             self.pixels_per_meter_x)
        control_lateral = 0.55 * raw_lateral + 0.45 * lookahead_lateral
        control_lateral += 0.20 * heading_error
        result.update({
            "found": confidence >= 0.52,
            "confidence": confidence,
            "lateral_error_m": control_lateral,
            "raw_lateral_error_m": raw_lateral,
            "heading_error_rad": heading_error,
            "center_near_px": center_near,
            "center_far_px": center_far,
            "near_edge_distance_m": near_distance,
            "rail_width_m": best["width"] / self.pixels_per_meter_x,
            "rail_center_px": center_near,
            "near_edge_y_px": near_y,
        })
        cv2.line(debug, best["left"]["line"][:2],
                 best["left"]["line"][2:], (0, 0, 255), 4)
        cv2.line(debug, best["right"]["line"][:2],
                 best["right"]["line"][2:], (0, 0, 255), 4)
        cv2.line(debug, near_bar["line"][:2], near_bar["line"][2:],
                 (0, 255, 0), 4)
        return result
