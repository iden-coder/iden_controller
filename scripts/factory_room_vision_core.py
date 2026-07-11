#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Ground white-square detection shared by the factory-room mission.

The calibration constants are taken from the working ucar_followline camera
pipeline.  This module deliberately has no ROS dependency so it can be tested
with recorded camera frames before the robot is allowed to move.
"""

import math

import cv2
import numpy as np


CAMERA_MATRIX = np.array([
    [637.5526471889214 * 0.5, 0.0, 639.0844243459007 * 0.5],
    [0.0, 637.5149155824262 * 0.5, 359.5701497245531 * 0.5],
    [0.0, 0.0, 1.0],
], dtype=np.float32)


def clamp(value, low, high):
    return max(low, min(high, value))


class GroundSquareDetector(object):
    """Detect the two side rails and crossbar of a 50 cm floor square."""

    def __init__(self, image_width=640, image_height=360,
                 camera_height_m=0.11, camera_pitch_deg=18.0,
                 ground_width_m=0.78, ground_depth_m=0.50):
        self.width = int(image_width)
        self.height = int(image_height)
        self.camera_height_m = float(camera_height_m)
        self.camera_pitch_deg = float(camera_pitch_deg)
        self.ground_width_m = float(ground_width_m)
        self.ground_depth_m = float(ground_depth_m)
        self.pixels_per_meter_x = self.width / self.ground_width_m
        self.homography = self._build_homography()

    def _build_homography(self):
        pitch = math.radians(self.camera_pitch_deg)
        ground = np.array([
            [-self.ground_width_m / 2.0, 0.0, 0.0],
            [self.ground_width_m / 2.0, 0.0, 0.0],
            [self.ground_width_m / 2.0, self.ground_depth_m, 0.0],
            [-self.ground_width_m / 2.0, self.ground_depth_m, 0.0],
        ], dtype=np.float32)
        image_points = []
        for x_ground, y_ground, _ in ground:
            x_cam = x_ground
            y_cam = (self.camera_height_m * math.cos(pitch) -
                     y_ground * math.sin(pitch))
            z_cam = (self.camera_height_m * math.sin(pitch) +
                     y_ground * math.cos(pitch))
            u = CAMERA_MATRIX[0, 0] * x_cam / z_cam + CAMERA_MATRIX[0, 2]
            v = CAMERA_MATRIX[1, 1] * y_cam / z_cam + CAMERA_MATRIX[1, 2]
            image_points.append([u, v])
        destination = np.array([
            [0.0, self.height - 1.0],
            [self.width - 1.0, self.height - 1.0],
            [self.width - 1.0, 0.0],
            [0.0, 0.0],
        ], dtype=np.float32)
        return cv2.getPerspectiveTransform(
            np.asarray(image_points, dtype=np.float32), destination)

    def _detect_raw_square(self, frame):
        """Find a perspective quadrilateral in the lower camera image.

        At normal sign-reading distance the 50 cm frame is a shallow
        trapezoid.  Detecting that trapezoid in the raw image is more reliable
        than extrapolating distant floor pixels into BEV coordinates.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([0, 0, 155], dtype=np.uint8),
            np.array([180, 105, 255], dtype=np.uint8))
        mask[:int(self.height * 0.42), :] = 0
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8),
            iterations=2)
        mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8),
                          iterations=1)
        found = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[-2]
        candidates = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 350.0:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            if (width < self.width * 0.09 or width > self.width * 0.78 or
                    height < self.height * 0.035 or height > self.height * 0.45 or
                    y < self.height * 0.40):
                continue
            aspect = float(width) / max(float(height), 1.0)
            if aspect < 1.35 or aspect > 9.0:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(approx) < 4 or len(approx) > 10:
                continue
            center_x = x + width * 0.5
            center_score = max(
                0.0, 1.0 - abs(center_x - self.width * 0.5) /
                (self.width * 0.48))
            rectangularity = clamp(area / max(float(width * height), 1.0),
                                   0.0, 1.0)
            expected_width = self.width * 0.25
            width_score = max(
                0.0, 1.0 - abs(width - expected_width) /
                (self.width * 0.25))
            bottom = float(y + height) / float(self.height)
            bottom_score = max(0.0, 1.0 - abs(bottom - 0.72) / 0.34)
            polygon_score = 1.0 if 4 <= len(approx) <= 6 else 0.55
            score = (0.36 * center_score + 0.22 * rectangularity +
                     0.13 * width_score + 0.12 * bottom_score +
                     0.17 * polygon_score)
            candidates.append({
                "score": score, "contour": contour, "approx": approx,
                "bbox": (x, y, width, height), "center_x": center_x,
                "rectangularity": rectangularity,
            })
        if not candidates:
            return None, mask, frame.copy()
        best = max(candidates, key=lambda item: item["score"])
        debug = frame.copy()
        for item in candidates:
            x, y, width, height = item["bbox"]
            color = (0, 0, 255) if item is best else (0, 180, 255)
            cv2.rectangle(debug, (x, y), (x + width, y + height), color, 2)
        confidence = clamp(best["score"], 0.0, 1.0)
        x, y, width, height = best["bbox"]
        lateral_error = ((best["center_x"] - self.width * 0.5) /
                         self.width * self.ground_width_m)
        cv2.putText(
            debug, "raw-square conf=%.2f off=%.2fm" %
            (confidence, lateral_error), (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        result = {
            "found": confidence >= 0.57,
            "confidence": confidence,
            "lateral_error_m": lateral_error,
            "near_edge_distance_m": None,
            "rail_width_m": float(width) / self.width * self.ground_width_m,
            "rail_center_px": best["center_x"],
            "near_edge_y_px": float(y + height),
            "vertical_count": 0,
            "horizontal_count": 0,
            "bev": frame,
            "mask": mask,
            "debug": debug,
        }
        return result, mask, debug

    def detect(self, bgr_image):
        empty = {
            "found": False,
            "confidence": 0.0,
            "lateral_error_m": 0.0,
            "near_edge_distance_m": None,
            "rail_width_m": None,
            "rail_center_px": None,
            "near_edge_y_px": None,
            "vertical_count": 0,
            "horizontal_count": 0,
            "bev": None,
            "mask": None,
            "debug": None,
        }
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            return empty

        frame = cv2.resize(bgr_image, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        raw_result, _, _ = self._detect_raw_square(frame)
        if raw_result is not None and raw_result["found"]:
            return raw_result
        bev = cv2.warpPerspective(
            frame, self.homography, (self.width, self.height),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 155], dtype=np.uint8),
                            np.array([180, 105, 255], dtype=np.uint8))

        # Ignore extrapolated borders produced by the perspective warp.
        border = max(5, int(self.width * 0.025))
        white[:, :border] = 0
        white[:, self.width - border:] = 0
        # The calibrated useful ground region starts below the upper warped
        # band.  Masking it prevents white walls, lamps and signs from being
        # mistaken for a parking-frame crossbar.
        white[:int(self.height * 0.28), :] = 0
        white[self.height - 4:, :] = 0
        white = cv2.morphologyEx(
            white, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8),
            iterations=1)
        white = cv2.morphologyEx(
            white, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8),
            iterations=1)

        lines = cv2.HoughLinesP(
            white, 1.0, np.pi / 180.0, threshold=36,
            minLineLength=48, maxLineGap=28)
        vertical = []
        horizontal = []
        if lines is not None:
            for packed in lines:
                x1, y1, x2, y2 = [int(v) for v in packed[0]]
                dx = x2 - x1
                dy = y2 - y1
                length = math.hypot(dx, dy)
                angle = abs(math.degrees(math.atan2(dy, dx)))
                angle = min(angle, 180.0 - angle)
                if angle >= 67.0 and length >= 58.0:
                    y_lo, y_hi = sorted((y1, y2))
                    if y_hi < self.height * 0.44:
                        continue
                    x_mid = 0.5 * (x1 + x2)
                    vertical.append({
                        "x": x_mid, "y_lo": y_lo, "y_hi": y_hi,
                        "length": length, "line": (x1, y1, x2, y2),
                    })
                elif angle <= 18.0 and length >= 92.0:
                    x_lo, x_hi = sorted((x1, x2))
                    horizontal.append({
                        "y": 0.5 * (y1 + y2), "x_lo": x_lo,
                        "x_hi": x_hi, "length": length,
                        "line": (x1, y1, x2, y2),
                    })

        best = None
        min_width_px = self.pixels_per_meter_x * 0.30
        max_width_px = self.pixels_per_meter_x * 0.68
        expected_width_px = self.pixels_per_meter_x * 0.50
        for left in vertical:
            for right in vertical:
                if right["x"] <= left["x"]:
                    continue
                separation = right["x"] - left["x"]
                if separation < min_width_px or separation > max_width_px:
                    continue
                overlap = min(left["y_hi"], right["y_hi"]) - max(
                    left["y_lo"], right["y_lo"])
                if overlap < 45.0:
                    continue
                center = 0.5 * (left["x"] + right["x"])
                crossing = []
                margin = 0.08 * separation
                overlap_lo = max(left["y_lo"], right["y_lo"])
                overlap_hi = min(left["y_hi"], right["y_hi"])
                for bar in horizontal:
                    if (bar["x_lo"] <= left["x"] + margin and
                            bar["x_hi"] >= right["x"] - margin and
                            overlap_lo - 22.0 <= bar["y"] <= overlap_hi + 22.0):
                        crossing.append(bar)
                width_score = max(
                    0.0, 1.0 - abs(separation - expected_width_px) /
                    max(expected_width_px * 0.45, 1.0))
                center_score = max(
                    0.0, 1.0 - abs(center - self.width * 0.5) /
                    (self.width * 0.65))
                rail_score = min(1.0, overlap / (self.height * 0.48))
                bar_score = 1.0 if crossing else 0.0
                score = (0.33 * width_score + 0.21 * center_score +
                         0.27 * rail_score + 0.19 * bar_score)
                if best is None or score > best["score"]:
                    best = {
                        "left": left, "right": right, "bars": crossing,
                        "center": center, "width": separation,
                        "score": score,
                    }

        debug = bev.copy()
        for line in vertical:
            cv2.line(debug, line["line"][:2], line["line"][2:],
                     (255, 180, 0), 2)
        for line in horizontal:
            cv2.line(debug, line["line"][:2], line["line"][2:],
                     (0, 220, 220), 2)

        result = dict(empty)
        result.update({
            "bev": bev,
            "mask": white,
            "debug": debug,
            "vertical_count": len(vertical),
            "horizontal_count": len(horizontal),
        })
        if best is None:
            return result

        near_bar = max(best["bars"], key=lambda item: item["y"]) \
            if best["bars"] else None
        near_y = near_bar["y"] if near_bar is not None else None
        near_distance = None
        if near_y is not None:
            near_distance = self.ground_depth_m * (
                (self.height - 1.0 - near_y) / (self.height - 1.0))
            near_distance = clamp(near_distance, 0.0, self.ground_depth_m)

        confidence = clamp(best["score"], 0.0, 1.0)
        found = confidence >= 0.52 and best["bars"]
        result.update({
            "found": bool(found),
            "confidence": confidence,
            "lateral_error_m": (
                (best["center"] - self.width * 0.5) /
                self.pixels_per_meter_x),
            "near_edge_distance_m": near_distance,
            "rail_width_m": best["width"] / self.pixels_per_meter_x,
            "rail_center_px": best["center"],
            "near_edge_y_px": near_y,
        })

        left = best["left"]["line"]
        right = best["right"]["line"]
        cv2.line(debug, left[:2], left[2:], (0, 0, 255), 4)
        cv2.line(debug, right[:2], right[2:], (0, 0, 255), 4)
        if near_bar is not None:
            line = near_bar["line"]
            cv2.line(debug, line[:2], line[2:], (0, 255, 0), 4)
        cv2.circle(debug, (int(best["center"]), self.height // 2),
                   7, (255, 0, 255), -1)
        cv2.putText(
            debug,
            "square=%s conf=%.2f off=%.2fm" %
            (str(bool(found)), confidence, result["lateral_error_m"]),
            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 0), 2)
        return result


def decode_ros_image(msg):
    """Convert a sensor_msgs/Image-like object to BGR without cv_bridge."""
    encoding = (getattr(msg, "encoding", "") or "").lower()
    height = int(getattr(msg, "height", 0))
    width = int(getattr(msg, "width", 0))
    if height <= 0 or width <= 0:
        return None
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if encoding in ("rgb8", "bgr8"):
            expected = height * width * 3
            image = raw[:expected].reshape((height, width, 3))
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(image)
        if encoding in ("rgba8", "bgra8"):
            expected = height * width * 4
            image = raw[:expected].reshape((height, width, 4))
            code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(image, code)
        if encoding in ("mono8", "8uc1"):
            image = raw[:height * width].reshape((height, width))
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    except (ValueError, IndexError):
        return None
    return None
