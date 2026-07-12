#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Adaptive far/close parking-frame detector for full-mission handoff."""

import cv2
import numpy as np

from factory_room_vision_parking_v5 import PerspectiveParkingDetector
from factory_room_vision_parking_v6 import CurvedCrossbarParkingDetector


class AdaptiveRangeParkingDetector(object):
    """Try the calibrated view, then a lower-image close-range view."""

    def __init__(self, max_width=800):
        self.max_width = int(max_width)
        self.far_detector = CurvedCrossbarParkingDetector(max_width=max_width)
        self.close_detector = PerspectiveParkingDetector(max_width=max_width)

    @staticmethod
    def _close_crossbar_proxy(frame):
        height, width = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv, np.array([0, 0, 175], dtype=np.uint8),
            np.array([180, 120, 255], dtype=np.uint8))
        lower = np.zeros_like(white)
        y0 = int(0.42 * height)
        lower[y0:height - 1, :] = white[y0:height - 1, :]
        lower = cv2.morphologyEx(
            lower, cv2.MORPH_CLOSE, np.ones((9, 9), dtype=np.uint8),
            iterations=1)
        contours = cv2.findContours(
            lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        best = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if (area < 0.004 * width * height or
                    box_w < 0.64 * width or
                    y < 0.42 * height):
                continue
            score = box_w / float(width) + min(
                1.0, area / (0.030 * width * height))
            if best is None or score > best["score"]:
                best = {"score": score, "contour": contour}
        if best is None:
            # At very close range the center of the curved entrance band has
            # passed below the image; only its two lower corner segments remain.
            edge_y0 = int(0.72 * height)
            edge_w = int(0.24 * width)
            left_pixels = np.where(white[edge_y0:, :edge_w] > 0)
            right_pixels = np.where(white[edge_y0:, width - edge_w:] > 0)
            min_edge_pixels = int(0.0025 * width * height)
            if (len(left_pixels[0]) < min_edge_pixels or
                    len(right_pixels[0]) < min_edge_pixels):
                return frame, None
            combined_y = np.concatenate((left_pixels[0], right_pixels[0]))
            real_y = int(np.median(combined_y)) + edge_y0
            proxy_y = int(0.64 * height)
            augmented = frame.copy()
            cv2.line(
                augmented, (int(0.04 * width), proxy_y),
                (int(0.96 * width), proxy_y), (245, 245, 245),
                max(8, int(0.016 * height)))
            return augmented, real_y

        component = np.zeros_like(white)
        cv2.drawContours(
            component, [best["contour"]], -1, 255,
            thickness=cv2.FILLED)
        ys, _ = np.where(component > 0)
        if len(ys) == 0:
            return frame, None
        real_y = int(np.median(ys))
        proxy_y = int(0.64 * height)
        augmented = frame.copy()
        cv2.line(
            augmented, (int(0.04 * width), proxy_y),
            (int(0.96 * width), proxy_y), (245, 245, 245),
            max(8, int(0.016 * height)))
        return augmented, real_y

    def detect(self, bgr_image):
        far = self.far_detector.detect(bgr_image)
        far["range_mode"] = "far"
        far_strong = (
            far.get("found") and
            int(far.get("left_count", 0)) >= 3 and
            int(far.get("right_count", 0)) >= 3 and
            int(far.get("horizontal_count", 0)) <= 80)
        if far_strong:
            return far
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            return far

        source_h, source_w = bgr_image.shape[:2]
        crop_top = int(0.42 * source_h)
        close_view = bgr_image[crop_top:source_h, :]
        if close_view.size == 0:
            return far
        close_view = cv2.resize(
            close_view, (source_w, source_h), interpolation=cv2.INTER_LINEAR)
        augmented, crossbar_y = self._close_crossbar_proxy(close_view)
        close = self.close_detector.detect(augmented)
        close["range_mode"] = "close"
        close["close_crop_top_px"] = crop_top
        close["close_crossbar_y_px"] = crossbar_y
        close["source_width"] = source_w
        close["source_height"] = source_h
        if close.get("debug") is not None:
            cv2.putText(
                close["debug"], "adaptive CLOSE crop=%d band=%s" %
                (crop_top, str(crossbar_y)), (12, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 2)

        if close.get("found"):
            return close
        if far.get("found"):
            far["weak_far_geometry"] = True
            return far
        # Return the attempt with more complete rail evidence for diagnostics.
        far_evidence = (int(far.get("vertical_count", 0)) +
                        int(far.get("horizontal_count", 0)))
        close_evidence = (int(close.get("vertical_count", 0)) +
                          int(close.get("horizontal_count", 0)))
        return close if close_evidence > far_evidence else far
