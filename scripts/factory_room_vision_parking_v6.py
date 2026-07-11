#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Parking-frame detector accepting a fisheye-curved entrance crossbar."""

import cv2
import numpy as np

from factory_room_vision_parking_v5 import PerspectiveParkingDetector


class CurvedCrossbarParkingDetector(PerspectiveParkingDetector):
    def detect(self, bgr_image):
        if bgr_image is None or getattr(bgr_image, "size", 0) == 0:
            return super(CurvedCrossbarParkingDetector, self).detect(bgr_image)

        source_h, source_w = bgr_image.shape[:2]
        scale = min(1.0, float(self.max_width) / max(float(source_w), 1.0))
        if scale < 0.999:
            frame = cv2.resize(
                bgr_image,
                (int(round(source_w * scale)), int(round(source_h * scale))),
                interpolation=cv2.INTER_AREA)
        else:
            frame = bgr_image.copy()
        height, width = frame.shape[:2]

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(
            hsv, np.array([0, 0, 185], dtype=np.uint8),
            np.array([180, 85, 255], dtype=np.uint8))
        y0 = int(0.30 * height)
        y1 = int(0.78 * height)
        lower = np.zeros_like(white)
        lower[y0:y1, :] = white[y0:y1, :]
        lower = cv2.morphologyEx(
            lower, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8),
            iterations=1)

        found = cv2.findContours(
            lower, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[-2]
        crossbar = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if (area < 0.006 * width * height or
                    box_w < 0.68 * width or
                    box_h < 0.025 * height or
                    y > 0.74 * height):
                continue
            score = (box_w / float(width) +
                     min(1.0, area / (0.035 * width * height)))
            if crossbar is None or score > crossbar["score"]:
                crossbar = {
                    "score": score, "contour": contour,
                    "bbox": (x, y, box_w, box_h),
                }

        augmented = frame.copy()
        curved_y = None
        if crossbar is not None:
            component = np.zeros_like(white)
            cv2.drawContours(
                component, [crossbar["contour"]], -1, 255,
                thickness=cv2.FILLED)
            ys, xs = np.where(component > 0)
            if len(ys) > 0:
                # Median is robust when side rails are connected to the band.
                curved_y = int(np.median(ys))
                curved_y = max(int(0.31 * height),
                               min(int(0.74 * height), curved_y))
                # The V5 pair validator accepts crossbars up to 66% image
                # height.  Draw only a validation proxy there; retain the real
                # fisheye-band row in curved_y for diagnostics.
                proxy_y = min(curved_y, int(0.64 * height))
                cv2.line(
                    augmented, (int(0.04 * width), proxy_y),
                    (int(0.96 * width), proxy_y), (245, 245, 245),
                    max(8, int(0.016 * height)))

        result = super(CurvedCrossbarParkingDetector, self).detect(augmented)
        result["curved_crossbar"] = curved_y is not None
        result["curved_crossbar_y_px"] = curved_y
        if result.get("debug") is not None and curved_y is not None:
            cv2.putText(
                result["debug"], "fisheye crossbar y=%d" % curved_y,
                (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 255, 255), 2)
        return result
