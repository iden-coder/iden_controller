#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure geometry for selecting a front wall and bootstrapping heading."""

import math

import numpy as np


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def norm_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def estimated_sign_heading(current_yaw, error_px, image_width,
                           camera_hfov_rad):
    if image_width <= 1.0:
        return norm_angle(current_yaw)
    normalized = clamp(error_px / (0.5 * image_width), -1.0, 1.0)
    bearing = -normalized * 0.5 * camera_hfov_rad
    return norm_angle(current_yaw + bearing)


def nearest_orthogonal_heading(estimated_heading):
    quarter_turn = 0.5 * math.pi
    return norm_angle(round(estimated_heading / quarter_turn) * quarter_turn)


def fit_oriented_front_wall(points, inlier_threshold, min_inliers,
                            min_span, max_heading):
    """Fit the best wall whose normal already points near robot-forward."""
    if len(points) < min_inliers:
        return None
    data = np.asarray(points, dtype=np.float64)
    count = len(data)
    stride = max(1, count // 42)
    candidates = range(0, count, stride)
    best_indices = None
    best_score = -1.0

    for first in candidates:
        for second in candidates:
            if second <= first + max(2, stride):
                continue
            delta = data[second] - data[first]
            length = float(np.linalg.norm(delta))
            if length < 0.55 * min_span:
                continue
            normal = np.array([-delta[1], delta[0]]) / length
            midpoint = 0.5 * (data[first] + data[second])
            if float(normal.dot(midpoint)) < 0.0:
                normal = -normal
            heading = math.atan2(float(normal[1]), float(normal[0]))
            if abs(heading) > max_heading:
                continue

            residual = np.abs((data - data[first]).dot(normal))
            indices = np.flatnonzero(residual <= inlier_threshold)
            if len(indices) < min_inliers:
                continue
            tangent = np.array([-normal[1], normal[0]])
            projection = data[indices].dot(tangent)
            span = float(projection.max() - projection.min())
            if span < min_span:
                continue
            score = len(indices) + 12.0 * span
            if score > best_score:
                best_score = score
                best_indices = indices

    if best_indices is None:
        return None
    inliers = data[best_indices]
    centroid = inliers.mean(axis=0)
    covariance = np.cov((inliers - centroid).T)
    values, vectors = np.linalg.eigh(covariance)
    normal = vectors[:, int(np.argmin(values))]
    if float(normal.dot(centroid)) < 0.0:
        normal = -normal
    heading = math.atan2(float(normal[1]), float(normal[0]))
    distance = float(normal.dot(centroid))
    if distance <= 0.0 or abs(heading) > max_heading:
        return None
    tangent = np.array([-normal[1], normal[0]])
    projection = inliers.dot(tangent)
    span = float(projection.max() - projection.min())
    residual = np.abs((inliers - centroid).dot(normal))
    return {
        "distance": distance,
        "heading_error": heading,
        "inliers": int(len(inliers)),
        "span": span,
        "rms": float(math.sqrt(np.mean(residual * residual))),
        "method": "oriented_front_fallback",
    }


def ocr_heading_command(error_px, image_width, tolerance_px, kp,
                        min_speed, max_speed):
    if image_width <= 1.0 or abs(error_px) <= tolerance_px:
        return 0.0
    normalized = error_px / max(1.0, 0.5 * image_width)
    command = -kp * normalized
    magnitude = clamp(abs(command), min_speed, max_speed)
    return math.copysign(magnitude, command)
