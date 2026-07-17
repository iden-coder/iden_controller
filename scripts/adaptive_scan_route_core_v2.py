#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ROS-independent geometry helpers for the v2 continuous scan route."""

import math


def approach_point(goal_x, goal_y, start_yaw, distance):
    """Return a point behind a scan goal along its desired arrival heading."""
    distance = max(0.0, float(distance))
    return (goal_x - distance * math.cos(start_yaw),
            goal_y - distance * math.sin(start_yaw))


def merge_groups(points, radius):
    """Group close 2-D points while preventing chain-merging distinct cones."""
    radius = max(0.0, float(radius))
    groups = []
    for index, point in enumerate(points):
        selected = None
        selected_distance = float("inf")
        for group in groups:
            distances = [math.hypot(point[0] - points[member][0],
                                    point[1] - points[member][1])
                         for member in group]
            if distances and max(distances) <= radius:
                mean_distance = sum(distances) / len(distances)
                if mean_distance < selected_distance:
                    selected = group
                    selected_distance = mean_distance
        if selected is None:
            groups.append([index])
        else:
            selected.append(index)
    return groups


def arrival_heading_correction(yaw_error, distance, blend_distance,
                               gain, maximum):
    """Smoothly add heading guidance only on the final moving approach."""
    blend_distance = max(float(blend_distance), 1.0e-3)
    blend = max(0.0, min(1.0,
                         (blend_distance - max(0.0, distance)) /
                         blend_distance))
    correction = blend * float(gain) * float(yaw_error)
    return max(-float(maximum), min(float(maximum), correction))
