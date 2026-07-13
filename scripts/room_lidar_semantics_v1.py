#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Separate compact cone returns from long wall returns in a LaserScan."""

import math


INF = float("inf")


def _sector(angle_deg, lo_deg, hi_deg):
    return lo_deg <= angle_deg <= hi_deg


def _cluster_geometry(points):
    count = len(points)
    if count == 1:
        return 0.0, 1.0
    xs = [p[2] for p in points]
    ys = [p[3] for p in points]
    span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
    mx = sum(xs) / count
    my = sum(ys) / count
    cxx = sum((x - mx) ** 2 for x in xs) / count
    cyy = sum((y - my) ** 2 for y in ys) / count
    cxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / count
    trace = cxx + cyy
    disc = math.sqrt(max(0.0, (cxx - cyy) ** 2 + 4.0 * cxy * cxy))
    major = 0.5 * (trace + disc)
    minor = 0.5 * (trace - disc)
    elongation = major / max(minor, 1.0e-6)
    return span, elongation


def classify_scan(msg, cone_base_pad_m=0.10, max_cone_range_m=1.50,
                  front_angle_deg=48.0, side_min_deg=35.0,
                  side_max_deg=82.0, wall_front_relax_m=0.09,
                  wall_side_relax_m=0.05):
    points = []
    for index, value in enumerate(msg.ranges):
        if math.isnan(value) or math.isinf(value):
            continue
        if value < msg.range_min or value > msg.range_max:
            continue
        angle = msg.angle_min + index * msg.angle_increment
        angle_deg = math.degrees(angle)
        points.append((index, value, value * math.cos(angle),
                       value * math.sin(angle), angle_deg))

    clusters = []
    current = []
    for point in points:
        if current:
            previous = current[-1]
            index_gap = point[0] - previous[0]
            spatial_gap = math.hypot(point[2] - previous[2],
                                     point[3] - previous[3])
            split_gap = max(0.075, 0.035 + 0.035 * min(point[1], previous[1]))
            if index_gap > 2 or spatial_gap > split_gap:
                clusters.append(current)
                current = []
        current.append(point)
    if current:
        clusters.append(current)

    result = {
        "cone_front": INF, "cone_left": INF, "cone_right": INF,
        "wall_front": INF, "wall_left": INF, "wall_right": INF,
        "effective_front": INF, "effective_left": INF,
        "effective_right": INF, "cone_clusters": 0, "wall_clusters": 0,
        "cone_observations": [],
        "cone_turn": INF,
    }
    for cluster in clusters:
        span, elongation = _cluster_geometry(cluster)
        count = len(cluster)
        nearest = min(p[1] for p in cluster)
        center_angle = sum(p[4] for p in cluster) / count
        is_wall = ((span >= 0.28 and elongation >= 10.0) or
                   span >= 0.45 or (count >= 24 and span >= 0.22))
        if not is_wall and nearest > max_cone_range_m:
            is_wall = True
        kind = "wall" if is_wall else "cone"
        result[kind + "_clusters"] += 1
        distance = nearest if is_wall else max(0.0, nearest - cone_base_pad_m)
        if not is_wall:
            nearest_point = min(cluster, key=lambda p: p[1])
            result["cone_observations"].append({
                "range": nearest_point[1],
                "angle_deg": nearest_point[4],
                "span": span,
                "count": count,
            })
            result["cone_turn"] = min(result["cone_turn"], distance)
        if _sector(center_angle, -front_angle_deg, front_angle_deg):
            key = kind + "_front"
            result[key] = min(result[key], distance)
        if _sector(center_angle, side_min_deg, side_max_deg):
            key = kind + "_left"
            result[key] = min(result[key], distance)
        if _sector(center_angle, -side_max_deg, -side_min_deg):
            key = kind + "_right"
            result[key] = min(result[key], distance)

    result["effective_front"] = min(
        result["cone_front"], result["wall_front"] + wall_front_relax_m)
    result["effective_left"] = min(
        result["cone_left"], result["wall_left"] + wall_side_relax_m)
    result["effective_right"] = min(
        result["cone_right"], result["wall_right"] + wall_side_relax_m)
    return result
