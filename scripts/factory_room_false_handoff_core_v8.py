#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure decisions for rejecting and recovering false OCR handoffs."""

import math


KNOWN_WORKSHOPS = {
    "食品加工车间",
    "日用品加工车间",
    "电子产品生产车间",
}


def handoff_bbox_is_usable(bbox, image_width, edge_margin_px,
                           minimum_width_px):
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    try:
        x0, _, x1, _ = [float(value) for value in bbox]
        width = float(image_width)
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (x0, x1, width)):
        return False
    if width <= 1.0 or x1 <= x0:
        return False
    if x1 - x0 < minimum_width_px:
        return False
    return x0 > edge_margin_px and x1 < width - edge_margin_px


def parking_recheck_result(payload, target, now, minimum_stamp,
                           maximum_age_s, minimum_votes):
    if not isinstance(payload, dict) or not payload.get("stable"):
        return "inconclusive", ""
    try:
        stamp = float(payload.get("stamp", 0.0))
        votes = int(payload.get("votes", 0))
    except (TypeError, ValueError):
        return "inconclusive", ""
    if (stamp < minimum_stamp or now - stamp > maximum_age_s or
            votes < minimum_votes):
        return "inconclusive", ""
    label = str(payload.get("label", "")).strip()
    if label == target:
        return "target", label
    if label in KNOWN_WORKSHOPS:
        return "non_target", label
    return "inconclusive", label


def resume_transition(route_index, route_count):
    if route_count <= 0 or route_index < 0:
        return "invalid"
    return "complete" if route_index >= route_count - 1 else "advance"

