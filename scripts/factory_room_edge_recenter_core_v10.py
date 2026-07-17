#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure image-edge decisions for continuous workshop-sign scanning."""

import math


def edge_target_recenter_decision(bbox, image_width, edge_margin_px,
                                  minimum_width_px, gain,
                                  minimum_speed, maximum_speed):
    """Return ``(state, wz)`` for a target sign near an image edge."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "invalid", 0.0
    try:
        x0, _, x1, _ = [float(value) for value in bbox]
        width = float(image_width)
    except (TypeError, ValueError):
        return "invalid", 0.0
    if (not all(math.isfinite(value) for value in (x0, x1, width)) or
            width <= 1.0 or x1 <= x0):
        return "invalid", 0.0
    if x1 - x0 < float(minimum_width_px):
        return "partial", 0.0
    if (x0 > float(edge_margin_px) and
            x1 < width - float(edge_margin_px)):
        return "usable", 0.0

    center = 0.5 * (x0 + x1)
    normalized_error = (0.5 * width - center) / max(0.5 * width, 1.0)
    command = float(gain) * normalized_error
    limit = max(0.0, float(maximum_speed))
    command = max(-limit, min(limit, command))
    floor = max(0.0, min(float(minimum_speed), limit))
    if abs(command) < floor:
        command = math.copysign(floor, normalized_error)
    return "recenter", command

