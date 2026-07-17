#!/usr/bin/env python3

"""Pure longitudinal decisions for precise wall parking."""


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def longitudinal_decision(
        distance_error, tolerance, stop_lead, settled,
        near_band, near_min_speed, near_max_speed,
        far_full_speed_error, far_max_speed, reverse_speed):
    """Return ``(phase, velocity)`` for a signed wall-distance error."""
    if abs(distance_error) <= tolerance:
        return "hold", 0.0
    if distance_error < -tolerance:
        return "reverse", -abs(reverse_speed)
    if distance_error <= stop_lead and not settled:
        return "settle", 0.0
    if distance_error <= stop_lead:
        return "crawl", abs(near_min_speed)
    if distance_error <= near_band:
        speed = clamp(
            0.8 * distance_error,
            abs(near_min_speed), abs(near_max_speed))
        return "near", speed

    denominator = max(0.001, far_full_speed_error - near_band)
    ratio = clamp((distance_error - near_band) / denominator, 0.0, 1.0)
    speed = near_max_speed + ratio * (far_max_speed - near_max_speed)
    return "far", clamp(speed, near_max_speed, far_max_speed)
