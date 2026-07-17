#!/usr/bin/env python3
"""ROS-independent helpers for smooth directed yaw sweeps."""

import math


def wrap_angle(angle):
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def directed_yaw_increment(previous_yaw, current_yaw, direction):
    """Return progress in the requested direction across the +/-pi seam."""
    signed = wrap_angle(float(current_yaw) - float(previous_yaw))
    return max(0.0, (1.0 if direction >= 0 else -1.0) * signed)


def braking_limited_speed(remaining, maximum_speed, acceleration,
                           minimum_speed=0.0, stop_tolerance=0.0):
    remaining = max(0.0, float(remaining))
    if remaining <= max(0.0, float(stop_tolerance)):
        return 0.0
    maximum = max(0.0, float(maximum_speed))
    acceleration = max(1.0e-6, float(acceleration))
    braking = math.sqrt(2.0 * acceleration * remaining)
    return min(maximum, max(float(minimum_speed), braking))


def slew_rate(current, target, acceleration, dt):
    step = max(0.0, float(acceleration)) * max(0.0, float(dt))
    error = float(target) - float(current)
    if abs(error) <= step:
        return float(target)
    return float(current) + math.copysign(step, error)

