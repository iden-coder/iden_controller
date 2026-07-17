#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure decisions for bringing a partially visible workshop sign on screen."""

import math


def suspect_lateral_command(bbox, image_width, center_tolerance_px,
                            command_sign, gain, minimum_speed,
                            maximum_speed):
    """Return ``(state, vy, error_px)`` for lateral sign acquisition."""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return "invalid", 0.0, 0.0
    try:
        x0, _, x1, _ = [float(value) for value in bbox]
        width = float(image_width)
    except (TypeError, ValueError):
        return "invalid", 0.0, 0.0
    if (not all(math.isfinite(value) for value in (x0, x1, width)) or
            width <= 1.0 or x1 <= x0):
        return "invalid", 0.0, 0.0

    error_px = 0.5 * (x0 + x1) - 0.5 * width
    if abs(error_px) <= max(0.0, float(center_tolerance_px)):
        return "centered", 0.0, error_px

    normalized = error_px / max(1.0, 0.5 * width)
    limit = max(0.0, float(maximum_speed))
    command = float(command_sign) * float(gain) * normalized
    command = max(-limit, min(limit, command))
    floor = max(0.0, min(float(minimum_speed), limit))
    if abs(command) < floor:
        command = math.copysign(floor, command)
    return "move", command, error_px


def lateral_acquisition_exit(displacement_m, maximum_shift_m,
                             observation_age_s, fresh_timeout_s,
                             no_progress_s, stuck_timeout_s):
    """Return an exit reason, or an empty string while acquisition may run."""
    values = (displacement_m, maximum_shift_m, observation_age_s,
              fresh_timeout_s, no_progress_s, stuck_timeout_s)
    try:
        values = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return "invalid"
    if any(math.isnan(value) for value in values):
        return "invalid"
    if (not math.isfinite(values[1]) or not math.isfinite(values[3]) or
            not math.isfinite(values[5])):
        return "invalid"
    if values[0] >= max(0.0, values[1]):
        return "shift_limit"
    if values[2] > max(0.0, values[3]):
        return "ocr_stale"
    if values[4] > max(0.0, values[5]):
        return "blocked"
    return ""


def physical_view_ignore_decision(now_s, minimum_until_s, maximum_until_s,
                                  current_yaw, anchor_yaw, same_route,
                                  release_yaw_rad):
    """Return ``(blocked, yaw_delta)`` for a confirmed non-target view."""
    try:
        now = float(now_s)
        minimum_until = float(minimum_until_s)
        maximum_until = float(maximum_until_s)
        release_yaw = max(0.0, float(release_yaw_rad))
    except (TypeError, ValueError):
        return False, float("inf")
    if (not same_route or any(math.isnan(value) for value in (
            now, minimum_until, maximum_until, release_yaw)) or
            now >= maximum_until):
        return False, float("inf")
    if current_yaw is None or anchor_yaw is None:
        return True, float("inf")
    try:
        delta = math.atan2(
            math.sin(float(current_yaw) - float(anchor_yaw)),
            math.cos(float(current_yaw) - float(anchor_yaw)))
    except (TypeError, ValueError):
        return True, float("inf")
    yaw_delta = abs(delta)
    return now < minimum_until or yaw_delta < release_yaw, yaw_delta


def orthogonal_alignment_command(wall_error, fallback_error, tolerance_rad,
                                 gain, minimum_speed, maximum_speed):
    """Return ``(state, wz, error, source)`` for wall-normal alignment."""
    source = "lidar" if wall_error is not None else "cardinal_fallback"
    selected = wall_error if wall_error is not None else fallback_error
    try:
        error = math.atan2(math.sin(float(selected)), math.cos(float(selected)))
        tolerance = max(0.0, float(tolerance_rad))
        gain = max(0.0, float(gain))
        minimum = max(0.0, float(minimum_speed))
        maximum = max(0.0, float(maximum_speed))
    except (TypeError, ValueError):
        return "invalid", 0.0, 0.0, source
    if not all(math.isfinite(value) for value in (
            error, tolerance, gain, minimum, maximum)):
        return "invalid", 0.0, 0.0, source
    if abs(error) <= tolerance:
        return "aligned", 0.0, error, source
    limit = max(minimum, maximum)
    command = max(-limit, min(limit, gain * error))
    floor = min(minimum, limit)
    if abs(command) < floor:
        command = math.copysign(floor, command)
    return "rotate", command, error, source


def orthogonal_handoff_ready(wall_error, wall_tolerance_rad, stable_label,
                             target_label, center_error_px,
                             center_tolerance_px):
    """Allow parking only after a fresh, square-on, centered target vote."""
    if not stable_label or stable_label != target_label:
        return False
    try:
        wall_error = abs(float(wall_error))
        wall_tolerance = max(0.0, float(wall_tolerance_rad))
        center_error = abs(float(center_error_px))
        center_tolerance = max(0.0, float(center_tolerance_px))
    except (TypeError, ValueError):
        return False
    if not all(math.isfinite(value) for value in (
            wall_error, wall_tolerance, center_error, center_tolerance)):
        return False
    return wall_error <= wall_tolerance and center_error <= center_tolerance


def near_scan_micro_command(forward_error_m, lateral_error_m, yaw_error_rad,
                            position_tolerance_m, yaw_slow_rad,
                            forward_gain, lateral_gain, yaw_gain,
                            maximum_forward_mps, maximum_lateral_mps,
                            maximum_yaw_rps):
    """Return a bounded holonomic correction near a scan position."""
    try:
        forward = float(forward_error_m)
        lateral = float(lateral_error_m)
        yaw_error = math.atan2(
            math.sin(float(yaw_error_rad)), math.cos(float(yaw_error_rad)))
        tolerance = max(0.0, float(position_tolerance_m))
        yaw_slow = max(0.01, float(yaw_slow_rad))
        forward_gain = max(0.0, float(forward_gain))
        lateral_gain = max(0.0, float(lateral_gain))
        yaw_gain = max(0.0, float(yaw_gain))
        max_forward = max(0.0, float(maximum_forward_mps))
        max_lateral = max(0.0, float(maximum_lateral_mps))
        max_yaw = max(0.0, float(maximum_yaw_rps))
    except (TypeError, ValueError):
        return "invalid", 0.0, 0.0, 0.0
    values = (forward, lateral, yaw_error, tolerance, yaw_slow,
              forward_gain, lateral_gain, yaw_gain, max_forward,
              max_lateral, max_yaw)
    if not all(math.isfinite(value) for value in values):
        return "invalid", 0.0, 0.0, 0.0

    distance = math.hypot(forward, lateral)
    if distance <= tolerance:
        return "reached", 0.0, 0.0, 0.0

    # Large heading errors receive only a little translation. Reverse is not
    # requested: the shared safety monitor intentionally forbids it outside
    # the close-wall parking controller.
    heading_scale = max(0.20, min(1.0, yaw_slow / max(yaw_slow, abs(yaw_error))))
    vx = max(0.0, min(max_forward, forward_gain * forward))
    vy = max(-max_lateral, min(max_lateral, lateral_gain * lateral))
    vx *= heading_scale
    vy *= max(0.45, heading_scale)
    wz = max(-max_yaw, min(max_yaw, yaw_gain * yaw_error))
    return "move", vx, vy, wz
