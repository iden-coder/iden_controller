#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure decisions for fast, single-pass workshop-sign centering."""

import math


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def direct_lateral_decision(error_px, tolerance_px, requested_speed,
                            clearance, hard_clearance, slow_clearance,
                            minimum_effective_speed, narrow_speed_cap):
    if abs(error_px) <= tolerance_px:
        return {"action": "centered", "command": 0.0, "guard": "centered"}
    if math.isfinite(clearance) and clearance <= hard_clearance:
        return {"action": "blocked", "command": 0.0, "guard": "blocked"}

    magnitude = max(abs(requested_speed), minimum_effective_speed)
    guard = "clear"
    if math.isfinite(clearance) and clearance < slow_clearance:
        guard = "narrow"
        ratio = clamp(
            (clearance - hard_clearance) /
            max(0.01, slow_clearance - hard_clearance), 0.0, 1.0)
        magnitude = max(minimum_effective_speed, magnitude * ratio)
        magnitude = min(magnitude, narrow_speed_cap)
    command = math.copysign(magnitude, requested_speed)
    return {"action": "move", "command": command, "guard": guard}


def progress_state(best_error, current_error, last_progress_time, now,
                   improvement_px, boost_after_s, fail_after_s):
    current = abs(current_error)
    if current + improvement_px < best_error:
        return {
            "best_error": current,
            "last_progress_time": now,
            "boost": False,
            "failed": False,
        }
    elapsed = now - last_progress_time
    return {
        "best_error": best_error,
        "last_progress_time": last_progress_time,
        "boost": elapsed >= boost_after_s,
        "failed": elapsed >= fail_after_s,
    }

