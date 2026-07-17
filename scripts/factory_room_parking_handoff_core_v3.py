#!/usr/bin/env python3

"""Pure guards for fresh OCR handoff and lateral parking preparation."""


def is_fresh_target_payload(payload, target, not_before):
    if not isinstance(payload, dict):
        return False
    if not payload.get("stable") or str(payload.get("label", "")) != target:
        return False
    if float(payload.get("stamp", 0.0)) < float(not_before):
        return False
    bbox = payload.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    return float(payload.get("image_width", 0.0)) > 1.0


def precenter_action(error_px, tolerance_px, clearance, hard_clearance,
                     straight_min_clearance, can_advance):
    if abs(float(error_px)) <= float(tolerance_px):
        return "centered"
    if float(clearance) > float(hard_clearance):
        return "lateral"
    if can_advance and float(clearance) >= float(straight_min_clearance):
        return "advance"
    return "reobserve"
