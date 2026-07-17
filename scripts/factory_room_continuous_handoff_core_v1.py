#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure validation helpers for the continuous-scan delivery handoff."""

import json
import re


VALID_WORKSHOPS = (
    "食品加工车间",
    "日用品加工车间",
    "电子产品生产车间",
)

_WORKSHOP_ALIASES = {
    "食品加工": "食品加工车间",
    "食品加工车间": "食品加工车间",
    "日用品加工": "日用品加工车间",
    "日用品加工车间": "日用品加工车间",
    "电子产品生产": "电子产品生产车间",
    "电子产品生产车间": "电子产品生产车间",
}


def compact_text(value):
    return re.sub(r"[\s，,。.;；:：]+", "", str(value or "")).strip()


def canonical_workshop(value):
    return _WORKSHOP_ALIASES.get(compact_text(value), "")


def parse_success_task(raw):
    payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
    if payload.get("status") != "success":
        raise ValueError("task status is not success")
    item = str(payload.get("selected_item", "")).strip()
    warehouse = canonical_workshop(payload.get("target_warehouse", ""))
    if not item or warehouse not in VALID_WORKSHOPS:
        raise ValueError("task result has no valid item/workshop")
    payload["selected_item"] = item
    payload["target_warehouse"] = warehouse
    return payload


def ocr_matches_target(payload, target):
    if not isinstance(payload, dict) or not payload.get("stable"):
        return False
    if canonical_workshop(payload.get("label", "")) != canonical_workshop(target):
        return False
    bbox = payload.get("bbox")
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4


def not_found_speech(target):
    warehouse = canonical_workshop(target) or str(target or "目标车间")
    return "未找到{}，小车已原地停止".format(warehouse)
