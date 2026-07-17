#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import unittest

from factory_room_continuous_handoff_core_v1 import (
    canonical_workshop,
    not_found_speech,
    ocr_matches_target,
    parse_success_task,
)


class ContinuousHandoffCoreTest(unittest.TestCase):
    def test_task_validation(self):
        task = parse_success_task(json.dumps({
            "status": "success",
            "selected_item": "大米",
            "target_warehouse": "食品加工车间",
        }, ensure_ascii=False))
        self.assertEqual(task["selected_item"], "大米")
        self.assertEqual(task["target_warehouse"], "食品加工车间")

    def test_invalid_task_is_rejected(self):
        with self.assertRaises(ValueError):
            parse_success_task({
                "status": "success",
                "selected_item": "大米",
                "target_warehouse": "食品加工类",
            })

    def test_ocr_requires_stable_bbox_and_exact_workshop(self):
        target = "电子产品生产车间"
        self.assertTrue(ocr_matches_target({
            "stable": True,
            "label": "电子产品生产车间",
            "bbox": [10, 20, 100, 80],
        }, target))
        self.assertFalse(ocr_matches_target({
            "stable": False,
            "label": target,
            "bbox": [10, 20, 100, 80],
        }, target))
        self.assertFalse(ocr_matches_target({
            "stable": True,
            "label": "日用品加工车间",
            "bbox": [10, 20, 100, 80],
        }, target))

    def test_normalization_is_deliberately_narrow(self):
        self.assertEqual(canonical_workshop(" 食品加工车间\n"), "食品加工车间")
        self.assertEqual(canonical_workshop("食品加工"), "食品加工车间")
        self.assertEqual(canonical_workshop("食品加工类"), "")

    def test_not_found_speech(self):
        self.assertEqual(
            not_found_speech("日用品加工车间"),
            "未找到日用品加工车间，小车已原地停止")


if __name__ == "__main__":
    unittest.main()
