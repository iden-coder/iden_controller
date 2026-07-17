#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_delivery_voice_core_v5 import delivery_voice


class DeliveryVoiceCoreV5Test(unittest.TestCase):
    def test_real_delivery_sentence(self):
        self.assertEqual(
            delivery_voice("手机", "电子产品生产车间"),
            "已将手机放入电子产品生产车间")

    def test_missing_values_still_use_only_delivery_sentence(self):
        self.assertEqual(delivery_voice(None, None), "已将货品放入目标车间")

    def test_forbidden_error_words_are_absent(self):
        speech = delivery_voice("大米", "食品加工车间")
        for forbidden in ("异常", "失败", "停止", "未找到", "安全"):
            self.assertNotIn(forbidden, speech)


if __name__ == "__main__":
    unittest.main()

