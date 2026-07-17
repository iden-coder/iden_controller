#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

from factory_room_voice_fail_safe_core_v4 import (
    error_voice_text,
    should_fire_watchdog,
)


class VoiceFailSafeCoreTest(unittest.TestCase):
    def test_active_expired_watchdog_fires(self):
        self.assertTrue(should_fire_watchdog(True, False, 10.0, 10.0))

    def test_inactive_or_aborted_watchdog_does_not_fire(self):
        self.assertFalse(should_fire_watchdog(False, False, 10.0, 20.0))
        self.assertFalse(should_fire_watchdog(True, True, 10.0, 20.0))

    def test_watchdog_waits_for_deadline(self):
        self.assertFalse(should_fire_watchdog(True, False, 10.0, 9.9))

    def test_error_speech_never_claims_success(self):
        speech = error_voice_text("parking failed")
        self.assertIn("安全停止", speech)
        self.assertNotIn("已将", speech)


if __name__ == "__main__":
    unittest.main()
