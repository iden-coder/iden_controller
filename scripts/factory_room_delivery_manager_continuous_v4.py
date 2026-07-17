#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conflict-free room delivery with mandatory error speech and watchdog."""

import threading
import time

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import String

from factory_room_delivery_manager_continuous_v3 import (
    ConflictFreeContinuousParkingManager,
)
from factory_room_voice_fail_safe_core_v4 import (
    error_voice_text,
    should_fire_watchdog,
)


class VoiceFailSafeContinuousParkingManager(
        ConflictFreeContinuousParkingManager):
    def __init__(self):
        self._room_watchdog_lock = threading.RLock()
        self._room_watchdog_active = False
        self._room_watchdog_aborted = False
        self._room_watchdog_deadline = None
        self._room_watchdog_phase = "idle"
        self._room_error_voice_sent = False
        self._room_pending_error_reason = None
        self._room_watchdog_timer = None
        # Parent initialization installs the task subscriber. Keep complete
        # defaults ready in case a latched task arrives before super() returns.
        self.room_hard_timeout = 170.0
        self.room_parking_timeout = 60.0
        self.room_error_voice = error_voice_text()
        super(VoiceFailSafeContinuousParkingManager, self).__init__()
        self.room_hard_timeout = float(rospy.get_param(
            "~room_hard_timeout_s", 170.0))
        self.room_parking_timeout = float(rospy.get_param(
            "~room_parking_timeout_s", 60.0))
        self.room_error_voice = str(rospy.get_param(
            "~room_error_voice_text", error_voice_text()))
        self._room_watchdog_timer = rospy.Timer(
            rospy.Duration(0.20), self._room_watchdog_tick)
        rospy.logwarn(
            "ROOM_VOICE_FAIL_SAFE_V4_READY hard_timeout=%.1fs "
            "parking_timeout=%.1fs error_voice_mandatory=true",
            self.room_hard_timeout, self.room_parking_timeout)

    def refresh_room_watchdog(self, phase, timeout_s=None):
        """Give a newly entered finite mission phase its own time budget."""
        timeout = (self.room_hard_timeout if timeout_s is None else
                   float(timeout_s))
        with self._room_watchdog_lock:
            if self._room_watchdog_aborted:
                return False
            self._room_watchdog_active = True
            self._room_watchdog_phase = str(phase)
            self._room_watchdog_deadline = (
                time.time() + max(10.0, timeout))
        rospy.logwarn(
            "ROOM_WATCHDOG_PHASE phase=%s timeout=%.1fs deadline_refreshed=true",
            phase, timeout)
        return True

    def _speak_error_once(self, reason):
        with self._room_watchdog_lock:
            if self._room_error_voice_sent:
                return False
            self._room_pending_error_reason = reason
            publisher = getattr(self, "tts_pub", None)
            if publisher is None:
                rospy.logerr(
                    "ROOM_ERROR_TTS_NOT_READY reason=%s; will retry", reason)
                return False
            try:
                publisher.publish(String(data=self.room_error_voice))
            except Exception as exc:
                rospy.logerr(
                    "ROOM_ERROR_TTS_RETRY reason=%s publish_error=%s",
                    reason, exc)
                return False
            self._room_error_voice_sent = True
            self._room_pending_error_reason = None
        rospy.logerr(
            "ROOM_ERROR_TTS_PUBLISHED text=%s reason=%s",
            self.room_error_voice, reason)
        return True

    def _stop_all_room_motion(self, reason):
        with self._room_watchdog_lock:
            self._room_watchdog_aborted = True
        try:
            self.request_search_stop()
        except Exception:
            pass
        try:
            self.move_client.cancel_all_goals()
        except Exception:
            pass
        try:
            self.set_parking_mode(False)
        except Exception:
            pass
        cmd_publisher = getattr(self, "cmd_pub", None)
        if cmd_publisher is not None:
            for _ in range(12):
                try:
                    cmd_publisher.publish(Twist())
                    rospy.sleep(0.01)
                except Exception:
                    break
        rospy.logerr("ROOM_ALL_MOTION_STOPPED reason=%s", reason)

    def _room_watchdog_tick(self, _event=None):
        now = time.time()
        with self._room_watchdog_lock:
            fire = should_fire_watchdog(
                self._room_watchdog_active,
                self._room_watchdog_aborted,
                self._room_watchdog_deadline,
                now)
            if fire:
                self._room_watchdog_aborted = True
            pending_reason = self._room_pending_error_reason
            phase = self._room_watchdog_phase
        if not fire:
            if pending_reason:
                self._speak_error_once(pending_reason)
            return
        reason = "大房间任务阶段超过安全时限：{}".format(phase)
        self._stop_all_room_motion(reason)
        self._speak_error_once(reason)
        self.publish_state(
            "ROOM_WATCHDOG_TIMEOUT",
            status="error", reason=reason,
            voice=self.room_error_voice)

    def publish_center_command(self, x=0.0, y=0.0, wz=0.0):
        with self._room_watchdog_lock:
            aborted = self._room_watchdog_aborted
        if aborted:
            self.cmd_pub.publish(Twist())
            return
        super(VoiceFailSafeContinuousParkingManager,
              self).publish_center_command(x=x, y=y, wz=wz)

    def fail(self, reason):
        self._stop_all_room_motion(reason)
        self._speak_error_once(reason)
        with self._room_watchdog_lock:
            self._room_watchdog_active = False
            self._room_watchdog_phase = "complete"
        return super(VoiceFailSafeContinuousParkingManager, self).fail(reason)

    def announce_search_failure(self, reason):
        with self._room_watchdog_lock:
            self._room_watchdog_active = False
            self._room_watchdog_phase = "complete"
        try:
            result = super(VoiceFailSafeContinuousParkingManager,
                           self).announce_search_failure(reason)
        except Exception as exc:
            fallback_reason = "{}; no-target announcement error: {}".format(
                reason, exc)
            self._stop_all_room_motion(fallback_reason)
            self._speak_error_once(fallback_reason)
            return None
        with self._room_watchdog_lock:
            # The inherited method has published its specific no-target voice.
            self._room_error_voice_sent = True
            self._room_pending_error_reason = None
        return result

    def publish_success(self):
        with self._room_watchdog_lock:
            self._room_watchdog_active = False
            self._room_watchdog_phase = "complete"
        return super(VoiceFailSafeContinuousParkingManager,
                     self).publish_success()

    def mission_thread(self):
        with self._room_watchdog_lock:
            self._room_watchdog_active = True
            self._room_watchdog_aborted = False
            self._room_watchdog_phase = "search"
            self._room_error_voice_sent = False
            self._room_pending_error_reason = None
            self._room_watchdog_deadline = (
                time.time() + max(30.0, self.room_hard_timeout))
        try:
            return super(VoiceFailSafeContinuousParkingManager,
                         self).mission_thread()
        finally:
            with self._room_watchdog_lock:
                self._room_watchdog_active = False
                self._room_watchdog_phase = "idle"


if __name__ == "__main__":
    VoiceFailSafeContinuousParkingManager().run()
