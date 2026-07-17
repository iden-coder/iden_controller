#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Large-room manager whose only spoken output is the delivery sentence."""

import rospy

from factory_room_delivery_manager_continuous_v4 import (
    VoiceFailSafeContinuousParkingManager,
)
from factory_room_delivery_voice_core_v5 import delivery_voice


class DeliveryOnlyVoiceContinuousParkingManager(
        VoiceFailSafeContinuousParkingManager):
    def __init__(self):
        super(DeliveryOnlyVoiceContinuousParkingManager, self).__init__()
        self.room_error_voice = self._delivery_voice()
        rospy.logwarn(
            "ROOM_DELIVERY_ONLY_VOICE_V5_READY "
            "allowed_format=delivery_sentence_only")

    def _delivery_voice(self):
        return delivery_voice(
            getattr(self, "target_item", None),
            getattr(self, "target_warehouse", None))

    def _speak_error_once(self, reason):
        # Keep the true reason in logs/state, never in audible output.
        self.room_error_voice = self._delivery_voice()
        return super(DeliveryOnlyVoiceContinuousParkingManager,
                     self)._speak_error_once(reason)

    def announce_search_failure(self, reason):
        # Bypass the inherited "not found" sentence. fail() preserves the
        # internal error result while v5 constrains the spoken text.
        rospy.logerr(
            "ROOM_SEARCH_FAILED_DELIVERY_VOICE_ONLY reason=%s", reason)
        return self.fail(reason)

    def mission_thread(self):
        self.room_error_voice = self._delivery_voice()
        return super(DeliveryOnlyVoiceContinuousParkingManager,
                     self).mission_thread()


if __name__ == "__main__":
    DeliveryOnlyVoiceContinuousParkingManager().run()

