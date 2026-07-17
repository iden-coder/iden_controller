#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Persistent room obstacle map layered on the V1 front-first action."""

import rospy

from factory_room_front_first_stable_action_v1 import (
    FactoryRoomFrontFirstStableActionBridge,
)
from factory_room_global_first_action_bridge import GlobalFirstActionBridge


class FactoryRoomFrontFirstStableActionBridgeV2(
        FactoryRoomFrontFirstStableActionBridge):
    def __init__(self):
        super(FactoryRoomFrontFirstStableActionBridgeV2, self).__init__()
        rospy.logwarn(
            "ROOM_STABLE_V2_READY persistent_tracks=true clear_preserves=true "
            "ttl=%.1fs", self.confirmed_ttl_s)

    def clear_dynamic_map(self, request):
        # Manager retries need a fresh path, not amnesia about confirmed cones.
        response = GlobalFirstActionBridge.clear_dynamic_map(self, request)
        self.dynamic_layer_signature = ()
        self.dynamic_layer_dirty = True
        if self.indoor_active and self.grid is not None:
            self._apply_dynamic_layer()
        rospy.logwarn(
            "ROOM_STABLE_V2_REPLAN_RESET tracks_preserved=%d",
            len(self._confirmed_tracks()))
        return response


if __name__ == "__main__":
    FactoryRoomFrontFirstStableActionBridgeV2().spin()
