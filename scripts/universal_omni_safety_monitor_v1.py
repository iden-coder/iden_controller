#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Enable directional holonomic safety in every mapped area."""

import rospy

from global_first_safety_monitor_omni_v1 import (
    DirectionalOmniSafetyMonitor,
)


class UniversalOmniSafetyMonitor(DirectionalOmniSafetyMonitor):
    def __init__(self):
        super(UniversalOmniSafetyMonitor, self).__init__()
        self.indoor_active = True
        rospy.logwarn(
            "UNIVERSAL_OMNI_SAFETY_ACTIVE room_boundaries_disabled=true")

    def cb_room_pose(self, _msg):
        self.indoor_active = True


if __name__ == "__main__":
    try:
        UniversalOmniSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
