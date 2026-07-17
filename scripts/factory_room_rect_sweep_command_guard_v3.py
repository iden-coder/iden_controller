#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V2 rectangular guard with race-free ROS shutdown handling."""

import rospy

from factory_room_rect_sweep_command_guard_v2 import (
    FactoryRoomRectSweepCommandGuardV2,
)


class ShutdownSafeRectSweepGuard(FactoryRoomRectSweepCommandGuardV2):
    def __init__(self):
        self.guard_shutting_down = False
        super(ShutdownSafeRectSweepGuard, self).__init__()
        rospy.logwarn("ROOM_RECT_COMMAND_GUARD_V3 shutdown_safe=true")

    def command_callback(self, command):
        if self.guard_shutting_down or rospy.is_shutdown():
            return
        try:
            super(ShutdownSafeRectSweepGuard, self).command_callback(command)
        except rospy.ROSException as exc:
            if not (self.guard_shutting_down or rospy.is_shutdown()):
                raise
            rospy.logdebug("late guard command ignored during shutdown: %s", exc)

    def watchdog(self, event):
        if self.guard_shutting_down or rospy.is_shutdown():
            return
        try:
            super(ShutdownSafeRectSweepGuard, self).watchdog(event)
        except rospy.ROSException as exc:
            if not (self.guard_shutting_down or rospy.is_shutdown()):
                raise
            rospy.logdebug("late guard watchdog ignored during shutdown: %s", exc)

    def shutdown(self):
        self.guard_shutting_down = True
        try:
            super(ShutdownSafeRectSweepGuard, self).shutdown()
        except rospy.ROSException:
            pass


if __name__ == "__main__":
    rospy.init_node("factory_room_rect_sweep_command_guard_v3")
    ShutdownSafeRectSweepGuard()
    rospy.spin()
