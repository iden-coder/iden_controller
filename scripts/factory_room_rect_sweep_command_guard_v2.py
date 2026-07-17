#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Fail-closed TF handling for the room rectangular command guard."""

import math

import rospy
import tf

from factory_room_rect_sweep_command_guard_v1 import (
    FactoryRoomRectSweepCommandGuard,
)


class FactoryRoomRectSweepCommandGuardV2(
        FactoryRoomRectSweepCommandGuard):
    def __init__(self):
        self.scan_transform_valid = False
        super(FactoryRoomRectSweepCommandGuardV2, self).__init__()
        rospy.logwarn("ROOM_RECT_COMMAND_GUARD_V2 fail_closed_tf=true")

    def _scan_points_in_base(self, msg):
        frame_id = msg.header.frame_id or self.base_frame
        tx = 0.0
        ty = 0.0
        yaw = 0.0
        if frame_id != self.base_frame:
            try:
                trans, rotation = self.tf_listener.lookupTransform(
                    self.base_frame, frame_id, rospy.Time(0))
                tx, ty = trans[0], trans[1]
                yaw = tf.transformations.euler_from_quaternion(rotation)[2]
            except Exception as exc:
                self.scan_transform_valid = False
                rospy.logerr_throttle(
                    1.0, "ROOM_RECT_TF_INVALID %s -> %s: %s",
                    frame_id, self.base_frame, exc)
                return []

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        points = []
        for index in range(0, len(msg.ranges), self.scan_stride):
            distance = msg.ranges[index]
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            if distance > self.scan_range:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)
            points.append((tx + cos_yaw * lx - sin_yaw * ly,
                           ty + sin_yaw * lx + cos_yaw * ly))
        self.scan_transform_valid = True
        return points

    def scan_callback(self, msg):
        points = self._scan_points_in_base(msg)
        with self.lock:
            self.scan_points = points
            if self.scan_transform_valid:
                self.scan_stamp = rospy.Time.now()
            else:
                self.scan_stamp = rospy.Time(0)


if __name__ == "__main__":
    rospy.init_node("factory_room_rect_sweep_command_guard_v2")
    FactoryRoomRectSweepCommandGuardV2()
    rospy.spin()
