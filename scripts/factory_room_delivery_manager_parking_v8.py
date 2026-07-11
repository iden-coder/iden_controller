#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quaternion parking manager with fisheye-curved frame perception."""

import rospy

from factory_room_delivery_manager_parking_v7 import (
    QuaternionLockedParkingManager,
)
from factory_room_vision_parking_v6 import CurvedCrossbarParkingDetector


class CurvedFrameQuaternionParkingManager(QuaternionLockedParkingManager):
    def __init__(self):
        super(CurvedFrameQuaternionParkingManager, self).__init__()
        self.square_detector = CurvedCrossbarParkingDetector()

    def detect_square_once(self):
        result = super(CurvedFrameQuaternionParkingManager,
                       self).detect_square_once()
        if result:
            rospy.logwarn_throttle(
                0.7,
                "CURVED_FRAME found=%s curved=%s conf=%.2f L/R/H=%d/%d/%d y=%s",
                str(bool(result.get("found"))),
                str(bool(result.get("curved_crossbar"))),
                float(result.get("confidence", 0.0)),
                int(result.get("left_count", 0)),
                int(result.get("right_count", 0)),
                int(result.get("horizontal_count", 0)),
                str(result.get("curved_crossbar_y_px")))
        return result


if __name__ == "__main__":
    node = CurvedFrameQuaternionParkingManager()
    rospy.spin()

