#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Flow-end parking manager using aspect-ratio-safe frame perception."""

import rospy

from factory_room_delivery_manager_parking_v5 import FlowEndStyleParkingManager
from factory_room_vision_parking_v5 import PerspectiveParkingDetector


class AspectSafeParkingManager(FlowEndStyleParkingManager):
    def __init__(self):
        super(AspectSafeParkingManager, self).__init__()
        self.square_detector = PerspectiveParkingDetector()

    def detect_square_once(self):
        result = super(AspectSafeParkingManager, self).detect_square_once()
        if result:
            rospy.logwarn_throttle(
                0.7,
                "WHITE_BOX_RAW_GEOMETRY found=%s conf=%.2f lines=L%d/R%d/H%d source=%dx%d",
                str(bool(result.get("found"))),
                float(result.get("confidence", 0.0)),
                int(result.get("left_count", 0)),
                int(result.get("right_count", 0)),
                int(result.get("horizontal_count", 0)),
                int(result.get("source_width", 0)),
                int(result.get("source_height", 0)))
        return result


if __name__ == "__main__":
    node = AspectSafeParkingManager()
    rospy.spin()

