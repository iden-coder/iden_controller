#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Corrected parking manager with adaptive far/close frame perception."""

import rospy

from factory_room_delivery_manager_parking_v9 import CorrectedSignParkingManager
from factory_room_vision_parking_v7 import AdaptiveRangeParkingDetector


class AdaptiveRangeParkingManager(CorrectedSignParkingManager):
    def __init__(self):
        super(AdaptiveRangeParkingManager, self).__init__()
        self.square_detector = AdaptiveRangeParkingDetector()

    def detect_square_once(self):
        result = super(AdaptiveRangeParkingManager, self).detect_square_once()
        if result:
            rospy.logwarn_throttle(
                0.7,
                "ADAPTIVE_FRAME mode=%s found=%s conf=%.2f L/R/H=%d/%d/%d crop=%s band=%s",
                str(result.get("range_mode", "unknown")),
                str(bool(result.get("found"))),
                float(result.get("confidence", 0.0)),
                int(result.get("left_count", 0)),
                int(result.get("right_count", 0)),
                int(result.get("horizontal_count", 0)),
                str(result.get("close_crop_top_px")),
                str(result.get("close_crossbar_y_px")))
        return result


if __name__ == "__main__":
    node = AdaptiveRangeParkingManager()
    rospy.spin()

