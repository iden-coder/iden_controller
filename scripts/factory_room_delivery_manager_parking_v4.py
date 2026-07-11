#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Centerline parking plus heartbeat-gated close-wall safety mode."""

import rospy
from std_msgs.msg import Bool

from factory_room_delivery_manager_parking_v3 import StrictParkingFactoryRoomManager
from factory_room_vision_parking_v4 import CenterlineParkingDetector


class CenterlineParkingFactoryRoomManager(StrictParkingFactoryRoomManager):
    def __init__(self):
        super(CenterlineParkingFactoryRoomManager, self).__init__()
        self.square_detector = CenterlineParkingDetector()
        self.parking_close_mode = False
        topic = rospy.get_param(
            "~parking_close_mode_topic", "/factory/parking_close_mode")
        self.parking_mode_pub = rospy.Publisher(
            topic, Bool, queue_size=1, latch=False)
        self.parking_mode_timer = rospy.Timer(
            rospy.Duration(0.2), self.publish_parking_mode)

    def detect_square_once(self):
        result = super(CenterlineParkingFactoryRoomManager,
                       self).detect_square_once()
        if result and result.get("found"):
            rospy.logwarn_throttle(
                0.7,
                "PARKING_CENTERLINE lateral=%.3fm raw=%.3fm heading=%.1fdeg width=%.3fm near=%.3fm",
                float(result.get("lateral_error_m", 0.0)),
                float(result.get("raw_lateral_error_m", 0.0)),
                57.2957795 * float(result.get("heading_error_rad", 0.0)),
                float(result.get("rail_width_m", 0.0)),
                float(result.get("near_edge_distance_m", 0.0)))
        return result

    def publish_parking_mode(self, _event=None):
        self.parking_mode_pub.publish(Bool(data=self.parking_close_mode))

    def park_inside_square(self):
        self.parking_close_mode = True
        self.publish_parking_mode()
        rospy.logwarn("PARKING_CLOSE_SAFETY_MODE enabled")
        try:
            return super(CenterlineParkingFactoryRoomManager,
                         self).park_inside_square()
        finally:
            self.parking_close_mode = False
            self.publish_parking_mode()
            self.publish_zero(15)
            rospy.logwarn("PARKING_CLOSE_SAFETY_MODE disabled")


if __name__ == "__main__":
    node = CenterlineParkingFactoryRoomManager()
    rospy.spin()
