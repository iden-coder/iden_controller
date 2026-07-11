#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Correct camera-to-chassis signs for lateral and visual yaw control."""

import rospy

from factory_room_delivery_manager_parking_v8 import (
    CurvedFrameQuaternionParkingManager,
)


class CorrectedSignParkingManager(CurvedFrameQuaternionParkingManager):
    def parking_errors(self, result):
        lateral, heading = super(CorrectedSignParkingManager,
                                 self).parking_errors(result)
        corrected_lateral = -lateral
        corrected_heading = -heading
        lateral_deadband = float(rospy.get_param(
            "~parking_lateral_deadband_m", 0.015))
        heading_deadband = float(rospy.get_param(
            "~parking_visual_heading_deadband_deg", 1.0)) * 0.01745329252
        if abs(corrected_lateral) <= lateral_deadband:
            corrected_lateral = 0.0
        if abs(corrected_heading) <= heading_deadband:
            corrected_heading = 0.0
        rospy.logwarn_throttle(
            0.7,
            "PARKING_SIGN_CORRECT raw_control=(%.3fm,%.2fdeg) corrected=(%.3fm,%.2fdeg)",
            lateral, heading * 57.2957795,
            corrected_lateral, corrected_heading * 57.2957795)
        return corrected_lateral, corrected_heading


if __name__ == "__main__":
    node = CorrectedSignParkingManager()
    rospy.spin()
