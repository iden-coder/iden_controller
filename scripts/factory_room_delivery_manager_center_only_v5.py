#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Centerline delivery manager with a bounded 180-degree workshop scan."""

import math

import rospy

from factory_room_delivery_manager_center_only_v2 import (
    NoWhiteFrameDeliveryManager,
)


class HalfTurnWorkshopDeliveryManager(NoWhiteFrameDeliveryManager):
    def __init__(self):
        super(HalfTurnWorkshopDeliveryManager, self).__init__()
        self.workshop_scan_arc_deg = float(rospy.get_param(
            "~workshop_scan_arc_deg", 180.0))
        rospy.logwarn(
            "WORKSHOP_SCAN_ARC_CONFIGURED arc=%.1fdeg step=%.1fdeg",
            self.workshop_scan_arc_deg, self.scan_step_deg)

    def guarded_lateral(self, requested, snapshot):
        clearance = (snapshot["left_clear"] if requested > 0.0
                     else snapshot["right_clear"])
        if clearance <= self.lateral_hard_clearance:
            return 0.0, clearance, "hard-stop"
        # Parking deliberately moves alongside the target wall. Above the hard
        # swept-body stop, preserve the controller's minimum effective speed so
        # it cannot decay below the chassis motion deadband.
        return requested, clearance, "parking-direct"

    def scan_for_target_at_current_pose(self, point_name):
        self.publish_state(
            "SCANNING_WORKSHOP", point=point_name,
            target=self.target_warehouse,
            maximum_arc_deg=self.workshop_scan_arc_deg)
        self.target_found.clear()
        self.ocr_control("reset")
        self.ocr_control("enable")
        rospy.sleep(1.5)
        if self.target_found.is_set():
            self.publish_zero(3)
            return True

        step_deg = max(self.scan_step_deg, 10.0)
        steps = max(1, int(math.ceil(self.workshop_scan_arc_deg / step_deg)))
        step_radians = math.radians(
            self.workshop_scan_arc_deg / float(steps))
        for step in range(steps):
            if rospy.is_shutdown():
                return False
            if self.target_found.is_set():
                self.publish_zero(3)
                return True
            if not self.rotate_relative(step_radians):
                if self.target_found.is_set():
                    return True
                rospy.logwarn(
                    "180deg scan rotation blocked at %s step=%d/%d",
                    point_name, step + 1, steps)
                break
            rospy.sleep(self.scan_settle_s)
        rospy.logwarn(
            "WORKSHOP_HALF_SCAN_COMPLETE point=%s swept=%.1fdeg found=%s",
            point_name, self.workshop_scan_arc_deg,
            str(self.target_found.is_set()))
        return self.target_found.is_set()


if __name__ == "__main__":
    HalfTurnWorkshopDeliveryManager().run()
