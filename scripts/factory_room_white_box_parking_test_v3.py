#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone V3 precise and faster white-box parking test."""

import math
import time

import rospy

from factory_room_delivery_manager_parking_v7 import QuaternionLockedParkingManager


class WhiteBoxParkingTestV3(QuaternionLockedParkingManager):
    def wait_for_parking_inputs(self, timeout_s):
        start = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            image, image_time = self.snapshot_image()
            odom_xy, odom_yaw = self.snapshot_odom()
            with self.lock:
                scan_ready = bool(self.scan_ranges)
            ready = (image is not None and time.time() - image_time < 1.0 and
                     scan_ready and odom_xy is not None and odom_yaw is not None)
            if ready:
                return True
            rospy.logwarn_throttle(1.0, "PARKING_V3_WAIT sensors not ready")
            rate.sleep()
        return False

    def run_test(self):
        self.publish_state("WHITE_BOX_TEST_V3_WAITING_INPUTS")
        self.publish_zero(20)
        if not self.wait_for_parking_inputs(float(rospy.get_param(
                "~input_timeout_s", 20.0))):
            rospy.logerr("WHITE_BOX_TEST_V3_ABORTED sensors unavailable")
            return False
        delay = float(rospy.get_param("~start_delay_s", 3.0))
        rospy.logwarn("WHITE_BOX_TEST_V3_READY; stopped for %.1fs", delay)
        end = time.time() + delay
        while not rospy.is_shutdown() and time.time() < end:
            self.publish_zero()
            rospy.sleep(0.05)
        square = self.acquire_square(timeout_s=float(rospy.get_param(
            "~acquire_timeout_s", 18.0)))
        if square is None:
            rospy.logerr("WHITE_BOX_TEST_V3_ABORTED frame not acquired")
            return False
        rospy.logwarn(
            "WHITE_BOX_TEST_V3_FRAME_OK conf=%.2f off=%.3fm heading=%.2fdeg",
            square["confidence"], square["raw_lateral_error_m"],
            math.degrees(square["heading_error_rad"]))
        ok = self.park_inside_square()
        self.publish_zero(30)
        if ok:
            self.publish_state("WHITE_BOX_TEST_V3_SUCCESS")
            rospy.logwarn("WHITE_BOX_TEST_V3_SUCCESS")
        else:
            self.publish_state("WHITE_BOX_TEST_V3_FAILED")
            rospy.logerr("WHITE_BOX_TEST_V3_FAILED; vehicle held stopped")
        return ok


if __name__ == "__main__":
    node = WhiteBoxParkingTestV3()
    try:
        rospy.sleep(1.0)
        node.run_test()
    finally:
        node.parking_close_mode = False
        node.publish_parking_mode()
        node.publish_zero(40)
        rospy.signal_shutdown("white-box parking test v3 finished")
