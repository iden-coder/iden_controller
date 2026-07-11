#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone safe/fast white-box parking test V4."""

import math
import time

import rospy

from factory_room_delivery_manager_parking_v8 import (
    CurvedFrameQuaternionParkingManager,
)


class WhiteBoxParkingTestV4(CurvedFrameQuaternionParkingManager):
    def wait_for_inputs(self, timeout_s):
        start = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            image, image_time = self.snapshot_image()
            odom_xy, odom_yaw = self.snapshot_odom()
            with self.lock:
                scan_ready = bool(self.scan_ranges)
            if (image is not None and time.time() - image_time < 1.0 and
                    scan_ready and odom_xy is not None and odom_yaw is not None):
                return True
            rospy.logwarn_throttle(1.0, "PARKING_V4_WAIT sensors not ready")
            rate.sleep()
        return False

    def run_test(self):
        self.publish_state("WHITE_BOX_TEST_V4_WAITING_INPUTS")
        self.publish_zero(20)
        if not self.wait_for_inputs(float(rospy.get_param(
                "~input_timeout_s", 20.0))):
            rospy.logerr("WHITE_BOX_TEST_V4_ABORTED sensors unavailable")
            return False
        delay = float(rospy.get_param("~start_delay_s", 3.0))
        rospy.logwarn("WHITE_BOX_TEST_V4_READY; stopped for %.1fs", delay)
        end = time.time() + delay
        while not rospy.is_shutdown() and time.time() < end:
            self.publish_zero()
            rospy.sleep(0.05)
        square = self.acquire_square(timeout_s=float(rospy.get_param(
            "~acquire_timeout_s", 15.0)))
        if square is None:
            rospy.logerr("WHITE_BOX_TEST_V4_ABORTED frame not acquired")
            return False
        rospy.logwarn(
            "WHITE_BOX_TEST_V4_FRAME_OK curved=%s conf=%.2f off=%.3fm heading=%.2fdeg",
            str(square.get("curved_crossbar")), square["confidence"],
            square["raw_lateral_error_m"],
            math.degrees(square["heading_error_rad"]))
        ok = self.park_inside_square()
        self.publish_zero(40)
        if ok:
            self.publish_state("WHITE_BOX_TEST_V4_SUCCESS")
            rospy.logwarn("WHITE_BOX_TEST_V4_SUCCESS")
        else:
            self.publish_state("WHITE_BOX_TEST_V4_FAILED")
            rospy.logerr("WHITE_BOX_TEST_V4_FAILED; vehicle held stopped")
        return ok


if __name__ == "__main__":
    node = WhiteBoxParkingTestV4()
    try:
        rospy.sleep(1.0)
        node.run_test()
    finally:
        node.parking_close_mode = False
        node.publish_parking_mode()
        node.publish_zero(50)
        rospy.signal_shutdown("white-box parking test v4 finished")

