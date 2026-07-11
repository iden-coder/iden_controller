#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Standalone real-car test for the flow_end-style white-box parking."""

import time

import rospy

from factory_room_delivery_manager_parking_v5 import FlowEndStyleParkingManager


class WhiteBoxParkingTest(FlowEndStyleParkingManager):
    def wait_for_parking_inputs(self, timeout_s):
        start = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            image, image_time = self.snapshot_image()
            odom_xy, odom_yaw = self.snapshot_odom()
            with self.lock:
                scan_ready = bool(self.scan_ranges)
            image_ready = (image is not None and
                           time.time() - image_time < 1.0)
            if image_ready and scan_ready and odom_xy is not None and odom_yaw is not None:
                return True
            rospy.logwarn_throttle(
                1.0,
                "PARKING_TEST_WAIT image=%s scan=%s odom=%s",
                str(image_ready), str(scan_ready),
                str(odom_xy is not None and odom_yaw is not None))
            rate.sleep()
        return False

    def run_test(self):
        input_timeout = float(rospy.get_param("~input_timeout_s", 20.0))
        acquire_timeout = float(rospy.get_param("~acquire_timeout_s", 18.0))
        start_delay = float(rospy.get_param("~start_delay_s", 3.0))
        self.publish_state("WHITE_BOX_TEST_WAITING_INPUTS")
        self.publish_zero(20)
        if not self.wait_for_parking_inputs(input_timeout):
            rospy.logerr("WHITE_BOX_TEST_ABORTED required sensor data unavailable")
            self.publish_zero(30)
            return False

        rospy.logwarn(
            "WHITE_BOX_TEST_READY; robot remains stopped for %.1fs",
            start_delay)
        end = time.time() + max(0.0, start_delay)
        while not rospy.is_shutdown() and time.time() < end:
            self.publish_zero()
            rospy.sleep(0.05)

        square = self.acquire_square(timeout_s=acquire_timeout)
        if square is None:
            rospy.logerr(
                "WHITE_BOX_TEST_ABORTED complete two-rail frame not acquired")
            self.publish_zero(30)
            return False

        rospy.logwarn(
            "WHITE_BOX_TEST_FRAME_OK width=%.3fm near=%.3fm lateral=%.3fm heading=%.1fdeg",
            float(square["rail_width_m"]),
            float(square["near_edge_distance_m"]),
            float(square.get("raw_lateral_error_m", 0.0)),
            57.2957795 * float(square.get("heading_error_rad", 0.0)))
        ok = self.park_inside_square()
        self.publish_zero(30)
        if ok:
            self.publish_state("WHITE_BOX_TEST_SUCCESS")
            rospy.logwarn("WHITE_BOX_TEST_SUCCESS vehicle stopped in target geometry")
        else:
            self.publish_state("WHITE_BOX_TEST_FAILED")
            rospy.logerr("WHITE_BOX_TEST_FAILED vehicle stopped safely")
        return ok


if __name__ == "__main__":
    node = WhiteBoxParkingTest()
    try:
        rospy.sleep(1.0)
        node.run_test()
    finally:
        node.parking_close_mode = False
        node.publish_parking_mode()
        node.publish_zero(30)
        rospy.signal_shutdown("standalone white-box parking test finished")

