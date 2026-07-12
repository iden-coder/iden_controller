#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""V4: hysteretic sign lock and practical, non-oscillating tolerances."""

import rospy
from std_msgs.msg import Bool

from factory_sign_center_parking_test_v1 import clamp
from factory_sign_center_parking_test_v3 import EffectiveHeadingParkingTest


class PracticalToleranceParkingTest(EffectiveHeadingParkingTest):
    def __init__(self):
        super(PracticalToleranceParkingTest, self).__init__()
        self.frame_disable_near_sign_px = float(rospy.get_param(
            "~frame_disable_near_sign_px", 48.0))

    def fused_center_error(self, ocr):
        width = float(ocr["image_width"])
        sign_center = float(ocr["filtered_center_x"])
        sign_error_px = sign_center - width * self.optical_center_x_ratio
        sign_normalized = sign_error_px / max(1.0, 0.5 * width)

        # The workshop sign is the primary reference. Once it is reasonably
        # close to center, a biased/partial floor-frame detection must not pull
        # the vehicle back across the target and create an endless oscillation.
        if abs(sign_error_px) <= self.frame_disable_near_sign_px:
            self.detect_frame(sign_center=sign_center)
            return sign_normalized, "sign_lock_band", None

        fused, mode, frame_normalized = super(
            PracticalToleranceParkingTest, self).fused_center_error(ocr)
        return fused, mode, frame_normalized


if __name__ == "__main__":
    node = PracticalToleranceParkingTest()
    try:
        rospy.sleep(1.0)
        node.success = node.run()
        node.finished = True
        node.publish_zero(30)
        if not node.success:
            rospy.logerr("SIGN_PARK_V4_TEST_FAILED; vehicle remains stopped")
        rospy.logwarn("SIGN_PARK_V4_TEST_FINISHED; node holds zero")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            node.publish_zero()
            node.parking_pub.publish(Bool(data=True))
            rospy.logwarn_throttle(
                3.0, "SIGN_PARK_V4 finished=%s success=%s HOLDING_ZERO",
                str(node.finished), str(node.success))
            rate.sleep()
    except rospy.ROSInterruptException:
        pass

