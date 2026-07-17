#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Keep the proven continuous QR scanner and its camera alive after success."""

import rospy

from xunfei2026_continuous_qr_hybrid_v1 import ContinuousQRHybrid


def main():
    rospy.init_node("xunfei2026_persistent_qr_hybrid")
    node = ContinuousQRHybrid()
    try:
        code = node.run()
        rospy.logwarn("XUNFEI2026_PERSISTENT_QR result=%d", code)
        if code == 0 and not rospy.is_shutdown():
            rospy.logwarn(
                "XUNFEI2026_QR_INFRASTRUCTURE_HELD camera_alive=true cmd_owner=false")
            rospy.spin()
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
