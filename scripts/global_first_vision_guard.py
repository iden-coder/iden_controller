#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import SetBool, SetBoolResponse


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class GlobalFirstVisionGuard:
    def __init__(self):
        rospy.init_node("global_first_vision_guard")

        self.enabled = rospy.get_param("~enabled", True)
        self.cmd_vel_in_topic = rospy.get_param("~cmd_vel_in_topic", "/cmd_vel_graph_raw")
        self.cmd_vel_out_topic = rospy.get_param("~cmd_vel_out_topic", "/cmd_vel_raw")
        self.image_topic = rospy.get_param("~image_topic", "/ucar_camera/image_raw")
        self.pass_through_on_no_image = rospy.get_param("~pass_through_on_no_image", True)

        self.image_timeout_s = rospy.get_param("~image_timeout_s", 0.8)
        self.roi_y_min_frac = rospy.get_param("~roi_y_min_frac", 0.58)
        self.roi_y_max_frac = rospy.get_param("~roi_y_max_frac", 0.95)
        self.sample_step_px = int(rospy.get_param("~sample_step_px", 6))
        self.bright_threshold = int(rospy.get_param("~bright_threshold", 205))
        self.min_confidence = rospy.get_param("~min_confidence", 0.28)
        self.center_deadband_frac = rospy.get_param("~center_deadband_frac", 0.055)
        self.max_correction_wz = rospy.get_param("~max_correction_wz", 0.10)
        self.max_abs_wz = rospy.get_param("~max_abs_wz", 0.72)
        self.offset_slow_start_frac = rospy.get_param("~offset_slow_start_frac", 0.18)
        self.offset_stop_frac = rospy.get_param("~offset_stop_frac", 0.46)
        self.min_slow_ratio = rospy.get_param("~min_slow_ratio", 0.50)
        self.allow_visual_stop = rospy.get_param("~allow_visual_stop", False)
        self.center_block_width_frac = rospy.get_param("~center_block_width_frac", 0.20)
        self.center_block_ratio = rospy.get_param("~center_block_ratio", 0.36)

        self.last_image_time = rospy.Time(0)
        self.last_offset = 0.0
        self.last_confidence = 0.0
        self.last_center_block = 0.0
        self.image_seen = False

        self.pub_cmd = rospy.Publisher(self.cmd_vel_out_topic, Twist, queue_size=1)
        self.pub_status = rospy.Publisher("~status", String, queue_size=3, latch=True)
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.cmd_vel_in_topic, Twist, self.cb_cmd, queue_size=1)
        rospy.Service("~toggle", SetBool, self.cb_toggle)

        rospy.logwarn(
            "GlobalFirstVisionGuard started: image=%s in=%s out=%s enabled=%s",
            self.image_topic, self.cmd_vel_in_topic, self.cmd_vel_out_topic,
            str(self.enabled))

    def cb_toggle(self, req):
        self.enabled = req.data
        return SetBoolResponse(success=True, message="OK")

    def decode_image(self, msg):
        enc = (msg.encoding or "").lower()
        if enc in ("rgb8", "bgr8"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            expected = msg.height * msg.width * 3
            if arr.size < expected:
                return None
            arr = arr[:expected].reshape((msg.height, msg.width, 3))
            if enc == "bgr8":
                arr = arr[:, :, ::-1]
            return arr
        if enc in ("mono8", "8uc1"):
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            expected = msg.height * msg.width
            if arr.size < expected:
                return None
            gray = arr[:expected].reshape((msg.height, msg.width))
            return np.dstack((gray, gray, gray))
        return None

    def cb_image(self, msg):
        image = self.decode_image(msg)
        if image is None or image.shape[0] < 20 or image.shape[1] < 20:
            return

        h, w = image.shape[:2]
        y0 = int(clamp(self.roi_y_min_frac, 0.0, 0.98) * h)
        y1 = int(clamp(self.roi_y_max_frac, self.roi_y_min_frac + 0.01, 1.0) * h)
        roi = image[y0:y1, :, :]
        if roi.size == 0:
            return

        bright = np.logical_and.reduce((
            roi[:, :, 0] >= self.bright_threshold,
            roi[:, :, 1] >= self.bright_threshold,
            roi[:, :, 2] >= self.bright_threshold))

        mid = w // 2
        centers = []
        valid_rows = 0
        step = max(1, self.sample_step_px)
        for y in range(bright.shape[0] - 1, -1, -step):
            xs = np.flatnonzero(bright[y])
            if xs.size == 0:
                continue
            left = xs[xs < mid]
            right = xs[xs > mid]
            center = None
            if left.size and right.size:
                center = 0.5 * (left[-1] + right[0])
            elif left.size:
                center = 0.5 * (left[-1] + (w - 1))
            elif right.size:
                center = 0.5 * right[0]
            if center is not None:
                centers.append(center)
                valid_rows += 1

        sampled_rows = max(1, int(math.ceil(float(bright.shape[0]) / float(step))))
        confidence = float(valid_rows) / float(sampled_rows)
        if centers:
            avg_center = float(np.median(np.asarray(centers)))
            offset = (avg_center - float(mid)) / max(float(mid), 1.0)
        else:
            offset = 0.0

        block_half = int(max(2, w * self.center_block_width_frac * 0.5))
        center_strip = bright[:, max(0, mid - block_half):min(w, mid + block_half)]
        center_block = float(np.mean(center_strip)) if center_strip.size else 0.0

        self.last_offset = offset
        self.last_confidence = confidence
        self.last_center_block = center_block
        self.last_image_time = rospy.Time.now()
        self.image_seen = True

    def image_fresh(self):
        return (self.image_seen and
                (rospy.Time.now() - self.last_image_time).to_sec() <= self.image_timeout_s)

    def publish_status(self, text):
        msg = String()
        msg.data = text
        self.pub_status.publish(msg)

    def cb_cmd(self, msg):
        out = Twist()
        out.linear.x = msg.linear.x
        out.linear.y = msg.linear.y
        out.linear.z = msg.linear.z
        out.angular.x = msg.angular.x
        out.angular.y = msg.angular.y
        out.angular.z = msg.angular.z

        action = "PASS"
        if not self.enabled:
            action = "DISABLED"
        elif not self.image_fresh():
            action = "NO_IMAGE" if not self.pass_through_on_no_image else "NO_IMAGE_PASS"
            if not self.pass_through_on_no_image:
                out.linear.x = 0.0
                out.angular.z = 0.0
        elif self.last_confidence < self.min_confidence:
            action = "LOW_CONF_PASS"
        else:
            offset = self.last_offset
            abs_offset = abs(offset)
            deadband = self.center_deadband_frac
            if abs_offset > deadband:
                effective = (abs_offset - deadband) / max(1.0 - deadband, 1.0e-6)
                correction = -math.copysign(
                    min(self.max_correction_wz, self.max_correction_wz * effective * 1.6),
                    offset)
                out.angular.z = clamp(out.angular.z + correction,
                                      -self.max_abs_wz, self.max_abs_wz)
                action = "CORRECT"

            if out.linear.x > 0.0 and abs_offset > self.offset_slow_start_frac:
                span = max(self.offset_stop_frac - self.offset_slow_start_frac, 1.0e-6)
                t = clamp((abs_offset - self.offset_slow_start_frac) / span, 0.0, 1.0)
                ratio = 1.0 - (1.0 - self.min_slow_ratio) * t
                out.linear.x *= ratio
                action = action + "_SLOW" if action != "PASS" else "SLOW"

            if (self.allow_visual_stop and out.linear.x > 0.0 and
                    self.last_center_block >= self.center_block_ratio):
                out.linear.x = 0.0
                action = "VISUAL_CENTER_BLOCK"

        self.pub_cmd.publish(out)
        if action not in ("PASS", "DISABLED", "NO_IMAGE_PASS", "LOW_CONF_PASS"):
            rospy.logwarn_throttle(
                0.7,
                "GlobalFirstVisionGuard: %s | offset=%.2f conf=%.2f block=%.2f | in=(%.3f, %.3f) out=(%.3f, %.3f)",
                action, self.last_offset, self.last_confidence, self.last_center_block,
                msg.linear.x, msg.angular.z, out.linear.x, out.angular.z)
        self.publish_status(
            "%s offset=%.3f confidence=%.3f center_block=%.3f" %
            (action, self.last_offset, self.last_confidence, self.last_center_block))

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        GlobalFirstVisionGuard().run()
    except rospy.ROSInterruptException:
        pass
