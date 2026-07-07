#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import os
import subprocess
import sys
import threading

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qr_room_spin_scan import ZBarScanner, clamp, norm_angle  # noqa: E402


class GlobalFirstNavThenQR(object):
    def __init__(self):
        rospy.init_node("global_first_nav_qr_room_scan")

        self.nav_status_topic = rospy.get_param(
            "~nav_status_topic", "/global_first_graph_nav/status")
        self.nav_node_name = rospy.get_param(
            "~nav_node_name", "/global_first_graph_nav")
        self.goal_status_substring = rospy.get_param(
            "~goal_status_substring", "goal reached").lower()
        self.start_immediately = bool(rospy.get_param("~start_immediately", False))
        self.kill_nav_on_goal = bool(rospy.get_param("~kill_nav_on_goal", True))
        self.nav_wait_timeout_s = float(rospy.get_param("~nav_wait_timeout_s", 600.0))
        self.after_nav_settle_s = float(rospy.get_param("~after_nav_settle_s", 0.6))

        self.image_topic = rospy.get_param("~image_topic", "/ucar_camera/image_raw")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel_raw")
        self.result_topic = rospy.get_param(
            "~result_topic", "/qr_room_scan_results")
        self.wall_count = int(rospy.get_param("~wall_count", 4))
        self.target_qr_count = int(rospy.get_param("~target_qr_count", 3))
        self.scan_per_wall_s = float(rospy.get_param("~scan_per_wall_s", 2.8))
        self.first_wall_scan_s = float(rospy.get_param(
            "~first_wall_scan_s", max(self.scan_per_wall_s, 5.0)))
        self.settle_s = float(rospy.get_param("~settle_s", 0.45))
        self.first_wall_settle_s = float(rospy.get_param(
            "~first_wall_settle_s", max(self.settle_s, 1.2)))
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 12.0))
        self.angular_speed = abs(float(rospy.get_param("~angular_speed", 0.52)))
        self.turn_tolerance_deg = float(rospy.get_param("~turn_tolerance_deg", 3.0))
        self.turn_timeout_s = float(rospy.get_param("~turn_timeout_s", 12.0))
        self.use_odom_turn = bool(rospy.get_param("~use_odom_turn", True))
        self.timed_turn_scale = float(rospy.get_param("~timed_turn_scale", 1.0))
        self.turn_direction = 1.0 if float(rospy.get_param(
            "~turn_direction", 1.0)) >= 0.0 else -1.0
        self.fetch_url = bool(rospy.get_param("~fetch_url", True))
        self.fetch_timeout_s = float(rospy.get_param("~fetch_timeout_s", 3.0))
        self.max_frame_width = int(rospy.get_param("~max_frame_width", 960))
        self.micro_sweep_enabled = bool(rospy.get_param(
            "~micro_sweep_enabled", True))
        self.micro_sweep_deg = float(rospy.get_param("~micro_sweep_deg", 12.0))
        self.micro_sweep_scan_s = float(rospy.get_param(
            "~micro_sweep_scan_s", 1.2))
        self.micro_sweep_speed = abs(float(rospy.get_param(
            "~micro_sweep_speed", 0.28)))

        self.lock = threading.Lock()
        self.nav_done = self.start_immediately
        self.nav_status_text = ""
        self.latest_gray = None
        self.latest_image_time = rospy.Time(0)
        self.current_yaw = None
        self.odom_time = rospy.Time(0)

        self.results = []
        self.wall_results = [None for _ in range(self.wall_count)]
        self.scanner = ZBarScanner()

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(self.nav_status_topic, String, self.cb_nav_status, queue_size=5)
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=1)

    def cb_nav_status(self, msg):
        text = msg.data or ""
        with self.lock:
            self.nav_status_text = text
            if self.goal_status_substring in text.lower():
                self.nav_done = True

    def cb_image(self, msg):
        gray = self.image_msg_to_gray(msg)
        if gray is None:
            return
        with self.lock:
            self.latest_gray = gray
            self.latest_image_time = rospy.Time.now()

    def cb_odom(self, msg):
        q = msg.pose.pose.orientation
        quat = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quat)
        with self.lock:
            self.current_yaw = yaw
            self.odom_time = rospy.Time.now()

    def image_msg_to_gray(self, msg):
        encoding = (msg.encoding or "").lower()
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            if encoding in ("rgb8", "bgr8"):
                image = arr.reshape((msg.height, msg.width, 3))
                code = cv2.COLOR_RGB2GRAY if encoding == "rgb8" else cv2.COLOR_BGR2GRAY
                gray = cv2.cvtColor(image, code)
            elif encoding in ("rgba8", "bgra8"):
                image = arr.reshape((msg.height, msg.width, 4))
                code = cv2.COLOR_RGBA2GRAY if encoding == "rgba8" else cv2.COLOR_BGRA2GRAY
                gray = cv2.cvtColor(image, code)
            elif encoding in ("mono8", "8uc1"):
                gray = arr.reshape((msg.height, msg.width))
            else:
                rospy.logwarn_throttle(2.0, "unsupported image encoding: %s", msg.encoding)
                return None
            if self.max_frame_width > 0 and gray.shape[1] > self.max_frame_width:
                scale = float(self.max_frame_width) / float(gray.shape[1])
                new_h = max(1, int(gray.shape[0] * scale))
                gray = cv2.resize(gray, (self.max_frame_width, new_h),
                                  interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(gray)
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "failed to convert image: %s", exc)
            return None

    def publish_zero(self):
        self.cmd_pub.publish(Twist())

    def publish_turn(self, wz):
        msg = Twist()
        msg.angular.z = wz
        self.cmd_pub.publish(msg)

    def wait_for_camera(self, timeout=10.0):
        start = rospy.Time.now()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            with self.lock:
                ready = self.latest_gray is not None
            if ready:
                return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()

    def wait_for_nav_done(self):
        if self.start_immediately:
            rospy.logwarn("start_immediately=true; QR scan starts without waiting for nav")
            return True
        rospy.loginfo("waiting for nav goal status on %s", self.nav_status_topic)
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            with self.lock:
                done = self.nav_done
                status_text = self.nav_status_text
            if done:
                rospy.loginfo("navigation done status received: %s", status_text)
                return True
            if (self.nav_wait_timeout_s > 0.0 and
                    (rospy.Time.now() - start).to_sec() > self.nav_wait_timeout_s):
                rospy.logerr("timed out waiting for navigation goal status")
                return False
            rate.sleep()

    def stop_navigation_node(self):
        self.publish_zero()
        if not self.kill_nav_on_goal:
            return
        rospy.logwarn("killing %s before QR scan to avoid cmd_vel conflict",
                      self.nav_node_name)
        try:
            subprocess.call(["rosnode", "kill", self.nav_node_name])
        except Exception as exc:
            rospy.logwarn("failed to kill nav node: %s", exc)
        end = rospy.Time.now() + rospy.Duration(self.after_nav_settle_s)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < end:
            self.publish_zero()
            rate.sleep()

    def get_yaw(self):
        with self.lock:
            if self.current_yaw is None:
                return None
            if (rospy.Time.now() - self.odom_time).to_sec() > 1.0:
                return None
            return self.current_yaw

    def rotate_90(self):
        self.publish_zero()
        rospy.sleep(0.1)
        start_yaw = self.get_yaw()
        if self.use_odom_turn and start_yaw is not None:
            target = norm_angle(start_yaw + self.turn_direction * math.pi / 2.0)
            deadline = rospy.Time.now() + rospy.Duration(self.turn_timeout_s)
            rate = rospy.Rate(20)
            while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                yaw = self.get_yaw()
                if yaw is None:
                    break
                error = norm_angle(target - yaw)
                if abs(math.degrees(error)) <= self.turn_tolerance_deg:
                    self.publish_zero()
                    return True
                speed = clamp(abs(error) * 1.4, 0.12, self.angular_speed)
                self.publish_turn(speed if error > 0.0 else -speed)
                rate.sleep()
            rospy.logwarn("odom-based 90deg turn failed or timed out; using timed turn")

        duration = (math.pi / 2.0) / max(self.angular_speed, 1.0e-3)
        duration *= self.timed_turn_scale
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self.publish_turn(self.turn_direction * self.angular_speed)
            rate.sleep()
        self.publish_zero()
        return True

    def turn_relative(self, angle_rad, speed=None):
        speed = abs(speed if speed is not None else self.micro_sweep_speed)
        if speed <= 1.0e-3:
            return
        direction = 1.0 if angle_rad >= 0.0 else -1.0
        duration = abs(angle_rad) / speed
        end = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < end:
            self.publish_turn(direction * speed)
            rate.sleep()
        self.publish_zero()

    def scan_current_view(self, wall_index, scan_s):
        start = rospy.Time.now()
        rate = rospy.Rate(self.process_rate_hz)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > scan_s:
                return None
            with self.lock:
                gray = None if self.latest_gray is None else self.latest_gray.copy()
            decoded = self.scanner.scan_gray(gray)
            if decoded:
                raw = decoded[0].strip()
                parsed = self.parse_payload(raw)
                result = {
                    "wall_index": wall_index,
                    "raw": raw,
                    "parsed": parsed,
                    "stamp": rospy.Time.now().to_sec(),
                }
                self.wall_results[wall_index] = result
                self.results.append(result)
                rospy.loginfo("wall_%d QR accepted as first physical code", wall_index)
                self.print_wall_result(result)
                return result
            rate.sleep()

    def scan_wall(self, wall_index):
        self.publish_zero()
        settle_s = self.first_wall_settle_s if wall_index == 0 else self.settle_s
        scan_s = self.first_wall_scan_s if wall_index == 0 else self.scan_per_wall_s
        rospy.loginfo("wall_%d settling %.2fs", wall_index, settle_s)
        rospy.sleep(settle_s)
        rospy.loginfo("wall_%d scanning center view up to %.1fs", wall_index, scan_s)
        result = self.scan_current_view(wall_index, scan_s)
        if result is not None:
            return result

        if not self.micro_sweep_enabled:
            rospy.logwarn("wall_%d no QR detected", wall_index)
            return None

        sweep = math.radians(abs(self.micro_sweep_deg))
        rospy.logwarn(
            "wall_%d no QR in center view; trying +/-%.1fdeg micro sweep",
            wall_index, self.micro_sweep_deg)
        self.turn_relative(sweep)
        rospy.sleep(max(0.15, self.settle_s * 0.5))
        result = self.scan_current_view(wall_index, self.micro_sweep_scan_s)
        self.turn_relative(-2.0 * sweep)
        rospy.sleep(max(0.15, self.settle_s * 0.5))
        if result is None:
            result = self.scan_current_view(wall_index, self.micro_sweep_scan_s)
        self.turn_relative(sweep)
        if result is None:
            rospy.logwarn("wall_%d no QR detected after micro sweep", wall_index)
        return result

    def parse_payload(self, raw):
        parsed = {
            "type": "raw",
            "json": None,
            "text": raw,
            "url": None,
            "error": None,
        }
        try:
            parsed["json"] = json.loads(raw)
            parsed["type"] = "json"
            parsed["text"] = None
            return parsed
        except Exception:
            pass

        if raw.startswith("http://") or raw.startswith("https://"):
            parsed["type"] = "url"
            parsed["url"] = raw
            if not self.fetch_url:
                return parsed
            try:
                import requests
                response = requests.get(raw, timeout=self.fetch_timeout_s)
                response.raise_for_status()
                text = response.content.decode("utf-8", errors="replace")
                try:
                    parsed["json"] = json.loads(text)
                    parsed["type"] = "url_json"
                    parsed["text"] = None
                except Exception:
                    parsed["text"] = text
                    parsed["type"] = "url_text"
            except Exception as exc:
                parsed["error"] = str(exc)
            return parsed
        return parsed

    def print_wall_result(self, result):
        parsed = result["parsed"]
        print("\n========== QR WALL %d ==========" % result["wall_index"], flush=True)
        print("RAW:", result["raw"], flush=True)
        if parsed.get("json") is not None:
            print("JSON:", flush=True)
            print(json.dumps(parsed["json"], ensure_ascii=False, indent=2), flush=True)
        elif parsed.get("text") is not None:
            print("TEXT:", parsed["text"], flush=True)
        if parsed.get("error"):
            print("ERROR:", parsed["error"], flush=True)
        print("================================\n", flush=True)

    def publish_summary(self):
        summary = {
            "status": "complete" if len(self.results) >= self.target_qr_count else "partial",
            "target_qr_count": self.target_qr_count,
            "detected_count": len(self.results),
            "wall_results": self.wall_results,
        }
        msg = String()
        msg.data = json.dumps(summary, ensure_ascii=False)
        self.result_pub.publish(msg)
        print("\n========== QR ROOM SUMMARY ==========", flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("=====================================\n", flush=True)

    def run(self):
        try:
            if not self.wait_for_nav_done():
                return
            if not self.wait_for_camera():
                rospy.logerr("no camera image received; cannot scan")
                return
            self.stop_navigation_node()
            rospy.loginfo("starting QR room scan with cmd topic %s", self.cmd_vel_topic)
            for wall in range(self.wall_count):
                if len(self.results) >= self.target_qr_count:
                    break
                self.scan_wall(wall)
                if len(self.results) >= self.target_qr_count:
                    break
                if wall + 1 < self.wall_count:
                    self.rotate_90()
            self.publish_zero()
            self.publish_summary()
        finally:
            self.publish_zero()
            self.scanner.close()


def main():
    node = GlobalFirstNavThenQR()
    node.run()
    rospy.loginfo("global_first_nav_qr_room_scan finished; node stays alive")
    rospy.spin()


if __name__ == "__main__":
    main()
