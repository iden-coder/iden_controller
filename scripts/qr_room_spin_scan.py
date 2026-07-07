#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ctypes
import json
import math
import threading
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from std_msgs.msg import String
from tf.transformations import euler_from_quaternion


INF = float("inf")


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def norm_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


class ZBarScanner(object):
    ZBAR_QRCODE = 64
    ZBAR_CFG_ENABLE = 0
    Y800 = (ord("Y") |
            (ord("8") << 8) |
            (ord("0") << 16) |
            (ord("0") << 24))

    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary("libzbar.so.0")
        self._setup_signatures()
        self.scanner = self.lib.zbar_image_scanner_create()
        if not self.scanner:
            raise RuntimeError("failed to create zbar image scanner")
        self.lib.zbar_image_scanner_set_config(
            self.scanner, self.ZBAR_QRCODE, self.ZBAR_CFG_ENABLE, 1)

    def _setup_signatures(self):
        c_void_p = ctypes.c_void_p
        self.lib.zbar_image_scanner_create.restype = c_void_p
        self.lib.zbar_image_scanner_destroy.argtypes = [c_void_p]
        self.lib.zbar_image_scanner_set_config.argtypes = [
            c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.zbar_image_create.restype = c_void_p
        self.lib.zbar_image_destroy.argtypes = [c_void_p]
        self.lib.zbar_image_set_format.argtypes = [c_void_p, ctypes.c_ulong]
        self.lib.zbar_image_set_size.argtypes = [
            c_void_p, ctypes.c_uint, ctypes.c_uint]
        self.lib.zbar_image_set_data.argtypes = [
            c_void_p, c_void_p, ctypes.c_ulong, c_void_p]
        self.lib.zbar_scan_image.argtypes = [c_void_p, c_void_p]
        self.lib.zbar_scan_image.restype = ctypes.c_int
        self.lib.zbar_image_first_symbol.argtypes = [c_void_p]
        self.lib.zbar_image_first_symbol.restype = c_void_p
        self.lib.zbar_symbol_next.argtypes = [c_void_p]
        self.lib.zbar_symbol_next.restype = c_void_p
        self.lib.zbar_symbol_get_data.argtypes = [c_void_p]
        self.lib.zbar_symbol_get_data.restype = ctypes.c_char_p

    def scan_gray(self, gray):
        if gray is None or gray.size == 0:
            return []
        gray = np.ascontiguousarray(gray)
        height, width = gray.shape[:2]
        image = self.lib.zbar_image_create()
        if not image:
            return []
        try:
            self.lib.zbar_image_set_format(image, self.Y800)
            self.lib.zbar_image_set_size(image, width, height)
            ptr = gray.ctypes.data_as(ctypes.c_void_p)
            self.lib.zbar_image_set_data(
                image, ptr, ctypes.c_ulong(gray.nbytes), None)
            count = self.lib.zbar_scan_image(self.scanner, image)
            if count <= 0:
                return []
            results = []
            symbol = self.lib.zbar_image_first_symbol(image)
            while symbol:
                data = self.lib.zbar_symbol_get_data(symbol)
                if data:
                    results.append(data.decode("utf-8", errors="replace"))
                symbol = self.lib.zbar_symbol_next(symbol)
            return results
        finally:
            self.lib.zbar_image_destroy(image)

    def close(self):
        if self.scanner:
            self.lib.zbar_image_scanner_destroy(self.scanner)
            self.scanner = None


class QRRoomSpinScan(object):
    def __init__(self):
        rospy.init_node("qr_room_spin_scan")
        self.image_topic = rospy.get_param("~image_topic", "/ucar_camera/image_raw")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.result_topic = rospy.get_param(
            "~result_topic", "/qr_room_scan_results")
        self.wall_count = int(rospy.get_param("~wall_count", 4))
        self.target_qr_count = int(rospy.get_param("~target_qr_count", 3))
        self.scan_per_wall_s = float(rospy.get_param("~scan_per_wall_s", 2.8))
        self.settle_s = float(rospy.get_param("~settle_s", 0.45))
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 12.0))
        self.angular_speed = abs(float(rospy.get_param("~angular_speed", 0.52)))
        self.turn_tolerance_deg = float(rospy.get_param(
            "~turn_tolerance_deg", 3.0))
        self.turn_timeout_s = float(rospy.get_param("~turn_timeout_s", 12.0))
        self.use_odom_turn = bool(rospy.get_param("~use_odom_turn", True))
        self.timed_turn_scale = float(rospy.get_param("~timed_turn_scale", 1.0))
        self.turn_direction = 1.0 if float(rospy.get_param(
            "~turn_direction", 1.0)) >= 0.0 else -1.0
        self.fetch_url = bool(rospy.get_param("~fetch_url", True))
        self.fetch_timeout_s = float(rospy.get_param("~fetch_timeout_s", 3.0))
        self.max_frame_width = int(rospy.get_param("~max_frame_width", 960))

        self.lock = threading.Lock()
        self.latest_gray = None
        self.latest_image_time = rospy.Time(0)
        self.current_yaw = None
        self.odom_time = rospy.Time(0)
        self.state = "IDLE"
        self.active_wall = -1

        self.scanner = ZBarScanner()
        self.results = []
        self.wall_results = [None for _ in range(self.wall_count)]

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=1, latch=True)
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=1)

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
                if encoding == "rgb8":
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif encoding in ("rgba8", "bgra8"):
                image = arr.reshape((msg.height, msg.width, 4))
                if encoding == "rgba8":
                    gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif encoding in ("mono8", "8uc1"):
                gray = arr.reshape((msg.height, msg.width))
            else:
                rospy.logwarn_throttle(
                    2.0, "unsupported image encoding: %s", msg.encoding)
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

    def wait_for_image(self, timeout=8.0):
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

    def get_yaw(self):
        with self.lock:
            if self.current_yaw is None:
                return None
            if (rospy.Time.now() - self.odom_time).to_sec() > 1.0:
                return None
            return self.current_yaw

    def rotate_90(self):
        self.state = "ROTATING"
        self.active_wall = -1
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

    def scan_wall(self, wall_index):
        self.state = "SETTLING"
        self.active_wall = wall_index
        self.publish_zero()
        rospy.loginfo("wall_%d settling %.1fs", wall_index, self.settle_s)
        rospy.sleep(self.settle_s)

        self.state = "SCANNING"
        start = rospy.Time.now()
        rate = rospy.Rate(self.process_rate_hz)
        rospy.loginfo("wall_%d scanning up to %.1fs", wall_index, self.scan_per_wall_s)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > self.scan_per_wall_s:
                rospy.logwarn("wall_%d no QR detected", wall_index)
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
                try:
                    parsed["json"] = response.json()
                    parsed["type"] = "url_json"
                    parsed["text"] = None
                except Exception:
                    parsed["text"] = response.text
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
            print(json.dumps(parsed["json"], ensure_ascii=False, indent=2),
                  flush=True)
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
        rospy.loginfo("waiting for camera image on %s", self.image_topic)
        if not self.wait_for_image():
            rospy.logerr("no camera image received; cannot scan")
            return

        rospy.loginfo(
            "QR room spin scan started: walls=%d target=%d cmd_vel=%s",
            self.wall_count, self.target_qr_count, self.cmd_vel_topic)
        try:
            for wall in range(self.wall_count):
                if len(self.results) >= self.target_qr_count:
                    break
                self.scan_wall(wall)
                if len(self.results) >= self.target_qr_count:
                    break
                if wall + 1 < self.wall_count:
                    self.rotate_90()
            self.state = "DONE"
            self.publish_zero()
            self.publish_summary()
        finally:
            self.publish_zero()
            self.scanner.close()


def main():
    node = QRRoomSpinScan()
    node.run()
    rospy.loginfo("qr_room_spin_scan finished; node stays alive for latched result")
    rospy.spin()


if __name__ == "__main__":
    main()
