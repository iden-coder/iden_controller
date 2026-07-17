#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Continuous QR-room scanner used after the Xunfei 2026 first-stage route.

Motion and decoding are intentionally independent: a ROS timer keeps angular
velocity continuous while a worker decodes the newest camera frame and HTTP
workers resolve QR URLs without blocking the base controller.
"""

import ctypes
import json
import math
import subprocess
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf.transformations import euler_from_quaternion


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def normalize_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def repair_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    for encoding in ("gbk", "latin1"):
        try:
            candidate = text.encode(encoding).decode("utf-8")
        except Exception:
            continue
        if candidate and candidate != text:
            return candidate
    return text


def repair_value(value):
    if isinstance(value, str):
        return repair_text(value)
    if isinstance(value, list):
        return [repair_value(item) for item in value]
    if isinstance(value, dict):
        return {key: repair_value(item) for key, item in value.items()}
    return value


class ZBarScanner(object):
    ZBAR_QRCODE = 64
    ZBAR_CFG_ENABLE = 0
    Y800 = ord("Y") | (ord("8") << 8) | (ord("0") << 16) | (ord("0") << 24)

    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary("libzbar.so.0")
        self._configure_signatures()
        self.scanner = self.lib.zbar_image_scanner_create()
        if not self.scanner:
            raise RuntimeError("cannot create zbar scanner")
        self.lib.zbar_image_scanner_set_config(
            self.scanner, self.ZBAR_QRCODE, self.ZBAR_CFG_ENABLE, 1)

    def _configure_signatures(self):
        pointer = ctypes.c_void_p
        self.lib.zbar_image_scanner_create.restype = pointer
        self.lib.zbar_image_scanner_destroy.argtypes = [pointer]
        self.lib.zbar_image_scanner_set_config.argtypes = [
            pointer, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.zbar_image_create.restype = pointer
        self.lib.zbar_image_destroy.argtypes = [pointer]
        self.lib.zbar_image_set_format.argtypes = [pointer, ctypes.c_ulong]
        self.lib.zbar_image_set_size.argtypes = [
            pointer, ctypes.c_uint, ctypes.c_uint]
        self.lib.zbar_image_set_data.argtypes = [
            pointer, pointer, ctypes.c_ulong, pointer]
        self.lib.zbar_scan_image.argtypes = [pointer, pointer]
        self.lib.zbar_scan_image.restype = ctypes.c_int
        self.lib.zbar_image_first_symbol.argtypes = [pointer]
        self.lib.zbar_image_first_symbol.restype = pointer
        self.lib.zbar_symbol_next.argtypes = [pointer]
        self.lib.zbar_symbol_next.restype = pointer
        self.lib.zbar_symbol_get_data.argtypes = [pointer]
        self.lib.zbar_symbol_get_data.restype = ctypes.c_char_p

    def scan(self, gray):
        if gray is None or gray.size == 0:
            return []
        gray = np.ascontiguousarray(gray, dtype=np.uint8)
        height, width = gray.shape[:2]
        image = self.lib.zbar_image_create()
        if not image:
            return []
        try:
            self.lib.zbar_image_set_format(image, self.Y800)
            self.lib.zbar_image_set_size(image, width, height)
            pointer = gray.ctypes.data_as(ctypes.c_void_p)
            self.lib.zbar_image_set_data(
                image, pointer, ctypes.c_ulong(gray.nbytes), None)
            if self.lib.zbar_scan_image(self.scanner, image) <= 0:
                return []
            values = []
            symbol = self.lib.zbar_image_first_symbol(image)
            while symbol:
                data = self.lib.zbar_symbol_get_data(symbol)
                if data:
                    text = data.decode("utf-8", errors="replace").strip()
                    if text and text not in values:
                        values.append(text)
                symbol = self.lib.zbar_symbol_next(symbol)
            return values
        finally:
            self.lib.zbar_image_destroy(image)

    def close(self):
        if self.scanner:
            self.lib.zbar_image_scanner_destroy(self.scanner)
            self.scanner = None


class HybridDecoder(object):
    def __init__(self, scales):
        self.zbar = ZBarScanner()
        try:
            self.detector = cv2.QRCodeDetector()
        except AttributeError:
            self.detector = None
            rospy.logwarn(
                "cv2.QRCodeDetector unavailable; using enhanced multi-scale ZBar")
        self.scales = scales
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _append(values, text):
        text = repair_text(text)
        if text and text not in values:
            values.append(text)

    def _opencv_decode(self, image, values):
        if self.detector is None:
            return
        try:
            result = self.detector.detectAndDecodeMulti(image)
            if isinstance(result, tuple) and len(result) >= 2:
                success, decoded = result[0], result[1]
                if success and decoded:
                    for text in decoded:
                        self._append(values, text)
        except Exception:
            pass
        if values:
            return
        try:
            text, _points, _straight = self.detector.detectAndDecode(image)
            self._append(values, text)
        except Exception:
            pass

    def decode_fast(self, gray):
        values = []
        self._opencv_decode(gray, values)
        for text in self.zbar.scan(gray):
            self._append(values, text)
        return values

    def decode_enhanced(self, gray):
        values = []
        enhanced = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        sharpened = cv2.addWeighted(enhanced, 1.6, blurred, -0.6, 0)
        for scale in self.scales:
            candidate = sharpened
            if abs(scale - 1.0) > 1.0e-6:
                candidate = cv2.resize(
                    sharpened, None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC)
            self._opencv_decode(candidate, values)
            for text in self.zbar.scan(candidate):
                self._append(values, text)
            if values:
                break
        return values

    def decode(self, gray, use_enhanced=True):
        values = self.decode_fast(gray)
        if values or not use_enhanced:
            return values
        return self.decode_enhanced(gray)

    def close(self):
        self.zbar.close()


class ContinuousQRHybrid(object):
    def __init__(self):
        self.image_topic = rospy.get_param("~image_topic", "/ucar_camera/image_raw")
        self.start_camera = bool(rospy.get_param("~start_camera", True))
        self.camera_device = rospy.get_param("~camera_device", "/dev/video0")
        self.camera_width = int(rospy.get_param("~camera_width", 1280))
        self.camera_height = int(rospy.get_param("~camera_height", 720))
        self.camera_rate = int(rospy.get_param("~camera_rate", 15))
        self.camera_low_exposure_enabled = bool(
            rospy.get_param("~camera_low_exposure_enabled", True))
        self.camera_low_exposure_absolute = int(
            rospy.get_param("~camera_low_exposure_absolute", 100))
        self.camera_exposure_service = rospy.get_param(
            "~camera_exposure_service", "/ucar_camera/set_exposure_profile")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.nav_status_topic = rospy.get_param(
            "~nav_status_topic", "/xunfei2026_first_stage/status")
        self.result_topic = rospy.get_param(
            "~result_topic", "/qr_room_scan_results")
        self.item_topic = rospy.get_param("~item_topic", "/qr_room_scan_item")
        self.status_topic = rospy.get_param(
            "~status_topic", "/xunfei2026_continuous_qr/status")
        self.target_count = max(1, int(rospy.get_param("~target_count", 3)))
        self.angular_speed = abs(float(rospy.get_param("~angular_speed", 0.22)))
        self.angular_accel = abs(float(rospy.get_param("~angular_accel", 0.75)))
        self.direction = 1.0 if float(rospy.get_param("~turn_direction", 1.0)) >= 0 else -1.0
        self.command_rate_hz = max(10.0, float(rospy.get_param("~command_rate_hz", 30.0)))
        self.decode_rate_hz = max(1.0, float(rospy.get_param("~decode_rate_hz", 15.0)))
        self.enhanced_rate_hz = max(
            0.2, float(rospy.get_param("~enhanced_rate_hz", 2.5)))
        self.nav_settle_s = max(0.0, float(rospy.get_param("~nav_settle_s", 0.7)))
        self.camera_timeout_s = max(1.0, float(rospy.get_param("~camera_timeout_s", 15.0)))
        self.odom_timeout_s = max(0.2, float(rospy.get_param("~odom_timeout_s", 0.8)))
        self.max_scan_s = max(0.0, float(rospy.get_param("~max_scan_s", 0.0)))
        self.fetch_timeout_s = max(0.5, float(rospy.get_param("~fetch_timeout_s", 4.0)))
        self.fetch_retries = max(1, int(rospy.get_param("~fetch_retries", 4)))
        self.fetch_retry_delay_s = max(0.0, float(rospy.get_param("~fetch_retry_delay_s", 0.35)))
        self.retry_cooldown_s = max(0.5, float(rospy.get_param("~retry_cooldown_s", 5.0)))
        self.require_url_identity = bool(rospy.get_param("~require_url_identity", True))
        self.max_frame_width = max(320, int(rospy.get_param("~max_frame_width", 960)))
        self.center_enabled = bool(rospy.get_param("~center_enabled", True))
        self.center_timeout_s = max(0.5, float(rospy.get_param("~center_timeout_s", 5.0)))
        self.center_tolerance_m = max(0.01, float(rospy.get_param("~center_tolerance_m", 0.035)))
        self.center_max_speed = max(0.02, float(rospy.get_param("~center_max_speed", 0.08)))
        self.center_gain = max(0.1, float(rospy.get_param("~center_gain", 0.75)))
        self.center_max_move_m = max(0.03, float(rospy.get_param("~center_max_move_m", 0.16)))
        self.center_hard_clearance_m = max(
            0.10, float(rospy.get_param("~center_hard_clearance_m", 0.20)))
        self.center_sector_half_deg = clamp(
            float(rospy.get_param("~center_sector_half_deg", 28.0)), 10.0, 42.0)
        self.center_pair_sum_min_m = max(
            0.3, float(rospy.get_param("~center_pair_sum_min_m", 0.65)))
        self.center_pair_sum_max_m = max(
            self.center_pair_sum_min_m + 0.2,
            float(rospy.get_param("~center_pair_sum_max_m", 2.8)))
        self.center_axis_error_limit_m = max(
            0.08, float(rospy.get_param("~center_axis_error_limit_m", 0.32)))
        scale_text = str(rospy.get_param("~decode_scales", "1.0,1.5,2.0"))
        self.decode_scales = self._parse_scales(scale_text)

        self.lock = threading.RLock()
        self.latest_gray = None
        self.frame_sequence = 0
        self.last_fast_sequence = -1
        self.last_enhanced_sequence = -1
        self.fast_attempts = 0
        self.enhanced_attempts = 0
        self.decode_stats_started_at = time.monotonic()
        self.image_stamp = rospy.Time(0)
        self.odom_stamp = rospy.Time(0)
        self.scan_stamp = rospy.Time(0)
        self.cardinal_ranges = None
        self.odom_position = None
        self.yaw = None
        self.last_yaw = None
        self.accumulated_yaw = 0.0
        self.reported_rounds = 0
        self.nav_status = ""
        self.nav_finished = threading.Event()
        self.nav_failed = threading.Event()
        self.scan_active = False
        self.motion_mode = "IDLE"
        self.center_command = (0.0, 0.0, 0.0)
        self.scan_started_at = None
        self.complete = threading.Event()
        self.results = []
        self.accepted_identities = set()
        self.inflight_identities = set()
        self.failed_at = {}
        # ZBar scanners are not shared across threads.  The fast raw-image
        # path therefore remains responsive while enhancement is expensive.
        self.fast_decoder = HybridDecoder(self.decode_scales)
        self.enhanced_decoder = HybridDecoder(self.decode_scales)
        self.camera_process = None

        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        self.result_pub = rospy.Publisher(self.result_topic, String, queue_size=1, latch=True)
        self.item_pub = rospy.Publisher(self.item_topic, String, queue_size=5)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=5, latch=True)
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback, queue_size=3)
        rospy.Subscriber(self.nav_status_topic, String, self.nav_status_callback, queue_size=10)
        self.command_timer = rospy.Timer(
            rospy.Duration(1.0 / self.command_rate_hz), self.command_timer_callback)
        self.decode_thread = threading.Thread(target=self.decode_loop)
        self.decode_thread.daemon = True
        self.decode_thread.start()
        self.enhanced_decode_thread = threading.Thread(
            target=self.enhanced_decode_loop)
        self.enhanced_decode_thread.daemon = True
        self.enhanced_decode_thread.start()
        rospy.on_shutdown(self.shutdown)

    @staticmethod
    def _parse_scales(text):
        values = []
        for token in text.split(","):
            try:
                value = float(token.strip())
            except Exception:
                continue
            if value >= 1.0 and value not in values:
                values.append(value)
        if 1.0 not in values:
            values.insert(0, 1.0)
        return sorted(values)

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))
        rospy.logwarn("XUNFEI2026_QR %s", text)

    def image_callback(self, msg):
        if not self.nav_finished.is_set():
            return
        try:
            array = np.frombuffer(msg.data, dtype=np.uint8)
            encoding = (msg.encoding or "").lower()
            if encoding in ("rgb8", "bgr8"):
                image = self.reshape_raw_image(msg, array, 3)
                conversion = cv2.COLOR_RGB2GRAY if encoding == "rgb8" else cv2.COLOR_BGR2GRAY
                gray = cv2.cvtColor(image, conversion)
            elif encoding in ("rgba8", "bgra8"):
                image = self.reshape_raw_image(msg, array, 4)
                conversion = cv2.COLOR_RGBA2GRAY if encoding == "rgba8" else cv2.COLOR_BGRA2GRAY
                gray = cv2.cvtColor(image, conversion)
            elif encoding in ("mono8", "8uc1"):
                gray = self.reshape_raw_image(msg, array, 1)
            else:
                rospy.logwarn_throttle(3.0, "QR unsupported image encoding=%s", msg.encoding)
                return
            if gray.shape[1] > self.max_frame_width:
                ratio = float(self.max_frame_width) / float(gray.shape[1])
                gray = cv2.resize(
                    gray, (self.max_frame_width, max(1, int(gray.shape[0] * ratio))),
                    interpolation=cv2.INTER_AREA)
            with self.lock:
                self.latest_gray = np.ascontiguousarray(gray)
                self.frame_sequence += 1
                self.image_stamp = rospy.Time.now()
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "QR image conversion failed: %s", exc)

    def reshape_raw_image(self, msg, array, channels):
        """Recover raw frames even when a camera driver publishes stale dimensions."""
        height = int(msg.height)
        width = int(msg.width)
        expected = height * width * channels
        if expected == array.size:
            return array.reshape((height, width, channels)) if channels > 1 else array.reshape((height, width))

        pixels = array.size // channels
        if pixels * channels != array.size:
            raise ValueError("raw image byte count is not divisible by channels")
        candidates = [
            (self.camera_width, self.camera_height),
            (640, 480), (800, 600), (1280, 720), (1280, 960),
            (1920, 1080), (320, 240),
        ]
        for actual_width, actual_height in candidates:
            if actual_width * actual_height != pixels:
                continue
            rospy.logwarn_throttle(
                5.0,
                "QR_CAMERA_METADATA_RECOVERED declared=%dx%d actual=%dx%d source=data",
                width, height, actual_width, actual_height)
            return (array.reshape((actual_height, actual_width, channels))
                    if channels > 1 else array.reshape((actual_height, actual_width)))

        # Some camera drivers leave width, height and step from the requested
        # mode even when V4L2 silently falls back to another resolution.  Step
        # is therefore only a last resort and must produce a plausible aspect.
        step = int(getattr(msg, "step", 0) or 0)
        if step >= channels and step % channels == 0 and array.size % step == 0:
            step_height = array.size // step
            step_width = step // channels
            aspect = float(step_width) / max(1.0, float(step_height))
            if step_height > 0 and step_width > 0 and 0.75 <= aspect <= 2.40:
                rospy.logwarn_throttle(
                    5.0,
                    "QR_CAMERA_METADATA_RECOVERED declared=%dx%d actual=%dx%d source=step",
                    width, height, step_width, step_height)
                return (array.reshape((step_height, step_width, channels))
                        if channels > 1 else array.reshape((step_height, step_width)))
        raise ValueError(
            "cannot infer raw image shape: declared={}x{} channels={} bytes={}".format(
                width, height, channels, array.size))

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        with self.lock:
            if self.last_yaw is not None and self.scan_active:
                delta = normalize_angle(yaw - self.last_yaw)
                if abs(delta) < math.radians(35.0):
                    self.accumulated_yaw += abs(delta)
            self.last_yaw = yaw
            self.yaw = yaw
            self.odom_position = (
                msg.pose.pose.position.x, msg.pose.pose.position.y)
            self.odom_stamp = rospy.Time.now()

    def scan_callback(self, msg):
        if not self.nav_finished.is_set():
            return
        cardinal = {
            "front": self.wall_distance(msg, 0.0),
            "left": self.wall_distance(msg, math.pi / 2.0),
            "rear": self.wall_distance(msg, math.pi),
            "right": self.wall_distance(msg, -math.pi / 2.0),
        }
        if any(value is None for value in cardinal.values()):
            return
        with self.lock:
            self.cardinal_ranges = cardinal
            self.scan_stamp = rospy.Time.now()

    def wall_distance(self, scan, center_angle):
        half_width = math.radians(self.center_sector_half_deg)
        projected = []
        angle = scan.angle_min
        for distance in scan.ranges:
            delta = normalize_angle(angle - center_angle)
            if (abs(delta) <= half_width and math.isfinite(distance) and
                    distance >= max(0.05, scan.range_min) and
                    distance <= scan.range_max):
                projected.append(distance * math.cos(delta))
            angle += scan.angle_increment
        if len(projected) < 5:
            return None
        # The lower-middle percentile rejects a doorway's long rays while
        # retaining enough wall samples to ignore isolated short noise.
        return float(np.percentile(np.asarray(projected), 38.0))

    def nav_status_callback(self, msg):
        status = (msg.data or "").strip().upper()
        with self.lock:
            self.nav_status = status
        if status == "SUCCEEDED":
            self.nav_finished.set()
        elif status.startswith("FAILED"):
            self.nav_failed.set()

    def command_timer_callback(self, _event):
        message = Twist()
        with self.lock:
            active = self.scan_active and not self.complete.is_set()
            started_at = self.scan_started_at
            motion_mode = self.motion_mode
            center_command = self.center_command
            odom_fresh = (
                self.odom_stamp != rospy.Time(0) and
                (rospy.Time.now() - self.odom_stamp).to_sec() <= self.odom_timeout_s)
        if motion_mode == "CENTER":
            message.linear.x = center_command[0]
            message.linear.y = center_command[1]
            message.angular.z = center_command[2]
        elif active and motion_mode == "SCAN" and odom_fresh and started_at is not None:
            elapsed = max(0.0, time.monotonic() - started_at)
            speed = min(self.angular_speed, max(0.08, elapsed * self.angular_accel))
            message.angular.z = self.direction * speed
        elif active and not odom_fresh:
            rospy.logwarn_throttle(2.0, "QR rotation paused: odometry is stale")
        else:
            # While move_base owns the first-stage route, publishing zero here
            # would race its 15 Hz velocity output and make navigation stutter.
            return
        self.cmd_pub.publish(message)

    def center_with_lidar(self):
        if not self.center_enabled:
            self.publish_status("LIDAR_CENTER_SKIPPED_DISABLED")
            return False

        ready_deadline = time.monotonic() + 2.0
        while not rospy.is_shutdown() and time.monotonic() < ready_deadline:
            with self.lock:
                scan_ready = (
                    self.cardinal_ranges is not None and
                    self.scan_stamp != rospy.Time(0) and
                    (rospy.Time.now() - self.scan_stamp).to_sec() <= 0.6 and
                    self.odom_position is not None)
            if scan_ready:
                break
            rospy.sleep(0.05)
        else:
            self.publish_status("LIDAR_CENTER_SKIPPED_NO_FRESH_SCAN")
            return False

        with self.lock:
            start_position = self.odom_position
            self.motion_mode = "CENTER"
            self.center_command = (0.0, 0.0, 0.0)
        self.publish_status("LIDAR_CENTERING_STARTED")
        deadline = time.monotonic() + self.center_timeout_s
        stable_cycles = 0
        rate = rospy.Rate(15)
        result = False
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                cardinal = None if self.cardinal_ranges is None else dict(self.cardinal_ranges)
                scan_age = ((rospy.Time.now() - self.scan_stamp).to_sec()
                            if self.scan_stamp != rospy.Time(0) else 999.0)
                position = self.odom_position
            if cardinal is None or scan_age > 0.7 or position is None:
                with self.lock:
                    self.center_command = (0.0, 0.0, 0.0)
                rospy.logwarn_throttle(1.0, "LIDAR_CENTER paused: scan or odometry stale")
                rate.sleep()
                continue

            moved = math.hypot(
                position[0] - start_position[0], position[1] - start_position[1])
            if moved >= self.center_max_move_m:
                rospy.logwarn("LIDAR_CENTER move limit reached: %.3fm", moved)
                break

            front = cardinal["front"]
            rear = cardinal["rear"]
            left = cardinal["left"]
            right = cardinal["right"]
            pair_x = front + rear
            pair_y = left + right
            valid_x = self.center_pair_sum_min_m <= pair_x <= self.center_pair_sum_max_m
            valid_y = self.center_pair_sum_min_m <= pair_y <= self.center_pair_sum_max_m
            error_x = 0.5 * (front - rear) if valid_x else 0.0
            error_y = 0.5 * (left - right) if valid_y else 0.0
            if abs(error_x) > self.center_axis_error_limit_m:
                rospy.logwarn_throttle(
                    1.0, "LIDAR_CENTER x rejected as doorway/outlier: %.3fm", error_x)
                error_x = 0.0
                valid_x = False
            if abs(error_y) > self.center_axis_error_limit_m:
                rospy.logwarn_throttle(
                    1.0, "LIDAR_CENTER y rejected as doorway/outlier: %.3fm", error_y)
                error_y = 0.0
                valid_y = False
            if not valid_x and not valid_y:
                rospy.logwarn("LIDAR_CENTER skipped: opposite wall pairs are unreliable %s", cardinal)
                break

            x_ok = (not valid_x) or abs(error_x) <= self.center_tolerance_m
            y_ok = (not valid_y) or abs(error_y) <= self.center_tolerance_m
            if x_ok and y_ok:
                stable_cycles += 1
            else:
                stable_cycles = 0
            if stable_cycles >= 5:
                result = True
                rospy.logwarn(
                    "LIDAR_CENTER_OK front=%.3f rear=%.3f left=%.3f right=%.3f moved=%.3f",
                    front, rear, left, right, moved)
                break

            vx = clamp(self.center_gain * error_x, -self.center_max_speed, self.center_max_speed)
            vy = clamp(self.center_gain * error_y, -self.center_max_speed, self.center_max_speed)
            if (vx > 0.0 and front <= self.center_hard_clearance_m) or (
                    vx < 0.0 and rear <= self.center_hard_clearance_m):
                vx = 0.0
            if (vy > 0.0 and left <= self.center_hard_clearance_m) or (
                    vy < 0.0 and right <= self.center_hard_clearance_m):
                vy = 0.0
            with self.lock:
                self.center_command = (vx, vy, 0.0)
            rospy.loginfo_throttle(
                0.5,
                "LIDAR_CENTER error=(%.3f,%.3f) cmd=(%.3f,%.3f) ranges=(%.2f,%.2f,%.2f,%.2f)",
                error_x, error_y, vx, vy, front, rear, left, right)
            rate.sleep()

        with self.lock:
            self.center_command = (0.0, 0.0, 0.0)
            self.motion_mode = "IDLE"
        self.publish_zero(8)
        if not result:
            self.publish_status("LIDAR_CENTER_FINISHED_BEST_EFFORT")
        return result

    def decode_loop(self):
        rate = rospy.Rate(self.decode_rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                active = self.scan_active and not self.complete.is_set()
                sequence = self.frame_sequence
                gray = None
                if active and sequence != self.last_fast_sequence and self.latest_gray is not None:
                    gray = self.latest_gray.copy()
                    self.last_fast_sequence = sequence
            if gray is not None:
                try:
                    decoded = self.fast_decoder.decode_fast(gray)
                    with self.lock:
                        self.fast_attempts += 1
                    for raw in decoded:
                        self.queue_payload(raw)
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "QR fast decode failed: %s", exc)
                self.report_decode_stats()
            rate.sleep()

    def enhanced_decode_loop(self):
        rate = rospy.Rate(self.enhanced_rate_hz)
        while not rospy.is_shutdown():
            with self.lock:
                active = self.scan_active and not self.complete.is_set()
                sequence = self.frame_sequence
                gray = None
                if (active and sequence != self.last_enhanced_sequence and
                        self.latest_gray is not None):
                    gray = self.latest_gray.copy()
                    self.last_enhanced_sequence = sequence
            if gray is not None:
                try:
                    decoded = self.enhanced_decoder.decode_enhanced(gray)
                    with self.lock:
                        self.enhanced_attempts += 1
                    for raw in decoded:
                        self.queue_payload(raw)
                except Exception as exc:
                    rospy.logwarn_throttle(2.0, "QR enhanced decode failed: %s", exc)
            rate.sleep()

    def report_decode_stats(self):
        elapsed = time.monotonic() - self.decode_stats_started_at
        if elapsed < 5.0:
            return
        with self.lock:
            fast_attempts = self.fast_attempts
            enhanced_attempts = self.enhanced_attempts
            frames = self.frame_sequence
            self.fast_attempts = 0
            self.enhanced_attempts = 0
            self.decode_stats_started_at = time.monotonic()
        rospy.loginfo(
            "QR_DECODE_RATE fast=%.2fHz enhanced=%.2fHz fast_attempts=%d enhanced_attempts=%d frames=%d",
            fast_attempts / elapsed, enhanced_attempts / elapsed,
            fast_attempts, enhanced_attempts, frames)

    @staticmethod
    def stable_identity(raw):
        text = (raw or "").strip()
        if text.startswith(("http://", "https://")):
            parsed = urlsplit(text)
            path = parsed.path.rstrip("/") or "/"
            query = "?" + parsed.query if parsed.query else ""
            return "url:{}://{}{}{}".format(
                parsed.scheme.lower(), parsed.netloc.lower(), path, query)
        return "raw:" + "".join(text.split())

    def queue_payload(self, raw):
        raw = (raw or "").strip()
        if self.require_url_identity and not raw.startswith(("http://", "https://")):
            rospy.logwarn_throttle(
                3.0, "QR_NON_URL_IGNORED; three distinct physical URLs are required")
            return
        identity = self.stable_identity(raw)
        if not identity:
            return
        now = time.monotonic()
        with self.lock:
            if identity in self.accepted_identities or identity in self.inflight_identities:
                return
            if now - self.failed_at.get(identity, -1.0e9) < self.retry_cooldown_s:
                return
            self.inflight_identities.add(identity)
        rospy.loginfo("QR_PHYSICAL_CODE_SEEN identity=%s", identity)
        worker = threading.Thread(target=self.resolve_payload, args=(raw, identity))
        worker.daemon = True
        worker.start()

    def fetch_url(self, url):
        import requests
        last_error = ""
        for attempt in range(1, self.fetch_retries + 1):
            try:
                rospy.loginfo(
                    "QR_URL_FETCH_START attempt=%d/%d url=%s",
                    attempt, self.fetch_retries, url)
                response = requests.get(
                    url, timeout=self.fetch_timeout_s,
                    headers={"User-Agent": "U-CAR-Xunfei2026-QR/1.0"})
                response.raise_for_status()
                try:
                    return repair_value(response.json()), None
                except Exception:
                    return repair_text(response.text), None
            except Exception as exc:
                last_error = str(exc)
                rospy.logwarn(
                    "QR_URL_FETCH_FAIL attempt=%d/%d error=%s",
                    attempt, self.fetch_retries, last_error)
                if attempt < self.fetch_retries:
                    rospy.sleep(self.fetch_retry_delay_s)
        return None, last_error

    def resolve_payload(self, raw, identity):
        parsed = {"type": "raw", "json": None, "text": raw, "url": None, "error": None}
        try:
            try:
                parsed["json"] = repair_value(json.loads(raw))
                parsed["type"] = "json"
                parsed["text"] = None
            except Exception:
                if raw.startswith(("http://", "https://")):
                    parsed["url"] = raw
                    value, error = self.fetch_url(raw)
                    parsed["error"] = error
                    if isinstance(value, (dict, list)):
                        parsed["json"] = value
                        parsed["text"] = None
                        parsed["type"] = "url_json"
                    elif isinstance(value, str) and value:
                        try:
                            parsed["json"] = repair_value(json.loads(value))
                            parsed["text"] = None
                            parsed["type"] = "url_json"
                        except Exception:
                            parsed["text"] = repair_text(value)
                            parsed["type"] = "url_text"
                    else:
                        parsed["text"] = None
                        parsed["type"] = "url_fetch_failed"

            if not self.has_item(parsed):
                with self.lock:
                    self.failed_at[identity] = time.monotonic()
                rospy.logwarn("QR_PAYLOAD_RETRY_LATER identity=%s parsed_type=%s error=%s",
                              identity, parsed["type"], parsed.get("error"))
                return
            self.accept_result(raw, identity, parsed)
        finally:
            with self.lock:
                self.inflight_identities.discard(identity)

    @staticmethod
    def has_item(parsed):
        value = parsed.get("json")
        if isinstance(value, dict):
            code = value.get("code")
            if code not in (None, 200, "200"):
                return False
            for key in ("result", "name", "item", "goods", "product", "货品", "物品", "名称"):
                if isinstance(value.get(key), str) and value.get(key).strip():
                    return True
        text = parsed.get("text")
        return bool(isinstance(text, str) and text.strip() and
                    not text.strip().startswith(("http://", "https://")))

    def accept_result(self, raw, identity, parsed):
        with self.lock:
            if identity in self.accepted_identities or len(self.results) >= self.target_count:
                return
            index = len(self.results)
            result = {
                "wall_index": index,
                "physical_id": identity,
                "raw": raw,
                "parsed": parsed,
                "stamp": rospy.Time.now().to_sec(),
            }
            self.accepted_identities.add(identity)
            self.results.append(result)
            count = len(self.results)
        rospy.logwarn("QR_UNIQUE_ACCEPTED count=%d/%d identity=%s", count, self.target_count, identity)
        print("\n========== CONTINUOUS QR %d/%d ==========" % (count, self.target_count), flush=True)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        print("=========================================\n", flush=True)
        event = {
            "status": "item", "target_qr_count": self.target_count,
            "detected_count": count, "result": result, "wall_results": [result],
        }
        self.item_pub.publish(String(data=json.dumps(event, ensure_ascii=False)))
        if count >= self.target_count:
            self.complete.set()

    def publish_summary(self):
        with self.lock:
            results = list(self.results)
        summary = {
            "status": "complete" if len(results) >= self.target_count else "partial",
            "target_qr_count": self.target_count,
            "detected_count": len(results),
            "wall_results": results,
            "scanner": "xunfei2026_continuous_hybrid_v1",
        }
        payload = json.dumps(summary, ensure_ascii=False)
        self.result_pub.publish(String(data=payload))
        print("\n========== QR ROOM SUMMARY ==========" , flush=True)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        print("=====================================\n", flush=True)

    def publish_zero(self, repeats=8):
        for _ in range(repeats):
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.025)

    def wait_for_camera(self):
        deadline = time.monotonic() + self.camera_timeout_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            with self.lock:
                ready = self.latest_gray is not None
            if ready:
                return True
            rospy.sleep(0.05)
        return False

    def start_camera_if_needed(self):
        if not self.start_camera:
            rospy.logwarn("QR camera autostart disabled; waiting for external image source")
            return True
        # The complete flow may already own /dev/video0.  Reuse its ROS stream
        # instead of starting a second driver and making both camera users fail.
        reuse_deadline = time.monotonic() + 1.0
        while not rospy.is_shutdown() and time.monotonic() < reuse_deadline:
            with self.lock:
                image_stamp = self.image_stamp
                ready = self.latest_gray is not None
            if ready and image_stamp is not None and (rospy.Time.now() - image_stamp).to_sec() < 1.0:
                rospy.logwarn("QR_CAMERA_REUSED topic=%s", self.image_topic)
                return True
            rospy.sleep(0.05)
        command = [
            "rosrun", "ucar_camera", "ucar_camera.py", "__name:=ucar_camera",
            "_cam_topic_name:={}".format(self.image_topic),
            "_device_path:={}".format(self.camera_device),
            "_image_width:={}".format(self.camera_width),
            "_image_height:={}".format(self.camera_height),
            "_rate:={}".format(self.camera_rate),
            "_low_exposure_absolute:={}".format(
                self.camera_low_exposure_absolute),
        ]
        try:
            self.camera_process = subprocess.Popen(command)
            rospy.logwarn(
                "QR_CAMERA_STARTED_AFTER_NAV device=%s size=%dx%d rate=%d",
                self.camera_device, self.camera_width, self.camera_height,
                self.camera_rate)
            return True
        except Exception as exc:
            rospy.logerr("failed to start QR camera after navigation: %s", exc)
            return False

    def enable_camera_low_exposure(self):
        if not self.camera_low_exposure_enabled:
            rospy.logwarn("QR_CAMERA_LOW_EXPOSURE_SKIPPED")
            return False
        try:
            rospy.wait_for_service(self.camera_exposure_service, timeout=4.0)
            response = rospy.ServiceProxy(
                self.camera_exposure_service, SetBool)(True)
            if response.success:
                rospy.logwarn(
                    "QR_CAMERA_LOW_EXPOSURE_ENABLED absolute=%d",
                    self.camera_low_exposure_absolute)
                return True
            rospy.logwarn("QR camera rejected low exposure: %s", response.message)
        except Exception as exc:
            rospy.logwarn("QR camera low exposure unavailable; continuing: %s", exc)
        return False

    def stop_camera(self):
        process = self.camera_process
        self.camera_process = None
        if process is None or process.poll() is not None:
            return
        try:
            process.terminate()
            process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass

    def run(self):
        self.publish_status("WAITING_FIRST_STAGE_NAVIGATION")
        while not rospy.is_shutdown():
            if self.nav_finished.wait(0.1):
                break
            if self.nav_failed.is_set():
                self.publish_status("FAILED_FIRST_STAGE_{}".format(self.nav_status))
                self.publish_zero()
                return 2
        if rospy.is_shutdown():
            return 1
        self.publish_status("NAVIGATION_SUCCEEDED_SETTLING")
        self.publish_zero()
        rospy.sleep(self.nav_settle_s)
        if not self.start_camera_if_needed():
            self.publish_status("FAILED_CAMERA_START")
            self.publish_zero()
            return 3
        self.enable_camera_low_exposure()
        self.center_with_lidar()
        if not self.wait_for_camera():
            self.publish_status("FAILED_CAMERA_TIMEOUT")
            self.publish_zero()
            return 3
        rospy.sleep(0.25)
        with self.lock:
            self.scan_started_at = time.monotonic()
            self.accumulated_yaw = 0.0
            self.reported_rounds = 0
            self.scan_active = True
            self.motion_mode = "SCAN"
        self.publish_status("CONTINUOUS_SCAN_STARTED")
        rospy.logwarn(
            "QR_CONTINUOUS_MOTION speed=%.3frad/s decode=%.1fHz target=%d scales=%s",
            self.angular_speed, self.decode_rate_hz, self.target_count, self.decode_scales)

        while not rospy.is_shutdown() and not self.complete.wait(0.1):
            with self.lock:
                elapsed = time.monotonic() - self.scan_started_at
                rounds = int(self.accumulated_yaw / (2.0 * math.pi))
                count = len(self.results)
                if rounds > self.reported_rounds:
                    self.reported_rounds = rounds
                    rospy.logwarn("QR_CONTINUOUS_ROUND round=%d count=%d/%d",
                                  rounds, count, self.target_count)
            if self.max_scan_s > 0.0 and elapsed >= self.max_scan_s:
                self.publish_status("FAILED_SCAN_TIMEOUT_COUNT_{}/{}".format(count, self.target_count))
                with self.lock:
                    self.scan_active = False
                    self.motion_mode = "IDLE"
                self.publish_zero()
                self.publish_summary()
                return 4

        with self.lock:
            self.scan_active = False
            self.motion_mode = "IDLE"
        self.publish_zero(12)
        if rospy.is_shutdown():
            return 1
        self.publish_status("COMPLETE_3_UNIQUE_QR_STOPPED")
        self.publish_summary()
        rospy.sleep(2.0)
        return 0

    def shutdown(self):
        with self.lock:
            self.scan_active = False
            self.motion_mode = "IDLE"
            self.center_command = (0.0, 0.0, 0.0)
        try:
            self.command_timer.shutdown()
        except Exception:
            pass
        self.publish_zero(4)
        self.stop_camera()
        self.fast_decoder.close()
        self.enhanced_decoder.close()


def main():
    rospy.init_node("xunfei2026_continuous_qr_hybrid")
    node = ContinuousQRHybrid()
    try:
        code = node.run()
        rospy.logwarn("XUNFEI2026_QR_EXIT code=%d", code)
    finally:
        node.shutdown()


if __name__ == "__main__":
    main()
