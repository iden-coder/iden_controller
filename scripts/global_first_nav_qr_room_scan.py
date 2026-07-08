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


def fix_mojibake_text(text):
    if not isinstance(text, str):
        return text
    # Some QR URL responses contain UTF-8 bytes that have already been decoded
    # as GBK. Repair only when the conversion is possible and visibly improves.
    suspicious = ("鐢", "佃", "剳", "澶", "姣", "涘", "肪", "棣", "欒", "晧")
    if not any(token in text for token in suspicious):
        return text
    try:
        repaired = text.encode("gbk").decode("utf-8")
    except Exception:
        return text
    if repaired and repaired != text:
        return repaired
    return text


def fix_mojibake(value):
    if isinstance(value, str):
        return fix_mojibake_text(value)
    if isinstance(value, list):
        return [fix_mojibake(item) for item in value]
    if isinstance(value, dict):
        return {key: fix_mojibake(item) for key, item in value.items()}
    return value


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
        self.item_topic = rospy.get_param(
            "~item_topic", "/qr_room_scan_item")
        self.control_topic = rospy.get_param(
            "~control_topic", "/qr_room_scan_control")
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
        self.fetch_timeout_s = float(rospy.get_param("~fetch_timeout_s", 6.0))
        self.fetch_retries = max(1, int(rospy.get_param("~fetch_retries", 3)))
        self.fetch_retry_delay_s = float(rospy.get_param(
            "~fetch_retry_delay_s", 0.4))
        self.max_frame_width = int(rospy.get_param("~max_frame_width", 960))
        self.micro_sweep_enabled = bool(rospy.get_param(
            "~micro_sweep_enabled", True))
        self.micro_sweep_deg = float(rospy.get_param("~micro_sweep_deg", 12.0))
        self.first_wall_micro_sweep_deg = float(rospy.get_param(
            "~first_wall_micro_sweep_deg", max(self.micro_sweep_deg, 26.0)))
        self.micro_sweep_scan_s = float(rospy.get_param(
            "~micro_sweep_scan_s", 1.2))
        self.first_wall_micro_sweep_scan_s = float(rospy.get_param(
            "~first_wall_micro_sweep_scan_s", max(self.micro_sweep_scan_s, 1.8)))
        self.micro_sweep_speed = abs(float(rospy.get_param(
            "~micro_sweep_speed", 0.28)))
        self.after_item_decision_wait_s = float(rospy.get_param(
            "~after_item_decision_wait_s", 20.0))

        self.lock = threading.Lock()
        self.nav_done = self.start_immediately
        self.nav_status_text = ""
        self.latest_gray = None
        self.latest_image_time = rospy.Time(0)
        self.current_yaw = None
        self.odom_time = rospy.Time(0)

        self.results = []
        self.wall_results = [None for _ in range(self.wall_count)]
        self.valid_wall_indices = set()
        self.stop_requested = False
        self.control_seq = 0
        self.last_control_command = ""
        self.scanner = ZBarScanner()

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=1, latch=True)
        self.item_pub = rospy.Publisher(self.item_topic, String, queue_size=5)
        rospy.Subscriber(self.nav_status_topic, String, self.cb_nav_status, queue_size=5)
        rospy.Subscriber(self.image_topic, Image, self.cb_image, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.cb_odom, queue_size=1)
        rospy.Subscriber(self.control_topic, String, self.cb_control, queue_size=5)

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

    def cb_control(self, msg):
        command = (msg.data or "").strip().lower()
        with self.lock:
            if command:
                self.control_seq += 1
                self.last_control_command = command
            if command in ("stop", "done", "success", "matched"):
                self.stop_requested = True
        if command in ("stop", "done", "success", "matched"):
            rospy.logwarn("QR_SCAN_CONTROL_STOP received; stopping further wall scan")
            self.publish_zero()
        if command in ("continue", "next", "no_match"):
            rospy.loginfo("QR_SCAN_CONTROL_CONTINUE received; scanning next wall")

    def should_stop_scan(self):
        with self.lock:
            return self.stop_requested

    def get_control_seq(self):
        with self.lock:
            return self.control_seq

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
                valid_payload = self.store_wall_result(result)
                if valid_payload:
                    rospy.loginfo("wall_%d QR accepted with item payload", wall_index)
                else:
                    rospy.logwarn(
                        "wall_%d QR decoded but item payload is unavailable; will retry this wall in the next scan round",
                        wall_index)
                self.print_wall_result(result)
                return result
            rate.sleep()

    def result_has_item_payload(self, result):
        if not isinstance(result, dict):
            return False
        parsed = result.get("parsed")
        if not isinstance(parsed, dict):
            return False
        data = parsed.get("json")
        if isinstance(data, dict):
            for key in ("result", "name", "item", "goods", "product", "货品", "物品", "名称"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return True
        text = parsed.get("text")
        if isinstance(text, str) and text.strip():
            stripped = text.strip()
            if not stripped.startswith(("http://", "https://")):
                return True
        return False

    def store_wall_result(self, result):
        wall_index = result.get("wall_index")
        if not isinstance(wall_index, int) or wall_index < 0 or wall_index >= self.wall_count:
            return False
        self.wall_results[wall_index] = result
        if not self.result_has_item_payload(result):
            return False
        if wall_index in self.valid_wall_indices:
            rospy.loginfo("wall_%d valid QR already counted; ignoring duplicate", wall_index)
            return True
        self.valid_wall_indices.add(wall_index)
        self.results.append(result)
        return True

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

        sweep_deg = (self.first_wall_micro_sweep_deg
                     if wall_index == 0 else self.micro_sweep_deg)
        sweep_scan_s = (self.first_wall_micro_sweep_scan_s
                        if wall_index == 0 else self.micro_sweep_scan_s)
        sweep = math.radians(abs(sweep_deg))
        rospy.logwarn(
            "wall_%d no QR in center view; trying +/-%.1fdeg micro sweep",
            wall_index, sweep_deg)
        self.turn_relative(sweep)
        rospy.sleep(max(0.15, self.settle_s * 0.5))
        result = self.scan_current_view(wall_index, sweep_scan_s)
        self.turn_relative(-2.0 * sweep)
        rospy.sleep(max(0.15, self.settle_s * 0.5))
        if result is None:
            result = self.scan_current_view(wall_index, sweep_scan_s)
        self.turn_relative(sweep)
        if result is None:
            rospy.logwarn("wall_%d no QR detected after micro sweep", wall_index)
        return result

    def fetch_qr_url(self, url):
        try:
            import requests
        except Exception as exc:
            return None, "requests import failed: {}".format(exc)

        last_error = ""
        for attempt in range(1, self.fetch_retries + 1):
            try:
                rospy.loginfo("QR_URL_FETCH attempt=%d/%d timeout=%.1fs url=%s",
                              attempt, self.fetch_retries,
                              self.fetch_timeout_s, url)
                response = requests.get(url, timeout=self.fetch_timeout_s)
                response.raise_for_status()
                rospy.loginfo("QR_URL_FETCH_OK attempt=%d/%d bytes=%d",
                              attempt, self.fetch_retries, len(response.content))
                return response.content.decode("utf-8", errors="replace"), None
            except Exception as exc:
                last_error = str(exc)
                rospy.logwarn("QR_URL_FETCH_FAIL attempt=%d/%d error=%s",
                              attempt, self.fetch_retries, last_error)
                if attempt < self.fetch_retries and self.fetch_retry_delay_s > 0.0:
                    rospy.sleep(self.fetch_retry_delay_s)
        return None, last_error

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
            text, error = self.fetch_qr_url(raw)
            if text is not None:
                try:
                    parsed["json"] = fix_mojibake(json.loads(text))
                    parsed["type"] = "url_json"
                    parsed["text"] = None
                except Exception:
                    parsed["text"] = fix_mojibake_text(text)
                    parsed["type"] = "url_text"
            else:
                parsed["type"] = "url_fetch_failed"
                parsed["text"] = None
                parsed["error"] = error
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
            "status": ("stopped_by_decision" if self.should_stop_scan()
                       else "complete" if len(self.results) >= self.target_qr_count
                       else "partial"),
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

    def publish_item(self, result):
        event = {
            "status": "item",
            "target_qr_count": self.target_qr_count,
            "detected_count": len(self.results),
            "result": result,
            "wall_results": [result],
        }
        msg = String()
        msg.data = json.dumps(event, ensure_ascii=False)
        self.item_pub.publish(msg)
        rospy.loginfo("QR_ITEM_PUBLISHED wall=%s detected_count=%d",
                      result.get("wall_index"), len(self.results))

    def wait_for_decision_after_item(self, start_seq=None):
        if self.after_item_decision_wait_s <= 0.0:
            return self.should_stop_scan()
        if start_seq is None:
            start_seq = self.get_control_seq()
        deadline = rospy.Time.now() + rospy.Duration(self.after_item_decision_wait_s)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self.publish_zero()
            if self.should_stop_scan():
                return True
            with self.lock:
                if (self.control_seq != start_seq and
                        self.last_control_command in ("continue", "next", "no_match")):
                    return False
            rate.sleep()
        return self.should_stop_scan()

    def run(self):
        try:
            if not self.wait_for_nav_done():
                return
            if not self.wait_for_camera():
                rospy.logerr("no camera image received; cannot scan")
                return
            self.stop_navigation_node()
            rospy.loginfo("starting QR room scan with cmd topic %s", self.cmd_vel_topic)
            wall = 0
            round_index = 1
            round_start_count = len(self.results)
            rospy.logwarn("QR_SCAN_ROUND_START round=%d valid=%d/%d",
                          round_index, len(self.results), self.target_qr_count)
            while not rospy.is_shutdown():
                if self.should_stop_scan():
                    break
                if len(self.results) >= self.target_qr_count:
                    break
                if wall in self.valid_wall_indices:
                    rospy.loginfo("wall_%d already has a valid QR item; skipping", wall)
                    result = None
                else:
                    result = self.scan_wall(wall)
                if result is not None and self.result_has_item_payload(result):
                    start_seq = self.get_control_seq()
                    self.publish_item(result)
                    if self.after_item_decision_wait_s > 0.0 and self.wait_for_decision_after_item(start_seq):
                        break
                if self.should_stop_scan():
                    break
                if len(self.results) >= self.target_qr_count:
                    break
                next_wall = (wall + 1) % self.wall_count
                if next_wall == 0 and len(self.results) == round_start_count:
                    rospy.logwarn(
                        "QR_SCAN_ROUND_NO_NEW_ITEMS round=%d valid=%d/%d; continuing until all QR items are collected",
                        round_index, len(self.results), self.target_qr_count)
                self.rotate_90()
                wall = next_wall
                if wall == 0:
                    round_index += 1
                    round_start_count = len(self.results)
                    rospy.logwarn("QR_SCAN_ROUND_START round=%d valid=%d/%d",
                                  round_index, len(self.results), self.target_qr_count)
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
