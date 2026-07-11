#!/home/ucar/venv3.9/bin/python
# -*- coding: utf-8 -*-

"""ROS image-topic wrapper for the copied RKNN factory OCR pipeline."""

import json
import logging
import os
import sys
import threading
import time
from collections import Counter, deque

# Load the Python 3.9 native modules before exposing Debian's Python 3.7
# package directory.  ROS Noetic's Python modules are pure Python here, while
# NumPy/OpenCV/RKNN must come from the working venv3.9 installation.
import cv2
import numpy as np


# ROS Noetic's logging configuration is parsed by this custom Python 3.9
# runtime.  On this image, logging.config can pass a textual level to a
# handler whose checker rejects it even though the normal level table contains
# the name.  Normalize known ROS/Python level strings locally before rospy
# configures its handlers.
_ORIGINAL_CHECK_LEVEL = logging._checkLevel
_TEXT_LOG_LEVELS = {
    "NOTSET": logging.NOTSET,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "FATAL": logging.CRITICAL,
    "CRITICAL": logging.CRITICAL,
}


def _ros_compatible_check_level(level):
    if isinstance(level, str):
        normalized = level.strip().upper()
        if normalized in _TEXT_LOG_LEVELS:
            return _TEXT_LOG_LEVELS[normalized]
    return _ORIGINAL_CHECK_LEVEL(level)


logging._checkLevel = _ros_compatible_check_level

for path in ("/usr/lib/python3/dist-packages",
             "/opt/ros/noetic/lib/python3/dist-packages"):
    if path not in sys.path:
        sys.path.append(path)

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String


OCR_DIR = "/home/ucar/instant_ws/src/iden_controller/factory_ocr_car_deploy"
if OCR_DIR not in sys.path:
    sys.path.insert(0, OCR_DIR)
os.chdir(OCR_DIR)

import camera_det_rec_final_working as ocr_impl  # noqa: E402

# RKNN's Python package registers one-letter logging level names (for example
# WARNING -> W).  rospy's stream handler indexes a table containing the full
# names, so restore the standard names and also accept the short aliases.
for _level, _name in (
        (logging.DEBUG, "DEBUG"),
        (logging.INFO, "INFO"),
        (logging.WARNING, "WARNING"),
        (logging.ERROR, "ERROR"),
        (logging.CRITICAL, "CRITICAL")):
    logging.addLevelName(_level, _name)

try:
    import rosgraph.roslogging as _roslogging
    _roslogging._logging_to_rospy_names.update({
        "D": ("DEBUG", "\033[32m"),
        "I": ("INFO", None),
        "W": ("WARN", "\033[33m"),
        "E": ("ERROR", "\033[31m"),
        "F": ("FATAL", "\033[31m"),
    })
except Exception:
    pass


def decode_image(msg):
    encoding = (msg.encoding or "").lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)
    try:
        if encoding in ("rgb8", "bgr8"):
            image = data[:msg.height * msg.width * 3].reshape(
                (msg.height, msg.width, 3))
            if encoding == "rgb8":
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return np.ascontiguousarray(image)
        if encoding in ("rgba8", "bgra8"):
            image = data[:msg.height * msg.width * 4].reshape(
                (msg.height, msg.width, 4))
            code = cv2.COLOR_RGBA2BGR if encoding == "rgba8" else cv2.COLOR_BGRA2BGR
            return cv2.cvtColor(image, code)
        if encoding in ("mono8", "8uc1"):
            gray = data[:msg.height * msg.width].reshape((msg.height, msg.width))
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    except (ValueError, IndexError):
        return None
    return None


class FactoryRoomOCRNode(object):
    def __init__(self):
        rospy.init_node("factory_room_ocr")
        self.image_topic = rospy.get_param(
            "~image_topic", "/ucar_camera/image_raw")
        self.result_topic = rospy.get_param(
            "~result_topic", "/factory_room/ocr_result")
        self.control_topic = rospy.get_param(
            "~control_topic", "/factory_room/ocr_control")
        self.health_topic = rospy.get_param(
            "~health_topic", "/factory_room/ocr_health")
        self.debug_topic = rospy.get_param(
            "~debug_topic", "/factory_room/ocr_debug")
        self.process_rate_hz = float(rospy.get_param("~process_rate_hz", 3.0))
        self.max_width = int(rospy.get_param("~max_width", 960))
        self.vote_window = int(rospy.get_param("~vote_window", 10))
        self.vote_need = int(rospy.get_param("~vote_need", 5))
        self.enabled = bool(rospy.get_param("~enabled_on_start", False))
        self.publish_debug = bool(rospy.get_param("~publish_debug", True))

        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_stamp = rospy.Time(0)
        self.last_processed_stamp = rospy.Time(0)
        self.votes = deque(maxlen=max(3, self.vote_window))
        self.det_rknn = None
        self.rec_rknn = None

        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=5, latch=True)
        self.health_pub = rospy.Publisher(
            self.health_topic, String, queue_size=2, latch=True)
        self.debug_pub = rospy.Publisher(
            self.debug_topic, Image, queue_size=1)
        rospy.Subscriber(self.image_topic, Image, self.image_callback,
                         queue_size=1, buff_size=4 * 1024 * 1024)
        rospy.Subscriber(self.control_topic, String, self.control_callback,
                         queue_size=5)

    def image_callback(self, msg):
        image = decode_image(msg)
        if image is None:
            rospy.logwarn_throttle(2.0, "factory OCR unsupported image encoding: %s",
                                   msg.encoding)
            return
        if self.max_width > 0 and image.shape[1] > self.max_width:
            scale = float(self.max_width) / float(image.shape[1])
            image = cv2.resize(
                image, (self.max_width, max(1, int(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA)
        with self.lock:
            self.latest_frame = image
            self.latest_stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()

    def control_callback(self, msg):
        command = (msg.data or "").strip().lower()
        with self.lock:
            if command in ("enable", "start", "scan", "on"):
                self.enabled = True
                self.votes.clear()
            elif command in ("disable", "stop", "off"):
                self.enabled = False
                self.votes.clear()
            elif command in ("reset", "clear"):
                self.votes.clear()
        rospy.loginfo("FACTORY_OCR_CONTROL command=%s enabled=%s",
                      command, str(self.enabled))

    def initialize_models(self):
        self.health_pub.publish(String(data="loading"))
        rospy.logwarn("FACTORY_OCR_MODEL_LOADING dir=%s", OCR_DIR)
        self.det_rknn = ocr_impl.load_rknn(ocr_impl.DET_MODEL, "det model")
        self.rec_rknn = ocr_impl.load_rknn(ocr_impl.REC_MODEL, "rec model")
        self.health_pub.publish(String(data="ready"))
        rospy.logwarn("FACTORY_OCR_READY image=%s result=%s",
                      self.image_topic, self.result_topic)

    def snapshot(self):
        with self.lock:
            if (not self.enabled or self.latest_frame is None or
                    self.latest_stamp == self.last_processed_stamp):
                return None, None
            frame = self.latest_frame.copy()
            stamp = self.latest_stamp
            self.last_processed_stamp = stamp
        return frame, stamp

    @staticmethod
    def best_bbox(best):
        if not best:
            return None
        box = best.get("box")
        if box is None or len(box) == 0:
            return None
        points = np.asarray(box, dtype=np.float32).reshape((-1, 2))
        return [float(points[:, 0].min()), float(points[:, 1].min()),
                float(points[:, 0].max()), float(points[:, 1].max())]

    def publish_debug_image(self, frame, det_items, best, payload):
        if not self.publish_debug:
            return
        debug = frame.copy()
        for item in det_items:
            ocr_impl.draw_poly(debug, item["box"], (0, 255, 0), 2)
        if best is not None:
            ocr_impl.draw_poly(debug, best["box"], (0, 0, 255), 3)
        cv2.putText(
            debug,
            "%s votes=%d stable=%s" % (
                ocr_impl.SHORT.get(payload["label"], "unknown"),
                payload["votes"], str(payload["stable"])),
            (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        rgb = cv2.cvtColor(debug, cv2.COLOR_BGR2RGB)
        out = Image()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = "factory_ocr"
        out.height, out.width = rgb.shape[:2]
        out.encoding = "rgb8"
        out.is_bigendian = False
        out.step = out.width * 3
        out.data = rgb.tobytes()
        self.debug_pub.publish(out)

    def process_once(self):
        frame, stamp = self.snapshot()
        if frame is None:
            return
        try:
            det_items, layout, det_max, _ = ocr_impl.run_det(
                self.det_rknn, frame)
            crops, rejected = ocr_impl.build_good_crops(frame, det_items)
            results = ocr_impl.recognize_crops(self.rec_rknn, crops)
            frame_label, raw_text, score, best = ocr_impl.decide_frame(results)
            with self.lock:
                self.votes.append(frame_label)
                counts = Counter(label for label in self.votes
                                 if label != "unknown")
                lead_label, lead_count = (counts.most_common(1)[0]
                                          if counts else ("unknown", 0))
            stable = (lead_label != "unknown" and
                      lead_count >= self.vote_need and
                      frame_label == lead_label)
            bbox = self.best_bbox(best)
            payload = {
                "label": lead_label if stable else frame_label,
                "frame_label": frame_label,
                "raw_text": raw_text,
                "score": float(score),
                "stable": bool(stable),
                "votes": int(lead_count),
                "vote_window": len(self.votes),
                "bbox": bbox,
                "image_width": int(frame.shape[1]),
                "image_height": int(frame.shape[0]),
                "det_count": len(det_items),
                "crop_count": len(crops),
                "reject_count": len(rejected),
                "det_layout": layout,
                "det_score_max": float(det_max),
                "image_stamp": stamp.to_sec(),
                "stamp": time.time(),
            }
            self.result_pub.publish(String(
                data=json.dumps(payload, ensure_ascii=False)))
            self.publish_debug_image(frame, det_items, best, payload)
            if stable:
                rospy.logwarn_throttle(
                    1.0,
                    "FACTORY_OCR_STABLE label=%s raw=%s votes=%d/%d score=%.2f",
                    lead_label, raw_text, lead_count, len(self.votes), score)
            else:
                rospy.loginfo_throttle(
                    2.0,
                    "FACTORY_OCR_SCANNING frame=%s lead=%s votes=%d/%d det=%d crops=%d",
                    frame_label, lead_label, lead_count, len(self.votes),
                    len(det_items), len(crops))
        except Exception as exc:
            self.health_pub.publish(String(data="runtime_error: %s" % exc))
            rospy.logerr_throttle(2.0, "factory OCR frame failed: %s", exc)

    def shutdown(self):
        if self.det_rknn is not None:
            self.det_rknn.release()
        if self.rec_rknn is not None:
            self.rec_rknn.release()

    def run(self):
        self.initialize_models()
        rospy.on_shutdown(self.shutdown)
        rate = rospy.Rate(max(0.5, self.process_rate_hz))
        while not rospy.is_shutdown():
            self.process_once()
            rate.sleep()


if __name__ == "__main__":
    try:
        FactoryRoomOCRNode().run()
    except rospy.ROSInterruptException:
        pass
