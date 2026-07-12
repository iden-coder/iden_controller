#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unique QR scan with lidar wall-normal and QR centerline alignment."""

import ctypes
import math
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan

from global_first_nav_qr_room_scan_unique import UniqueQRRoomScanner
from qr_room_spin_scan import clamp


class WallAlignedUniqueQRScanner(UniqueQRRoomScanner):
    def __init__(self):
        self.wall_model = None
        self.wall_stamp = 0.0
        self.left_clear = float("inf")
        self.right_clear = float("inf")
        self.current_xy = None
        super(WallAlignedUniqueQRScanner, self).__init__()
        self.setup_zbar_location_api()
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.wall_sector = math.radians(float(rospy.get_param(
            "~qr_wall_sector_deg", 62.0)))
        self.wall_inlier = float(rospy.get_param(
            "~qr_wall_inlier_m", 0.025))
        self.wall_min_span = float(rospy.get_param(
            "~qr_wall_min_span_m", 0.30))
        self.wall_min_inliers = int(rospy.get_param(
            "~qr_wall_min_inliers", 18))
        self.wall_heading_limit = math.radians(float(rospy.get_param(
            "~qr_wall_heading_limit_deg", 42.0)))
        self.wall_tolerance = math.radians(float(rospy.get_param(
            "~qr_wall_tolerance_deg", 2.8)))
        self.wall_align_timeout = float(rospy.get_param(
            "~qr_wall_align_timeout_s", 10.0))
        self.wall_min_wz = float(rospy.get_param(
            "~qr_wall_min_wz", 0.12))
        self.wall_max_wz = float(rospy.get_param(
            "~qr_wall_max_wz", 0.24))
        self.qr_center_tolerance_px = float(rospy.get_param(
            "~qr_center_tolerance_px", 24.0))
        self.qr_center_stable_frames = int(rospy.get_param(
            "~qr_center_stable_frames", 2))
        self.qr_lateral_kp = float(rospy.get_param(
            "~qr_lateral_kp", 0.22))
        self.qr_lateral_min_speed = float(rospy.get_param(
            "~qr_lateral_min_speed", 0.014))
        self.qr_lateral_max_speed = float(rospy.get_param(
            "~qr_lateral_max_speed", 0.060))
        self.qr_lateral_limit = float(rospy.get_param(
            "~qr_lateral_limit_m", 0.20))
        self.qr_center_timeout = float(rospy.get_param(
            "~qr_center_timeout_s", 6.0))
        self.qr_side_hard = float(rospy.get_param(
            "~qr_side_hard_m", 0.17))
        self.return_tolerance = float(rospy.get_param(
            "~qr_return_tolerance_m", 0.018))
        self.return_timeout = float(rospy.get_param(
            "~qr_return_timeout_s", 5.0))
        rospy.Subscriber(self.scan_topic, LaserScan, self.cb_scan_wall,
                         queue_size=1)

    def setup_zbar_location_api(self):
        c_void_p = ctypes.c_void_p
        lib = self.scanner.lib
        lib.zbar_symbol_get_loc_size.argtypes = [c_void_p]
        lib.zbar_symbol_get_loc_size.restype = ctypes.c_uint
        lib.zbar_symbol_get_loc_x.argtypes = [c_void_p, ctypes.c_uint]
        lib.zbar_symbol_get_loc_x.restype = ctypes.c_int
        lib.zbar_symbol_get_loc_y.argtypes = [c_void_p, ctypes.c_uint]
        lib.zbar_symbol_get_loc_y.restype = ctypes.c_int

    def scan_gray_detailed(self, gray):
        if gray is None or gray.size == 0:
            return []
        gray = np.ascontiguousarray(gray)
        height, width = gray.shape[:2]
        lib = self.scanner.lib
        image = lib.zbar_image_create()
        if not image:
            return []
        try:
            lib.zbar_image_set_format(image, self.scanner.Y800)
            lib.zbar_image_set_size(image, width, height)
            pointer = gray.ctypes.data_as(ctypes.c_void_p)
            lib.zbar_image_set_data(
                image, pointer, ctypes.c_ulong(gray.nbytes), None)
            if lib.zbar_scan_image(self.scanner.scanner, image) <= 0:
                return []
            results = []
            symbol = lib.zbar_image_first_symbol(image)
            while symbol:
                data = lib.zbar_symbol_get_data(symbol)
                if data:
                    count = int(lib.zbar_symbol_get_loc_size(symbol))
                    points = [
                        (float(lib.zbar_symbol_get_loc_x(symbol, index)),
                         float(lib.zbar_symbol_get_loc_y(symbol, index)))
                        for index in range(count)
                    ]
                    center = None
                    if points:
                        center = (
                            sum(point[0] for point in points) / len(points),
                            sum(point[1] for point in points) / len(points))
                    results.append({
                        "raw": data.decode("utf-8", errors="replace"),
                        "points": points,
                        "center": center,
                        "width": int(width),
                        "height": int(height),
                    })
                symbol = lib.zbar_symbol_next(symbol)
            return results
        finally:
            lib.zbar_image_destroy(image)

    def cb_odom(self, msg):
        super(WallAlignedUniqueQRScanner, self).cb_odom(msg)
        with self.lock:
            self.current_xy = (msg.pose.pose.position.x,
                               msg.pose.pose.position.y)

    def cb_scan_wall(self, msg):
        points = []
        left = []
        right = []
        for index, distance in enumerate(msg.ranges):
            if (not math.isfinite(distance) or distance < msg.range_min or
                    distance > msg.range_max):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            if abs(angle) <= self.wall_sector and distance <= 2.0:
                points.append((distance * math.cos(angle),
                               distance * math.sin(angle)))
            if math.radians(48.0) <= angle <= math.radians(112.0):
                left.append(distance)
            if math.radians(-112.0) <= angle <= math.radians(-48.0):
                right.append(distance)
        model = self.fit_wall(points)
        with self.lock:
            if model is not None:
                previous = self.wall_model
                if previous is not None:
                    alpha = 0.35
                    delta = math.atan2(
                        math.sin(model["heading"] - previous["heading"]),
                        math.cos(model["heading"] - previous["heading"]))
                    model["heading"] = previous["heading"] + alpha * delta
                    model["distance"] = previous["distance"] + alpha * (
                        model["distance"] - previous["distance"])
                self.wall_model = model
                self.wall_stamp = time.time()
            self.left_clear = min(left) if left else float("inf")
            self.right_clear = min(right) if right else float("inf")

    def fit_wall(self, points):
        if len(points) < self.wall_min_inliers:
            return None
        data = np.asarray(points, dtype=np.float64)
        count = len(data)
        stride = max(1, count // 34)
        best = None
        best_score = -1.0
        for first in range(0, count, stride):
            for second in range(first + max(3, stride), count, stride):
                delta = data[second] - data[first]
                length = float(np.linalg.norm(delta))
                if length < 0.55 * self.wall_min_span:
                    continue
                normal = np.array([-delta[1], delta[0]]) / length
                residual = np.abs((data - data[first]).dot(normal))
                indices = np.flatnonzero(residual <= self.wall_inlier)
                if len(indices) < self.wall_min_inliers:
                    continue
                tangent = delta / length
                projection = data[indices].dot(tangent)
                span = float(projection.max() - projection.min())
                if span < self.wall_min_span:
                    continue
                score = len(indices) + 12.0 * span
                if score > best_score:
                    best_score = score
                    best = indices
        if best is None:
            return None
        inliers = data[best]
        center = inliers.mean(axis=0)
        values, vectors = np.linalg.eigh(np.cov((inliers - center).T))
        normal = vectors[:, int(np.argmin(values))]
        if float(normal.dot(center)) < 0.0:
            normal = -normal
        heading = math.atan2(float(normal[1]), float(normal[0]))
        if abs(heading) > self.wall_heading_limit:
            return None
        return {
            "heading": heading,
            "distance": float(normal.dot(center)),
            "inliers": int(len(inliers)),
        }

    def wall_snapshot(self):
        with self.lock:
            return (None if self.wall_model is None else dict(self.wall_model),
                    self.wall_stamp, self.left_clear, self.right_clear,
                    self.current_xy, self.current_yaw)

    def align_wall(self):
        stable = 0
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < self.wall_align_timeout:
            model, stamp, _, _, _, _ = self.wall_snapshot()
            if model is None or time.time() - stamp > 1.0:
                self.publish_zero()
                stable = 0
                rate.sleep()
                continue
            error = model["heading"]
            if abs(error) <= self.wall_tolerance:
                stable += 1
                wz = 0.0
            else:
                stable = 0
                wz = clamp(1.35 * error, -self.wall_max_wz,
                           self.wall_max_wz)
                if abs(wz) < self.wall_min_wz:
                    wz = math.copysign(self.wall_min_wz, error)
            self.publish_turn(wz)
            rospy.logwarn_throttle(
                0.35,
                "QR_WALL_ALIGN distance=%.3f error=%.2fdeg stable=%d/3 cmd_w=%.3f",
                model["distance"], math.degrees(error), stable, wz)
            if stable >= 3:
                self.publish_zero()
                rospy.logwarn("QR_WALL_ALIGNED error=%.2fdeg",
                              math.degrees(error))
                return True
            rate.sleep()
        self.publish_zero()
        rospy.logwarn("QR_WALL_ALIGN_DEGRADED; continuing safe scan")
        return False

    def detect_qr_detail(self):
        with self.lock:
            gray = None if self.latest_gray is None else self.latest_gray.copy()
        if gray is None:
            return None
        decoded = self.scan_gray_detailed(gray)
        if not decoded:
            return None
        detail = decoded[0]
        detail["raw"] = detail["raw"].strip()
        return detail

    @staticmethod
    def projected_lateral(anchor_xy, anchor_yaw, current_xy):
        if anchor_xy is None or current_xy is None or anchor_yaw is None:
            return None
        dx = current_xy[0] - anchor_xy[0]
        dy = current_xy[1] - anchor_xy[1]
        return -dx * math.sin(anchor_yaw) + dy * math.cos(anchor_yaw)

    def center_detected_qr(self, initial):
        if initial.get("center") is None:
            return initial, False
        self.align_wall()
        _, _, _, _, anchor_xy, anchor_yaw = self.wall_snapshot()
        last = initial
        last_seen = time.time()
        stable = 0
        start = time.time()
        rate = rospy.Rate(12)
        while not rospy.is_shutdown() and time.time() - start < self.qr_center_timeout:
            model, stamp, left, right, current_xy, _ = self.wall_snapshot()
            if (model is None or time.time() - stamp > 1.0 or
                    abs(model["heading"]) > math.radians(6.0)):
                self.publish_zero()
                self.align_wall()
                stable = 0
                rate.sleep()
                continue
            detail = self.detect_qr_detail()
            if detail is not None and detail.get("center") is not None:
                last = detail
                last_seen = time.time()
            elif time.time() - last_seen > 1.2:
                rospy.logwarn("QR_CENTER target lost after wall alignment")
                break
            center = last.get("center")
            if center is None:
                break
            error = center[0] - 0.5 * float(last["width"])
            lateral = self.projected_lateral(
                anchor_xy, anchor_yaw, current_xy)
            if lateral is not None and abs(lateral) > self.qr_lateral_limit:
                rospy.logwarn("QR_CENTER lateral limit %.3fm", lateral)
                break
            if abs(error) <= self.qr_center_tolerance_px:
                stable += 1
                command_y = 0.0
            else:
                stable = 0
                normalized = error / max(1.0, 0.5 * float(last["width"]))
                command_y = clamp(
                    -self.qr_lateral_kp * normalized,
                    -self.qr_lateral_max_speed,
                    self.qr_lateral_max_speed)
                if abs(command_y) < self.qr_lateral_min_speed:
                    command_y = math.copysign(
                        self.qr_lateral_min_speed, command_y)
                clearance = left if command_y > 0.0 else right
                if clearance < self.qr_side_hard:
                    rospy.logwarn(
                        "QR_CENTER side blocked clear=%.3fm", clearance)
                    command_y = 0.0
                    break
            command = Twist()
            command.linear.y = command_y
            self.cmd_pub.publish(command)
            rospy.logwarn_throttle(
                0.30,
                "QR_CENTERLINE error=%.1fpx stable=%d/%d lateral=%s cmd_y=%.3f",
                error, stable, self.qr_center_stable_frames,
                "none" if lateral is None else "%.3f" % lateral,
                command_y)
            if stable >= self.qr_center_stable_frames:
                self.publish_zero()
                return last, True
            rate.sleep()
        self.publish_zero()
        return last, False

    def return_to_anchor(self, anchor_xy, anchor_yaw):
        if anchor_xy is None or anchor_yaw is None:
            return False
        start = time.time()
        rate = rospy.Rate(15)
        while not rospy.is_shutdown() and time.time() - start < self.return_timeout:
            model, stamp, left, right, current_xy, _ = self.wall_snapshot()
            lateral = self.projected_lateral(
                anchor_xy, anchor_yaw, current_xy)
            if lateral is None:
                break
            if abs(lateral) <= self.return_tolerance:
                self.publish_zero()
                return True
            if (model is None or time.time() - stamp > 1.0 or
                    abs(model["heading"]) > math.radians(6.0)):
                self.publish_zero()
                self.align_wall()
                continue
            command_y = clamp(-1.1 * lateral, -0.06, 0.06)
            clearance = left if command_y > 0.0 else right
            if clearance < self.qr_side_hard:
                rospy.logwarn("QR_RETURN blocked clear=%.3fm", clearance)
                break
            command = Twist()
            command.linear.y = command_y
            self.cmd_pub.publish(command)
            rate.sleep()
        self.publish_zero()
        return False

    def scan_current_view(self, wall_index, scan_s):
        start = rospy.Time.now()
        rate = rospy.Rate(self.process_rate_hz)
        while not rospy.is_shutdown():
            if (rospy.Time.now() - start).to_sec() > scan_s:
                return None
            detail = self.detect_qr_detail()
            if detail is not None:
                refined, aligned = self.center_detected_qr(detail)
                raw = refined["raw"]
                parsed = self.parse_payload(raw)
                result = {
                    "wall_index": wall_index,
                    "raw": raw,
                    "parsed": parsed,
                    "qr_center_aligned": bool(aligned),
                    "qr_center_px": refined.get("center"),
                    "stamp": rospy.Time.now().to_sec(),
                }
                self.store_wall_result(result)
                self.print_wall_result(result)
                return result
            rate.sleep()

    def scan_wall(self, wall_index):
        self.align_wall()
        _, _, _, _, anchor_xy, anchor_yaw = self.wall_snapshot()
        try:
            settle_s = (self.first_wall_settle_s if wall_index == 0
                        else self.settle_s)
            scan_s = (self.first_wall_scan_s if wall_index == 0
                      else self.scan_per_wall_s)
            rospy.sleep(settle_s)
            result = self.scan_current_view(wall_index, scan_s)
            if result is not None or not self.micro_sweep_enabled:
                return result
            sweep_deg = (self.first_wall_micro_sweep_deg
                         if wall_index == 0 else self.micro_sweep_deg)
            sweep_scan_s = (self.first_wall_micro_sweep_scan_s
                            if wall_index == 0 else self.micro_sweep_scan_s)
            sweep = math.radians(abs(sweep_deg))
            for delta in (sweep, -2.0 * sweep, sweep):
                self.turn_relative(delta)
                rospy.sleep(max(0.15, 0.5 * self.settle_s))
                result = self.scan_current_view(wall_index, sweep_scan_s)
                if result is not None:
                    return result
            rospy.logwarn("wall_%d no QR after wall-aligned sweep", wall_index)
            return None
        finally:
            self.align_wall()
            self.return_to_anchor(anchor_xy, anchor_yaw)
            self.align_wall()


def main():
    node = WallAlignedUniqueQRScanner()
    node.run()
    rospy.loginfo("wall-aligned unique QR scan finished; node stays alive")
    rospy.spin()


if __name__ == "__main__":
    main()
