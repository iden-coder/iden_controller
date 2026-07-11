#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deliver the selected QR item to the OCR-matched workshop parking box."""

import json
import math
import threading
import time

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from factory_room_vision_core import GroundSquareDetector, clamp, decode_ros_image


def norm_angle(angle):
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def as_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true", "yes", "on")


class FactoryRoomDeliveryManager(object):
    def __init__(self):
        rospy.init_node("factory_room_delivery_manager")

        self.task_result_topic = rospy.get_param(
            "~task_result_topic", "/factory/subtask1_result")
        self.tts_topic = rospy.get_param("~tts_topic", "/factory/tts_text")
        self.result_topic = rospy.get_param(
            "~result_topic", "/factory/delivery_result")
        self.state_topic = rospy.get_param(
            "~state_topic", "/factory/room_task_state")
        self.ocr_result_topic = rospy.get_param(
            "~ocr_result_topic", "/factory_room/ocr_result")
        self.ocr_control_topic = rospy.get_param(
            "~ocr_control_topic", "/factory_room/ocr_control")
        self.image_topic = rospy.get_param(
            "~image_topic", "/ucar_camera/image_raw")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.pose_topic = rospy.get_param("~pose_topic", "/amcl_pose")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel_raw")
        self.move_base_action = rospy.get_param(
            "~move_base_action", "/factory_room/move_base")
        self.clear_costmaps_service = rospy.get_param(
            "~clear_costmaps_service",
            "/factory_room/move_base/clear_costmaps")

        self.start_after_tts_s = float(rospy.get_param("~start_after_tts_s", 7.0))
        self.navigate_start_pose = as_bool(rospy.get_param(
            "~navigate_start_pose", True))
        self.nav_goal_timeout_s = float(rospy.get_param("~nav_goal_timeout_s", 75.0))
        self.nav_stuck_timeout_s = float(rospy.get_param("~nav_stuck_timeout_s", 9.0))
        self.nav_progress_m = float(rospy.get_param("~nav_progress_m", 0.07))
        self.nav_goal_retries = int(rospy.get_param("~nav_goal_retries", 3))
        self.search_cycles = int(rospy.get_param("~search_cycles", 3))
        self.adaptive_search_enabled = as_bool(rospy.get_param(
            "~adaptive_search_enabled", True))
        self.adaptive_ring_radius_m = float(rospy.get_param(
            "~adaptive_ring_radius_m", 0.62))
        self.adaptive_min_spacing_m = float(rospy.get_param(
            "~adaptive_min_spacing_m", 0.32))
        self.adaptive_point_limit = int(rospy.get_param(
            "~adaptive_point_limit", 6))
        self.mission_timeout_s = float(rospy.get_param("~mission_timeout_s", 600.0))
        self.scan_step_deg = float(rospy.get_param("~scan_step_deg", 45.0))
        self.scan_turn_speed = float(rospy.get_param("~scan_turn_speed", 0.36))
        self.scan_settle_s = float(rospy.get_param("~scan_settle_s", 0.75))
        self.spin_clearance_m = float(rospy.get_param("~spin_clearance_m", 0.27))
        self.image_mirrored = as_bool(rospy.get_param("~image_mirrored", True))
        self.wall_standoff_m = float(rospy.get_param("~wall_standoff_m", 0.68))
        self.wall_final_distance_m = float(rospy.get_param(
            "~wall_final_distance_m", 0.29))
        self.square_front_hard_stop_m = float(rospy.get_param(
            "~square_front_hard_stop_m", 0.235))
        self.square_entry_distance_m = float(rospy.get_param(
            "~square_entry_distance_m", 0.25))
        self.square_approach_timeout_s = float(rospy.get_param(
            "~square_approach_timeout_s", 28.0))
        self.square_lateral_tolerance_m = float(rospy.get_param(
            "~square_lateral_tolerance_m", 0.09))
        self.square_speed_fast = float(rospy.get_param(
            "~square_speed_fast", 0.10))
        self.square_speed_slow = float(rospy.get_param(
            "~square_speed_slow", 0.055))

        start = rospy.get_param("~start_pose", [-0.85, -1.41, -1.5707963268])
        self.start_pose = (float(start[0]), float(start[1]), float(start[2]))
        raw_points = rospy.get_param("~observation_points", [
            {"name": "d1", "x": -1.63, "y": -2.57, "yaw": math.pi},
            {"name": "d2", "x": 0.41, "y": -1.60, "yaw": math.pi / 2.0},
            {"name": "d3", "x": 2.54, "y": -2.81, "yaw": -math.pi / 2.0},
        ])
        self.fixed_observation_points = []
        for index, point in enumerate(raw_points):
            self.fixed_observation_points.append({
                "name": str(point.get("name", "p%d" % index)),
                "x": float(point["x"]),
                "y": float(point["y"]),
                "yaw": float(point.get("yaw", 0.0)),
            })
        self.adaptive_observation_points = self.build_adaptive_search_points(
            self.fixed_observation_points)
        self.observation_points = (
            list(self.fixed_observation_points) +
            list(self.adaptive_observation_points))

        self.lock = threading.Lock()
        self.launch_time = time.time()
        self.task_payload = None
        self.mission_running = False
        self.abort_requested = False
        self.target_item = ""
        self.target_warehouse = ""
        self.latest_ocr = None
        self.target_ocr_time = 0.0
        self.target_found = threading.Event()
        self.latest_image = None
        self.latest_image_time = 0.0
        self.pose = None
        self.pose_time = 0.0
        self.odom_pose = None
        self.odom_yaw = None
        self.scan_ranges = []
        self.front_min = float("inf")
        self.front_wall = float("inf")
        self.all_min = float("inf")

        self.square_detector = GroundSquareDetector()
        self.move_client = actionlib.SimpleActionClient(
            self.move_base_action, MoveBaseAction)
        self.clear_costmaps = rospy.ServiceProxy(
            self.clear_costmaps_service, Empty)

        self.cmd_pub = rospy.Publisher(
            self.cmd_vel_topic, Twist, queue_size=1)
        self.tts_pub = rospy.Publisher(
            self.tts_topic, String, queue_size=5)
        self.result_pub = rospy.Publisher(
            self.result_topic, String, queue_size=5, latch=True)
        self.state_pub = rospy.Publisher(
            self.state_topic, String, queue_size=5, latch=True)
        self.ocr_control_pub = rospy.Publisher(
            self.ocr_control_topic, String, queue_size=5)

        rospy.Subscriber(self.task_result_topic, String, self.task_callback,
                         queue_size=5)
        rospy.Subscriber(self.ocr_result_topic, String, self.ocr_callback,
                         queue_size=10)
        rospy.Subscriber(self.image_topic, Image, self.image_callback,
                         queue_size=1, buff_size=4 * 1024 * 1024)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback,
                         queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseWithCovarianceStamped,
                         self.pose_callback, queue_size=1)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback,
                         queue_size=1)

        rospy.on_shutdown(self.shutdown)
        self.publish_state("WAITING_QR_DECISION")
        rospy.logwarn(
            "FACTORY_ROOM_MANAGER_READY action=%s fixed=%s adaptive=%s",
            self.move_base_action,
            [point["name"] for point in self.fixed_observation_points],
            [point["name"] for point in self.adaptive_observation_points])

    def build_adaptive_search_points(self, fixed_points):
        if not self.adaptive_search_enabled or len(fixed_points) < 2:
            return []

        center_x = sum(point["x"] for point in fixed_points) / len(fixed_points)
        center_y = sum(point["y"] for point in fixed_points) / len(fixed_points)
        min_y = min(point["y"] for point in fixed_points) - 0.45
        max_y = max(point["y"] for point in fixed_points) + 0.12

        candidates = [(center_x, center_y)]
        for index, first in enumerate(fixed_points):
            for second in fixed_points[index + 1:]:
                candidates.append((
                    0.5 * (first["x"] + second["x"]),
                    0.5 * (first["y"] + second["y"])))
        for angle in (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi):
            candidates.append((
                center_x + self.adaptive_ring_radius_m * math.cos(angle),
                clamp(center_y + self.adaptive_ring_radius_m * math.sin(angle),
                      min_y, max_y)))

        accepted = []
        occupied = [(point["x"], point["y"]) for point in fixed_points]
        for x, y in candidates:
            if any(math.hypot(x - old_x, y - old_y) < self.adaptive_min_spacing_m
                   for old_x, old_y in occupied):
                continue
            yaw = math.atan2(center_y - y, center_x - x)
            accepted.append({
                "name": "auto_%02d" % (len(accepted) + 1),
                "x": x,
                "y": y,
                "yaw": yaw,
            })
            occupied.append((x, y))
            if len(accepted) >= max(0, self.adaptive_point_limit):
                break
        return accepted

    def publish_state(self, state, **extra):
        payload = {"state": state, "stamp": time.time()}
        payload.update(extra)
        self.state_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn("FACTORY_ROOM_STATE %s", json.dumps(payload, ensure_ascii=False))

    def publish_zero(self, repeats=1):
        for _ in range(max(1, repeats)):
            self.cmd_pub.publish(Twist())
            if repeats > 1:
                rospy.sleep(0.03)

    def task_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            rospy.logwarn("factory room ignored non-JSON task result")
            return
        if payload.get("status") != "success":
            return
        item = str(payload.get("selected_item", "")).strip()
        warehouse = str(payload.get("target_warehouse", "")).strip()
        if not item or warehouse not in (
                "食品加工车间", "日用品加工车间", "电子产品生产车间"):
            rospy.logwarn("factory room ignored incomplete decision: %s", msg.data)
            return
        stamp = float(payload.get("stamp", time.time()))
        if stamp < self.launch_time - 2.0:
            rospy.logwarn("factory room ignored stale latched result stamp=%.3f", stamp)
            return
        with self.lock:
            if self.mission_running or self.task_payload is not None:
                return
            self.task_payload = payload
            self.target_item = item
            self.target_warehouse = warehouse
            self.mission_running = True
        rospy.logwarn("FACTORY_ROOM_TASK_ACCEPTED item=%s warehouse=%s",
                      item, warehouse)
        threading.Thread(target=self.mission_thread, daemon=True).start()

    def ocr_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self.lock:
            self.latest_ocr = payload
        if (payload.get("stable") and
                str(payload.get("label", "")) == self.target_warehouse):
            self.target_ocr_time = time.time()
            self.target_found.set()
            rospy.logwarn_throttle(
                1.0, "TARGET_WORKSHOP_SEEN label=%s score=%.2f bbox=%s",
                self.target_warehouse, float(payload.get("score", 0.0)),
                payload.get("bbox"))

    def image_callback(self, msg):
        image = decode_ros_image(msg)
        if image is None:
            return
        with self.lock:
            self.latest_image = image
            self.latest_image_time = time.time()

    def pose_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        with self.lock:
            self.pose = (msg.pose.pose.position.x,
                         msg.pose.pose.position.y, yaw)
            self.pose_time = time.time()

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        with self.lock:
            self.odom_pose = (msg.pose.pose.position.x,
                              msg.pose.pose.position.y)
            self.odom_yaw = yaw

    def scan_callback(self, msg):
        valid_all = []
        front = []
        front_wall = []
        for index, distance in enumerate(msg.ranges):
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            valid_all.append(distance)
            if abs(angle) <= math.radians(18.0):
                front.append(distance)
            if abs(angle) <= math.radians(12.0) and distance <= 4.0:
                front_wall.append(distance)
        with self.lock:
            self.scan_ranges = valid_all
            self.all_min = min(valid_all) if valid_all else float("inf")
            self.front_min = min(front) if front else float("inf")
            if front_wall:
                ordered = sorted(front_wall)
                # A high percentile ignores short cone returns and estimates
                # the wall carrying the sign.
                self.front_wall = ordered[int(0.72 * (len(ordered) - 1))]
            else:
                self.front_wall = float("inf")

    def snapshot_pose(self):
        with self.lock:
            return self.pose

    def snapshot_odom(self):
        with self.lock:
            return self.odom_pose, self.odom_yaw

    def snapshot_image(self):
        with self.lock:
            if self.latest_image is None:
                return None, 0.0
            return self.latest_image.copy(), self.latest_image_time

    def wait_for_inputs(self, timeout_s=20.0):
        start = time.time()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            image, image_time = self.snapshot_image()
            with self.lock:
                ready = (self.pose is not None and self.odom_yaw is not None and
                         self.scan_ranges and image is not None and
                         time.time() - image_time < 1.0)
            if ready:
                return True
            rate.sleep()
        return False

    def wait_for_move_base(self):
        self.publish_state("WAITING_MOVE_BASE")
        if self.move_client.wait_for_server(rospy.Duration(35.0)):
            return True
        self.fail("房间导航服务未启动")
        return False

    @staticmethod
    def make_goal(x, y, yaw):
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = x
        goal.target_pose.pose.position.y = y
        quat = quaternion_from_euler(0.0, 0.0, yaw)
        goal.target_pose.pose.orientation.x = quat[0]
        goal.target_pose.pose.orientation.y = quat[1]
        goal.target_pose.pose.orientation.z = quat[2]
        goal.target_pose.pose.orientation.w = quat[3]
        return goal

    def clear_navigation_costmaps(self):
        try:
            rospy.wait_for_service(self.clear_costmaps_service, timeout=2.0)
            self.clear_costmaps()
            rospy.logwarn("FACTORY_ROOM_COSTMAPS_CLEARED")
        except Exception as exc:
            rospy.logwarn("failed to clear room costmaps: %s", exc)

    def navigate(self, name, x, y, yaw, allow_offsets=True):
        offsets = [(0.0, 0.0)]
        if allow_offsets:
            offsets.extend([(0.20, 0.0), (-0.20, 0.0),
                            (0.0, 0.20), (0.0, -0.20)])
        attempts = max(1, self.nav_goal_retries)
        for attempt in range(attempts):
            off_x, off_y = offsets[min(attempt, len(offsets) - 1)]
            goal_x, goal_y = x + off_x, y + off_y
            self.publish_state(
                "NAVIGATING", goal=name, attempt=attempt + 1,
                x=goal_x, y=goal_y, yaw=yaw)
            rospy.logwarn(
                "ROOM_NAV_GOAL name=%s attempt=%d x=%.3f y=%.3f yaw=%.1fdeg",
                name, attempt + 1, goal_x, goal_y, math.degrees(yaw))
            self.move_client.send_goal(self.make_goal(goal_x, goal_y, yaw))
            start = time.time()
            last_progress = start
            pose = self.snapshot_pose()
            last_pose = pose[:2] if pose else None
            rate = rospy.Rate(5)
            while not rospy.is_shutdown():
                state = self.move_client.get_state()
                if state == GoalStatus.SUCCEEDED:
                    self.publish_zero(3)
                    rospy.logwarn("ROOM_NAV_REACHED name=%s", name)
                    return True
                if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                             GoalStatus.RECALLED, GoalStatus.LOST):
                    rospy.logwarn("ROOM_NAV_ACTION_END name=%s state=%d", name, state)
                    break
                now = time.time()
                pose = self.snapshot_pose()
                if pose is not None:
                    current = pose[:2]
                    if (last_pose is None or
                            math.hypot(current[0] - last_pose[0],
                                       current[1] - last_pose[1]) >= self.nav_progress_m):
                        last_pose = current
                        last_progress = now
                if now - last_progress > self.nav_stuck_timeout_s:
                    rospy.logwarn("ROOM_NAV_STUCK name=%s; cancelling and replanning", name)
                    break
                if now - start > self.nav_goal_timeout_s:
                    rospy.logwarn("ROOM_NAV_TIMEOUT name=%s", name)
                    break
                rate.sleep()
            current_state = self.move_client.get_state()
            if current_state in (
                    GoalStatus.PENDING, GoalStatus.ACTIVE,
                    GoalStatus.PREEMPTING, GoalStatus.RECALLING):
                self.move_client.cancel_goal()
            self.publish_zero(5)
            self.clear_navigation_costmaps()
            rospy.sleep(0.8)
        rospy.logerr("ROOM_NAV_FAILED name=%s after %d attempts", name, attempts)
        return False

    def ocr_control(self, command):
        self.ocr_control_pub.publish(String(data=command))
        rospy.sleep(0.15)

    def rotate_relative(self, delta_yaw):
        _, start_yaw = self.snapshot_odom()
        if start_yaw is None:
            return False
        with self.lock:
            clearance = self.all_min
        if clearance < self.spin_clearance_m:
            rospy.logwarn("ROOM_SPIN_SKIPPED clearance=%.3f < %.3f",
                          clearance, self.spin_clearance_m)
            return False
        target = norm_angle(start_yaw + delta_yaw)
        timeout = max(4.0, abs(delta_yaw) / max(self.scan_turn_speed, 0.1) * 2.1)
        start = time.time()
        rate = rospy.Rate(16)
        while not rospy.is_shutdown() and time.time() - start < timeout:
            if self.target_found.is_set():
                self.publish_zero(3)
                return True
            _, yaw = self.snapshot_odom()
            if yaw is None:
                break
            error = norm_angle(target - yaw)
            if abs(error) <= math.radians(3.0):
                self.publish_zero(3)
                return True
            command = Twist()
            command.angular.z = math.copysign(
                max(0.16, min(self.scan_turn_speed, abs(error) * 1.2)), error)
            self.cmd_pub.publish(command)
            rate.sleep()
        self.publish_zero(5)
        return False

    def scan_for_target_at_current_pose(self, point_name):
        self.publish_state("SCANNING_WORKSHOP", point=point_name,
                           target=self.target_warehouse)
        self.target_found.clear()
        self.ocr_control("reset")
        self.ocr_control("enable")
        rospy.sleep(1.5)
        if self.target_found.is_set():
            self.publish_zero(3)
            return True
        steps = max(4, int(round(360.0 / max(self.scan_step_deg, 10.0))))
        step_radians = math.radians(360.0 / float(steps))
        for step in range(steps):
            if rospy.is_shutdown():
                return False
            if self.target_found.is_set():
                self.publish_zero(3)
                return True
            if not self.rotate_relative(step_radians):
                if self.target_found.is_set():
                    return True
                rospy.logwarn("scan rotation blocked at %s step=%d", point_name, step)
                break
            rospy.sleep(self.scan_settle_s)
        return self.target_found.is_set()

    def align_to_target_sign(self, timeout_s=7.0):
        start = time.time()
        aligned_count = 0
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            with self.lock:
                payload = dict(self.latest_ocr) if self.latest_ocr else None
            if (not payload or payload.get("label") != self.target_warehouse or
                    not payload.get("bbox")):
                self.publish_zero()
                rate.sleep()
                continue
            if time.time() - float(payload.get("stamp", 0.0)) > 2.0:
                self.publish_zero()
                rate.sleep()
                continue
            bbox = payload["bbox"]
            width = max(1.0, float(payload.get("image_width", 640)))
            center = 0.5 * (float(bbox[0]) + float(bbox[2]))
            image_error = (center - width * 0.5) / (width * 0.5)
            if abs(image_error) <= 0.09:
                aligned_count += 1
                self.publish_zero()
                if aligned_count >= 3:
                    rospy.logwarn("TARGET_SIGN_ALIGNED error=%.3f", image_error)
                    return True
            else:
                aligned_count = 0
                command = Twist()
                direction = 1.0 if self.image_mirrored else -1.0
                command.angular.z = clamp(
                    direction * image_error * 0.42, -0.19, 0.19)
                if abs(command.angular.z) < 0.11:
                    command.angular.z = math.copysign(0.11, command.angular.z)
                self.cmd_pub.publish(command)
            rate.sleep()
        self.publish_zero(4)
        rospy.logwarn("target sign alignment timed out; using detected heading")
        return False

    def approach_target_wall_with_navigation(self):
        self.align_to_target_sign()
        pose = self.snapshot_pose()
        if pose is None:
            return False
        with self.lock:
            wall_range = self.front_wall
        if math.isinf(wall_range):
            rospy.logwarn("wall range unavailable; staying at observation pose")
            return True
        travel = clamp(wall_range - self.wall_standoff_m, 0.0, 0.85)
        if travel < 0.12:
            rospy.logwarn("already near target wall: estimate=%.3fm", wall_range)
            return True
        x, y, yaw = pose
        goal_x = x + travel * math.cos(yaw)
        goal_y = y + travel * math.sin(yaw)
        self.ocr_control("disable")
        ok = self.navigate("target_wall_standoff", goal_x, goal_y, yaw,
                           allow_offsets=True)
        self.ocr_control("reset")
        self.ocr_control("enable")
        rospy.sleep(1.5)
        self.align_to_target_sign(timeout_s=5.0)
        return ok

    def detect_square_once(self):
        image, image_time = self.snapshot_image()
        if image is None or time.time() - image_time > 1.0:
            return None
        return self.square_detector.detect(image)

    def acquire_square(self, timeout_s=8.0):
        self.publish_state("FINDING_WHITE_BOX")
        start = time.time()
        stable = 0
        best = None
        rate = rospy.Rate(6)
        while not rospy.is_shutdown() and time.time() - start < timeout_s:
            result = self.detect_square_once()
            if result and result["found"]:
                stable += 1
                best = result
                if stable >= 3:
                    rospy.logwarn(
                        "WHITE_BOX_ACQUIRED conf=%.2f off=%.3fm width=%.3fm near=%s",
                        result["confidence"], result["lateral_error_m"],
                        result["rail_width_m"],
                        str(result["near_edge_distance_m"]))
                    return result
            else:
                stable = max(0, stable - 1)
            rate.sleep()
        return best if best and best["found"] else None

    def move_closer_for_square(self):
        pose = self.snapshot_pose()
        if pose is None:
            return False
        with self.lock:
            wall_range = self.front_wall
        if math.isinf(wall_range) or wall_range <= 0.52:
            return False
        travel = clamp(wall_range - 0.52, 0.12, 0.42)
        x, y, yaw = pose
        return self.navigate(
            "white_box_view", x + travel * math.cos(yaw),
            y + travel * math.sin(yaw), yaw, allow_offsets=False)

    @staticmethod
    def odom_distance(start_pose, current_pose):
        if start_pose is None or current_pose is None:
            return 0.0
        return math.hypot(current_pose[0] - start_pose[0],
                          current_pose[1] - start_pose[1])

    def park_inside_square(self):
        self.move_client.cancel_all_goals()
        self.publish_zero(6)
        self.publish_state("PARKING_IN_WHITE_BOX")
        start_time = time.time()
        last_seen_time = start_time
        last_result = None
        entry_odom = None
        movement_anchor, _ = self.snapshot_odom()
        last_progress_time = start_time
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() - start_time < self.square_approach_timeout_s:
            result = self.detect_square_once()
            if result and result["found"]:
                last_result = result
                last_seen_time = time.time()
                lateral = float(result["lateral_error_m"])
                near_edge = result["near_edge_distance_m"]
                if near_edge is not None and near_edge <= 0.11 and entry_odom is None:
                    entry_odom, _ = self.snapshot_odom()
                    rospy.logwarn("WHITE_BOX_ENTRY_EDGE_REACHED near=%.3fm", near_edge)
            elif last_result is None or time.time() - last_seen_time > 2.5:
                self.publish_zero()
                rospy.logwarn_throttle(1.0, "WHITE_BOX_LOST; stopping to reacquire")
                rate.sleep()
                continue
            else:
                lateral = float(last_result["lateral_error_m"])

            current_odom, _ = self.snapshot_odom()
            entered_distance = self.odom_distance(entry_odom, current_odom)
            total_motion = self.odom_distance(movement_anchor, current_odom)
            if total_motion >= 0.025:
                movement_anchor = current_odom
                last_progress_time = time.time()

            with self.lock:
                front = self.front_min
                wall = self.front_wall
            centered = abs(lateral) <= self.square_lateral_tolerance_m
            if (centered and wall <= self.wall_final_distance_m and
                    time.time() - start_time > 1.0):
                self.publish_zero(12)
                rospy.logwarn(
                    "WHITE_BOX_PARKED_BY_LIDAR front_min=%.3fm wall=%.3fm off=%.3fm entry=%.3fm",
                    front, wall, lateral, entered_distance)
                return True
            if entry_odom is not None and entered_distance >= self.square_entry_distance_m:
                self.publish_zero(12)
                rospy.logwarn(
                    "WHITE_BOX_PARKED_BY_ODOM front=%.3fm off=%.3fm entry=%.3fm",
                    front, lateral, entered_distance)
                return centered
            if front <= self.square_front_hard_stop_m:
                self.publish_zero(12)
                valid_wall_stop = (centered and
                                   wall <= self.wall_final_distance_m + 0.04)
                rospy.logerr(
                    "WHITE_BOX_HARD_STOP front_min=%.3fm wall=%.3fm off=%.3fm valid=%s",
                    front, wall, lateral, str(valid_wall_stop))
                return valid_wall_stop
            if time.time() - last_progress_time > 5.0:
                self.publish_zero(12)
                rospy.logerr("WHITE_BOX_PARKING_STUCK; stopped safely")
                return False

            direction = 1.0 if self.image_mirrored else -1.0
            command = Twist()
            command.angular.z = clamp(direction * lateral * 2.2, -0.24, 0.24)
            if abs(lateral) > 0.13:
                command.linear.x = 0.025
            elif front < 0.42 or entry_odom is not None:
                command.linear.x = self.square_speed_slow
            else:
                command.linear.x = self.square_speed_fast
            self.cmd_pub.publish(command)
            rospy.loginfo_throttle(
                0.8,
                "WHITE_BOX_SERVO front_min=%.3f wall=%.3f off=%.3f near=%s entry=%.3f cmd=(%.3f,%.3f)",
                front, wall, lateral,
                str(last_result.get("near_edge_distance_m") if last_result else None),
                entered_distance, command.linear.x, command.angular.z)
            rate.sleep()

        self.publish_zero(12)
        rospy.logerr("WHITE_BOX_PARKING_TIMEOUT; robot stopped")
        return False

    def find_target_workshop(self):
        for cycle in range(max(1, self.search_cycles)):
            points = (self.observation_points if cycle % 2 == 0
                      else list(reversed(self.observation_points)))
            for point in points:
                if rospy.is_shutdown():
                    return False
                self.ocr_control("disable")
                reached = self.navigate(
                    point["name"], point["x"], point["y"], point["yaw"],
                    allow_offsets=True)
                if not reached:
                    # An occupied observation point must not terminate the
                    # whole mission; continue to a different view of the room.
                    continue
                if self.scan_for_target_at_current_pose(point["name"]):
                    return True
            rospy.logwarn("WORKSHOP_SEARCH_CYCLE_COMPLETE cycle=%d/%d target=%s",
                          cycle + 1, self.search_cycles, self.target_warehouse)
            self.clear_navigation_costmaps()
        return False

    def mission_thread(self):
        mission_start = time.time()
        try:
            self.publish_state(
                "WAITING_FOR_TTS", item=self.target_item,
                warehouse=self.target_warehouse,
                wait_s=self.start_after_tts_s)
            rospy.sleep(max(0.0, self.start_after_tts_s))
            if not self.wait_for_move_base() or not self.wait_for_inputs():
                self.fail("房间任务所需的导航、激光或相机数据未就绪")
                return

            if self.navigate_start_pose:
                if not self.navigate(
                        "start", self.start_pose[0], self.start_pose[1],
                        self.start_pose[2], allow_offsets=False):
                    self.fail("未能安全到达大房间入口start点")
                    return
            else:
                rospy.logwarn("ROOM_NAV_START_SKIPPED; first goal will be d1")

            if time.time() - mission_start > self.mission_timeout_s:
                self.fail("房间任务超时，已安全停车")
                return
            if not self.find_target_workshop():
                self.fail("巡检全部观察点后仍未可靠识别到目标车间")
                return

            self.publish_zero(5)
            self.publish_state("TARGET_WORKSHOP_FOUND",
                               warehouse=self.target_warehouse)
            self.approach_target_wall_with_navigation()

            square = self.acquire_square(timeout_s=8.0)
            if square is None:
                rospy.logwarn("white box not yet visible; moving to a closer safe view")
                self.ocr_control("disable")
                self.move_closer_for_square()
                self.ocr_control("enable")
                rospy.sleep(1.0)
                square = self.acquire_square(timeout_s=8.0)
            if square is None:
                self.fail("已找到目标车间，但未能可靠识别墙根白色停车框")
                return

            if not self.park_inside_square():
                self.fail("白框停车未通过安全与位置校验，小车已停止")
                return

            self.ocr_control("disable")
            final_text = "已将{}放入{}".format(
                self.target_item, self.target_warehouse)
            self.tts_pub.publish(String(data=final_text))
            payload = {
                "status": "success",
                "selected_item": self.target_item,
                "target_warehouse": self.target_warehouse,
                "broadcast_text": final_text,
                "parked_inside_white_box": True,
                "stamp": time.time(),
            }
            self.result_pub.publish(String(
                data=json.dumps(payload, ensure_ascii=False)))
            self.publish_state("DONE", **payload)
            rospy.logwarn("FACTORY_DELIVERY_COMPLETE %s", final_text)
            self.publish_zero(20)
        except Exception as exc:
            rospy.logerr("factory room mission exception: %s", exc)
            self.fail("房间任务发生异常，小车已安全停车：{}".format(exc))
        finally:
            self.publish_zero(10)
            with self.lock:
                self.mission_running = False

    def fail(self, reason):
        self.move_client.cancel_all_goals()
        self.ocr_control("disable")
        self.publish_zero(15)
        payload = {
            "status": "error",
            "reason": reason,
            "selected_item": self.target_item,
            "target_warehouse": self.target_warehouse,
            "stamp": time.time(),
        }
        self.result_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))
        self.publish_state("ERROR", **payload)
        rospy.logerr("FACTORY_DELIVERY_FAILED %s", reason)

    def shutdown(self):
        try:
            self.move_client.cancel_all_goals()
        except Exception:
            pass
        self.publish_zero(10)

    def run(self):
        rospy.spin()


if __name__ == "__main__":
    try:
        FactoryRoomDeliveryManager().run()
    except rospy.ROSInterruptException:
        pass
