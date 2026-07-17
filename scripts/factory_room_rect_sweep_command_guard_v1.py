#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Guard all room turning commands with an oriented rectangular footprint."""

import math
import threading

import rospy
import tf
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


INF = float("inf")


class FactoryRoomRectSweepCommandGuard(object):
    def __init__(self):
        self.lock = threading.RLock()
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        self.request_topic = rospy.get_param(
            "~request_topic", "/factory_room/cmd_vel_request")
        self.output_topic = rospy.get_param("~output_topic", "/cmd_vel_raw")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.pose_topic = rospy.get_param("~pose_topic", "/amcl_pose")
        self.parking_topic = rospy.get_param(
            "~parking_mode_topic", "/factory/parking_close_mode")
        self.trigger_y = float(rospy.get_param("~indoor_trigger_y", -1.75))
        self.trigger_less_than = bool(rospy.get_param(
            "~indoor_trigger_less_than", True))

        self.half_length = float(rospy.get_param(
            "~rect_robot_half_length_m", 0.171))
        self.half_width = float(rospy.get_param(
            "~rect_robot_half_width_m", 0.128))
        self.margin = float(rospy.get_param(
            "~rect_footprint_margin_m", 0.015))
        self.horizon = float(rospy.get_param(
            "~rect_guard_horizon_s", 0.80))
        self.dt = float(rospy.get_param("~rect_guard_sim_dt_s", 0.05))
        self.scan_range = float(rospy.get_param(
            "~rect_guard_scan_range_m", 0.90))
        self.scan_stride = max(1, int(rospy.get_param(
            "~rect_guard_scan_stride", 2)))
        self.scan_timeout = float(rospy.get_param(
            "~rect_guard_scan_timeout_s", 0.45))
        self.command_timeout = float(rospy.get_param(
            "~rect_guard_command_timeout_s", 0.35))
        self.minimum_turn = float(rospy.get_param(
            "~rect_guard_min_turn_rate_rps", 0.08))
        self.worsen_allowance = float(rospy.get_param(
            "~rect_guard_worsen_allowance_m", 0.003))
        self.turn_scales = self._scale_list(rospy.get_param(
            "~rect_guard_turn_scales", [1.0, 0.75, 0.50, 0.25, 0.0]))
        self.speed_scales = self._scale_list(rospy.get_param(
            "~rect_guard_speed_scales", [1.0, 0.75, 0.50, 0.25, 0.0]))

        self.tf_listener = tf.TransformListener()
        self.scan_points = []
        self.scan_stamp = rospy.Time(0)
        self.last_request = rospy.Time(0)
        self.last_output_nonzero = False
        self.room_active = False
        self.parking_mode = False
        self.was_limiting = False

        self.output_pub = rospy.Publisher(
            self.output_topic, Twist, queue_size=1)
        rospy.Subscriber(self.scan_topic, LaserScan, self.scan_callback,
                         queue_size=1)
        rospy.Subscriber(self.pose_topic, PoseWithCovarianceStamped,
                         self.pose_callback, queue_size=1)
        rospy.Subscriber(self.parking_topic, Bool, self.parking_callback,
                         queue_size=1)
        rospy.Subscriber(self.request_topic, Twist, self.command_callback,
                         queue_size=1)
        rospy.Timer(rospy.Duration(0.05), self.watchdog)
        rospy.on_shutdown(self.shutdown)
        rospy.logwarn(
            "ROOM_RECT_COMMAND_GUARD_READY request=%s output=%s "
            "trigger_y=%.2f footprint=(%.3fx%.3f)m margin=%.3f",
            self.request_topic, self.output_topic, self.trigger_y,
            2.0 * self.half_length, 2.0 * self.half_width, self.margin)

    @staticmethod
    def _scale_list(raw):
        values = []
        if isinstance(raw, (list, tuple)):
            for value in raw:
                try:
                    values.append(max(0.0, min(1.0, float(value))))
                except (TypeError, ValueError):
                    pass
        values.append(0.0)
        return sorted(set(values), reverse=True)

    @staticmethod
    def _nonzero(command):
        return (abs(command.linear.x) > 1.0e-5 or
                abs(command.linear.y) > 1.0e-5 or
                abs(command.angular.z) > 1.0e-5)

    def pose_callback(self, msg):
        if self.room_active:
            return
        y = msg.pose.pose.position.y
        inside = y <= self.trigger_y if self.trigger_less_than else y >= self.trigger_y
        if inside:
            self.room_active = True
            rospy.logwarn("ROOM_RECT_COMMAND_GUARD_ACTIVE y=%.3f", y)

    def parking_callback(self, msg):
        enabled = bool(msg.data)
        if enabled != self.parking_mode:
            rospy.logwarn("ROOM_RECT_COMMAND_GUARD parking_bypass=%s",
                          str(enabled).lower())
        self.parking_mode = enabled

    def scan_callback(self, msg):
        points = self._scan_points_in_base(msg)
        with self.lock:
            self.scan_points = points
            self.scan_stamp = rospy.Time.now()

    def _scan_points_in_base(self, msg):
        frame_id = msg.header.frame_id or self.base_frame
        tx = 0.0
        ty = 0.0
        yaw = 0.0
        if frame_id != self.base_frame:
            try:
                trans, rotation = self.tf_listener.lookupTransform(
                    self.base_frame, frame_id, rospy.Time(0))
                tx, ty = trans[0], trans[1]
                yaw = tf.transformations.euler_from_quaternion(rotation)[2]
            except Exception:
                return []
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        points = []
        for index in range(0, len(msg.ranges), self.scan_stride):
            distance = msg.ranges[index]
            if math.isnan(distance) or math.isinf(distance):
                continue
            if distance < msg.range_min or distance > msg.range_max:
                continue
            if distance > self.scan_range:
                continue
            angle = msg.angle_min + index * msg.angle_increment
            lx = distance * math.cos(angle)
            ly = distance * math.sin(angle)
            points.append((tx + cos_yaw * lx - sin_yaw * ly,
                           ty + sin_yaw * lx + cos_yaw * ly))
        return points

    def _footprint_gap(self, x, y, theta, points):
        if not points:
            return INF
        half_length = self.half_length + self.margin
        half_width = self.half_width + self.margin
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)
        best = INF
        for obstacle_x, obstacle_y in points:
            dx = obstacle_x - x
            dy = obstacle_y - y
            local_x = cos_theta * dx + sin_theta * dy
            local_y = -sin_theta * dx + cos_theta * dy
            outside_x = max(abs(local_x) - half_length, 0.0)
            outside_y = max(abs(local_y) - half_width, 0.0)
            if outside_x > 0.0 or outside_y > 0.0:
                gap = math.hypot(outside_x, outside_y)
            else:
                gap = -min(half_length - abs(local_x),
                           half_width - abs(local_y))
            best = min(best, gap)
        return best

    def _trajectory_result(self, vx, vy, wz, points):
        start_gap = self._footprint_gap(0.0, 0.0, 0.0, points)
        minimum_gap = start_gap
        x = 0.0
        y = 0.0
        theta = 0.0
        steps = max(3, int(math.ceil(self.horizon / self.dt)))
        for _ in range(steps):
            x += (math.cos(theta) * vx - math.sin(theta) * vy) * self.dt
            y += (math.sin(theta) * vx + math.cos(theta) * vy) * self.dt
            theta += wz * self.dt
            gap = self._footprint_gap(x, y, theta, points)
            minimum_gap = min(minimum_gap, gap)
            if start_gap > 0.0 and gap <= 0.0:
                return False, minimum_gap
            if start_gap <= 0.0 and gap < start_gap - self.worsen_allowance:
                return False, minimum_gap
        return True, minimum_gap

    def _closest_safe_command(self, command, points):
        vx = command.linear.x
        vy = command.linear.y
        wz = command.angular.z
        safe, minimum_gap = self._trajectory_result(vx, vy, wz, points)
        if safe:
            return command, minimum_gap, False
        best = None
        linear_base = max(math.hypot(vx, vy), 0.05)
        turn_base = max(abs(wz), self.minimum_turn)
        for turn_scale in self.turn_scales:
            for speed_scale in self.speed_scales:
                candidate_vx = vx * speed_scale
                candidate_vy = vy * speed_scale
                candidate_wz = wz * turn_scale
                safe, gap = self._trajectory_result(
                    candidate_vx, candidate_vy, candidate_wz, points)
                if not safe:
                    continue
                speed_error = math.hypot(
                    vx - candidate_vx, vy - candidate_vy) / linear_base
                turn_error = abs(wz - candidate_wz) / turn_base
                score = 0.62 * speed_error ** 2 + 0.38 * turn_error ** 2
                candidate = (score, -gap, candidate_vx,
                             candidate_vy, candidate_wz, gap)
                if best is None or candidate < best:
                    best = candidate
        output = Twist()
        if best is None:
            return output, minimum_gap, True
        output.linear.x = best[2]
        output.linear.y = best[3]
        output.angular.z = best[4]
        return output, best[5], True

    def command_callback(self, command):
        now = rospy.Time.now()
        with self.lock:
            self.last_request = now
            points = list(self.scan_points)
            scan_fresh = ((now - self.scan_stamp).to_sec() <= self.scan_timeout)

        if (self.parking_mode or not self.room_active or
                abs(command.angular.z) < self.minimum_turn or
                not self._nonzero(command)):
            output = command
            limited = False
            minimum_gap = INF
        elif not scan_fresh:
            output = Twist()
            limited = True
            minimum_gap = -INF
            rospy.logwarn_throttle(
                1.0, "ROOM_RECT_COMMAND_STOP scan_unavailable")
        else:
            output, minimum_gap, limited = self._closest_safe_command(
                command, points)

        if limited:
            rospy.logwarn_throttle(
                0.35,
                "ROOM_RECT_COMMAND_LIMIT in=(%.3f,%.3f,%.3f) "
                "out=(%.3f,%.3f,%.3f) gap=%.3f points=%d",
                command.linear.x, command.linear.y, command.angular.z,
                output.linear.x, output.linear.y, output.angular.z,
                minimum_gap, len(points))
        elif self.was_limiting:
            rospy.logwarn("ROOM_RECT_COMMAND_CLEAR")
        self.was_limiting = limited
        self.last_output_nonzero = self._nonzero(output)
        self.output_pub.publish(output)

    def watchdog(self, _event):
        if not self.last_output_nonzero:
            return
        if ((rospy.Time.now() - self.last_request).to_sec() <=
                self.command_timeout):
            return
        self.last_output_nonzero = False
        self.output_pub.publish(Twist())
        rospy.logwarn_throttle(1.0, "ROOM_RECT_COMMAND_WATCHDOG_STOP")

    def shutdown(self):
        for _ in range(5):
            self.output_pub.publish(Twist())
            rospy.sleep(0.02)


if __name__ == "__main__":
    rospy.init_node("factory_room_rect_sweep_command_guard_v1")
    FactoryRoomRectSweepCommandGuard()
    rospy.spin()
