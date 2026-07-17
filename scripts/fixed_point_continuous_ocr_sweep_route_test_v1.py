#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Front-first route test with continuous OCR sweeps at d1, d2 and d3."""

import json
import math

import rospy
from std_msgs.msg import String

from continuous_yaw_sweep_core_v1 import (
    braking_limited_speed,
    directed_yaw_increment,
    slew_rate,
    wrap_angle,
)
from front_first_room_rect_sweep_route_test_v1 import (
    FrontFirstRoomRectSweepRouteTest,
)


class FixedPointContinuousOcrSweepRouteTest(
        FrontFirstRoomRectSweepRouteTest):
    def __init__(self):
        # The inherited navigator creates its control timer during __init__.
        self.continuous_sweep_ready = False
        self.scan_active = False
        self.scan_phase = "idle"
        self.scan_specs = {}
        self.scanned_route_indexes = set()
        self.scan_candidate_attempts = {}
        self.scan_route_started = None
        self.scan_command_wz = 0.0
        self.scan_last_tick = None
        self.scan_last_yaw = None
        self.scan_progress = 0.0
        self.scan_stable = 0
        self.scan_last_motion = None
        self.scan_latest_ocr_stamp = 0.0
        super(FixedPointContinuousOcrSweepRouteTest, self).__init__()

        self.scan_specs = self._load_scan_specs(rospy.get_param(
            "~continuous_scan_specs", []))
        self.scan_max_speed = float(rospy.get_param(
            "~continuous_scan_max_speed_rps", 0.14))
        self.scan_min_speed = float(rospy.get_param(
            "~continuous_scan_min_speed_rps", 0.045))
        self.scan_acceleration = float(rospy.get_param(
            "~continuous_scan_acceleration_rps2", 0.28))
        self.scan_start_tolerance = math.radians(float(rospy.get_param(
            "~continuous_scan_start_tolerance_deg", 2.0)))
        self.scan_end_tolerance = math.radians(float(rospy.get_param(
            "~continuous_scan_end_tolerance_deg", 1.5)))
        self.scan_stable_frames = max(2, int(rospy.get_param(
            "~continuous_scan_stable_frames", 3)))
        self.scan_stall_timeout = float(rospy.get_param(
            "~continuous_scan_stall_timeout_s", 3.2))
        self.scan_point_timeout = float(rospy.get_param(
            "~continuous_scan_point_timeout_s", 42.0))
        self.scan_replan_failures = max(1, int(rospy.get_param(
            "~continuous_scan_replan_failures", 2)))
        self.scan_spin_min_gap = float(rospy.get_param(
            "~continuous_scan_spin_min_gap_m", 0.015))
        self.scan_static_margin = float(rospy.get_param(
            "~continuous_scan_static_margin_m", 0.020))
        self.scan_dynamic_margin = float(rospy.get_param(
            "~continuous_scan_dynamic_margin_m", 0.015))
        self.scan_candidate_offsets = self._load_offsets(rospy.get_param(
            "~continuous_scan_candidate_offsets_m", []))
        self.ocr_control_topic = rospy.get_param(
            "~ocr_control_topic", "/factory_room/ocr_control")
        self.ocr_result_topic = rospy.get_param(
            "~ocr_result_topic", "/factory_room/ocr_result")
        self.ocr_control_pub = rospy.Publisher(
            self.ocr_control_topic, String, queue_size=10, latch=True)
        rospy.Subscriber(
            self.ocr_result_topic, String, self.ocr_callback, queue_size=20)

        self.continuous_sweep_ready = True
        self.scan_route_started = rospy.Time.now()
        rospy.logwarn(
            "FIXED_CONTINUOUS_SCAN_READY points=%s speed=%.3f accel=%.3f "
            "cone_radius=%.2fm ocr_continuous=true parking=false",
            sorted(self.scan_specs), self.scan_max_speed,
            self.scan_acceleration, self.dynamic_exclusion_radius)

    @staticmethod
    def _load_scan_specs(raw_specs):
        specs = {}
        if not isinstance(raw_specs, list):
            return specs
        for raw in raw_specs:
            if not isinstance(raw, dict) or "name" not in raw:
                continue
            direction_name = str(raw.get("direction", "ccw")).lower()
            direction = 1 if direction_name in ("ccw", "+", "1") else -1
            specs[str(raw["name"])] = {
                "start": math.radians(float(raw["start_deg"])),
                "travel": math.radians(abs(float(raw["travel_deg"]))),
                "direction": direction,
                "direction_name": "ccw" if direction > 0 else "cw",
            }
        return specs

    @staticmethod
    def _load_offsets(raw_offsets):
        offsets = [(0.0, 0.0)]
        if isinstance(raw_offsets, list):
            for raw in raw_offsets:
                if isinstance(raw, (list, tuple)) and len(raw) == 2:
                    offsets.append((float(raw[0]), float(raw[1])))
        return sorted(set(offsets), key=lambda item: (
            math.hypot(item[0], item[1]), abs(item[0]), abs(item[1])))

    def _set_ocr(self, enabled):
        if enabled:
            self.ocr_control_pub.publish(String(data="reset"))
            self.ocr_control_pub.publish(String(data="enable"))
        else:
            self.ocr_control_pub.publish(String(data="disable"))

    def ocr_callback(self, msg):
        if not self.scan_active:
            return
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if not payload.get("stable"):
            return
        stamp = float(payload.get("stamp", 0.0))
        if stamp <= self.scan_latest_ocr_stamp:
            return
        self.scan_latest_ocr_stamp = stamp
        label = str(payload.get("label", "")).strip()
        if not label:
            return
        point = self.current_route_point()["name"]
        yaw_deg = (float("nan") if self.pose is None else
                   math.degrees(self.pose[2]))
        rospy.logwarn(
            "CONTINUOUS_SCAN_OCR point=%s phase=%s yaw=%.1fdeg "
            "label=%s bbox=%s",
            point, self.scan_phase, yaw_deg, label,
            payload.get("bbox", []))

    def _scan_point_name(self):
        if not self.route_ready:
            return ""
        return self.current_route_point()["name"]

    def _is_scan_point(self):
        return self._scan_point_name() in self.scan_specs

    def _static_spin_safe(self, x, y):
        corner_radius = math.hypot(
            self.robot_half_length + self.footprint_margin,
            self.robot_half_width + self.footprint_margin)
        return self._static_clearance(x, y) >= (
            corner_radius + self.scan_static_margin)

    def _dynamic_spin_safe(self, x, y):
        minimum = self.dynamic_exclusion_radius + self.scan_dynamic_margin
        return all(math.hypot(track.x - x, track.y - y) >= minimum
                   for track in self._confirmed_tracks())

    def _live_spin_gap(self):
        if not self.rect_scan_points:
            return -float("inf")
        corner_radius = math.hypot(
            self.robot_half_length + self.footprint_margin,
            self.robot_half_width + self.footprint_margin)
        return min(math.hypot(x, y) - corner_radius
                   for x, y in self.rect_scan_points)

    def _install_planned_candidate(self, point, requested, result, reason):
        active = result["active_goal_world"]
        point["x"], point["y"] = requested
        self.goal_x, self.goal_y = requested
        self.active_goal = active
        self.path_world = result["path_world"]
        self.path_index = 0
        self.last_progress_time = rospy.Time.now()
        self.last_progress_pose = (self.pose[0], self.pose[1])
        self.last_goal_dist = self.distance_to_active_goal()
        self.replan_fail_count = 0
        self.scan_route_started = rospy.Time.now()
        self.publish_path()
        rospy.logwarn(
            "CONTINUOUS_SCAN_ALTERNATE_SELECTED point=%s requested=(%.3f,%.3f) "
            "active=(%.3f,%.3f) offset=%.3fm reason=%s",
            point["name"], requested[0], requested[1], active[0], active[1],
            math.hypot(requested[0] - point["nominal_x"],
                       requested[1] - point["nominal_y"]), reason)

    def _select_alternative_scan_point(self, reason):
        if self.pose is None:
            return False
        point = self.current_route_point()
        name = point["name"]
        nominal_x = float(point.get("nominal_x", point["x"]))
        nominal_y = float(point.get("nominal_y", point["y"]))
        point["nominal_x"] = nominal_x
        point["nominal_y"] = nominal_y
        attempts = self.scan_candidate_attempts.setdefault(name, set())
        attempts.add((round(point["x"], 3), round(point["y"], 3)))
        self.publish_zero("SCAN_POINT_ALTERNATE_SEARCH")
        self._set_ocr(False)
        self.scan_active = False

        start = (self.pose[0], self.pose[1])
        for offset_x, offset_y in self.scan_candidate_offsets:
            requested = (nominal_x + offset_x, nominal_y + offset_y)
            key = (round(requested[0], 3), round(requested[1], 3))
            if key in attempts:
                continue
            attempts.add(key)
            if (not self._static_spin_safe(*requested) or
                    not self._dynamic_spin_safe(*requested)):
                continue
            result = self.planner.plan(start, requested)
            if not result.get("ok"):
                continue
            active = result["active_goal_world"]
            if (not self._static_spin_safe(*active) or
                    not self._dynamic_spin_safe(*active)):
                continue
            self._install_planned_candidate(
                point, requested, result, reason)
            return True
        rospy.logerr(
            "CONTINUOUS_SCAN_NO_SAFE_CANDIDATE point=%s nominal=(%.3f,%.3f) "
            "reason=%s candidates=%d",
            name, nominal_x, nominal_y, reason,
            len(self.scan_candidate_offsets))
        self.finished = True
        self.publish_zero("SCAN_POINT_NO_SAFE_CANDIDATE")
        return False

    def _begin_scan(self):
        point = self.current_route_point()
        live_gap = self._live_spin_gap()
        if (not self._static_spin_safe(self.pose[0], self.pose[1]) or
                not self._dynamic_spin_safe(self.pose[0], self.pose[1]) or
                live_gap < self.scan_spin_min_gap):
            rospy.logwarn(
                "CONTINUOUS_SCAN_POSE_REJECTED point=%s pose=(%.3f,%.3f) "
                "live_spin_gap=%.3f; searching nearest safe candidate",
                point["name"], self.pose[0], self.pose[1], live_gap)
            return self._select_alternative_scan_point(
                "arrival pose cannot complete rectangular sweep")

        self.path_world = []
        self.path_index = 0
        self.scan_active = True
        self.scan_phase = "align"
        self.scan_command_wz = 0.0
        self.scan_last_tick = rospy.Time.now()
        self.scan_last_yaw = self.pose[2]
        self.scan_progress = 0.0
        self.scan_stable = 0
        self.scan_last_motion = rospy.Time.now()
        self.scan_latest_ocr_stamp = 0.0
        self._set_ocr(True)
        spec = self.scan_specs[point["name"]]
        endpoint = spec["start"] + spec["direction"] * spec["travel"]
        rospy.logwarn(
            "CONTINUOUS_SCAN_START point=%s pose=(%.3f,%.3f) "
            "start=%.1fdeg end_unwrapped=%.1fdeg direction=%s travel=%.1fdeg "
            "live_gap=%.3f ocr=enabled",
            point["name"], self.pose[0], self.pose[1],
            math.degrees(spec["start"]), math.degrees(endpoint),
            spec["direction_name"], math.degrees(spec["travel"]), live_gap)
        return True

    def _finish_scan(self):
        name = self.current_route_point()["name"]
        self.publish_zero("CONTINUOUS_SCAN_COMPLETE")
        self._set_ocr(False)
        self.scanned_route_indexes.add(self.route_index)
        self.scan_active = False
        self.scan_phase = "idle"
        self.scan_command_wz = 0.0
        rospy.logwarn(
            "CONTINUOUS_SCAN_COMPLETE point=%s progress=%.1fdeg ocr=disabled",
            name, math.degrees(self.scan_progress))

    def _scan_stalled(self, now, yaw_delta):
        if abs(yaw_delta) >= math.radians(0.20):
            self.scan_last_motion = now
            return False
        return ((now - self.scan_last_motion).to_sec() >=
                self.scan_stall_timeout and
                abs(self.scan_command_wz) >= 0.07)

    def _scan_control_loop(self):
        if (not self.pose_fresh() or not self.scan_fresh() or
                not self._guard_scan_fresh()):
            self.publish_zero("CONTINUOUS_SCAN_WAIT_SENSOR")
            return
        now = rospy.Time.now()
        dt = max(0.02, min(0.16, (now - self.scan_last_tick).to_sec()))
        self.scan_last_tick = now
        current_yaw = self.pose[2]
        yaw_delta = wrap_angle(current_yaw - self.scan_last_yaw)
        self.scan_last_yaw = current_yaw
        if self._scan_stalled(now, yaw_delta):
            rospy.logwarn(
                "CONTINUOUS_SCAN_STALLED point=%s phase=%s; relocating",
                self._scan_point_name(), self.scan_phase)
            self._select_alternative_scan_point(
                "rectangular sweep made no yaw progress")
            return

        spec = self.scan_specs[self._scan_point_name()]
        if self.scan_phase == "align":
            error = wrap_angle(spec["start"] - current_yaw)
            remaining = abs(error)
            desired_magnitude = braking_limited_speed(
                remaining, self.scan_max_speed, self.scan_acceleration,
                self.scan_min_speed, self.scan_start_tolerance)
            desired = math.copysign(desired_magnitude, error)
            self.scan_command_wz = slew_rate(
                self.scan_command_wz, desired,
                self.scan_acceleration, dt)
            if (remaining <= self.scan_start_tolerance and
                    abs(self.scan_command_wz) <= 0.025):
                self.scan_stable += 1
            else:
                self.scan_stable = 0
            if self.scan_stable >= self.scan_stable_frames:
                self.scan_phase = "sweep"
                self.scan_progress = 0.0
                self.scan_stable = 0
                self.scan_last_yaw = current_yaw
                self.scan_last_motion = now
                self.scan_command_wz = 0.0
                rospy.logwarn(
                    "CONTINUOUS_SCAN_ALIGNED point=%s yaw=%.1fdeg; "
                    "starting uninterrupted OCR sweep",
                    self._scan_point_name(), math.degrees(current_yaw))
                self.publish_cmd(0.0, 0.0)
                return
            rospy.logwarn_throttle(
                0.5,
                "CONTINUOUS_SCAN_ALIGN point=%s yaw=%.1fdeg error=%.1fdeg "
                "cmd_wz=%.3f ocr=enabled",
                self._scan_point_name(), math.degrees(current_yaw),
                math.degrees(error), self.scan_command_wz)
            self.publish_cmd(0.0, self.scan_command_wz)
            return

        self.scan_progress += directed_yaw_increment(
            current_yaw - yaw_delta, current_yaw, spec["direction"])
        remaining = max(0.0, spec["travel"] - self.scan_progress)
        desired_magnitude = braking_limited_speed(
            remaining, self.scan_max_speed, self.scan_acceleration,
            self.scan_min_speed, self.scan_end_tolerance)
        desired = spec["direction"] * desired_magnitude
        self.scan_command_wz = slew_rate(
            self.scan_command_wz, desired, self.scan_acceleration, dt)
        if (remaining <= self.scan_end_tolerance and
                abs(self.scan_command_wz) <= 0.025):
            self.scan_stable += 1
        else:
            self.scan_stable = 0
        rospy.logwarn_throttle(
            0.5,
            "CONTINUOUS_SCAN_SWEEP point=%s progress=%.1f/%.1fdeg "
            "remaining=%.1fdeg cmd_wz=%.3f ocr=enabled",
            self._scan_point_name(), math.degrees(self.scan_progress),
            math.degrees(spec["travel"]), math.degrees(remaining),
            self.scan_command_wz)
        self.publish_cmd(0.0, self.scan_command_wz)
        if self.scan_stable >= self.scan_stable_frames:
            self._finish_scan()

    def control_loop(self, event):
        if not self.continuous_sweep_ready:
            super(FixedPointContinuousOcrSweepRouteTest,
                  self).control_loop(event)
            return
        if self.scan_active:
            self._scan_control_loop()
            return
        if self._is_scan_point() and self.route_index not in self.scanned_route_indexes:
            elapsed = ((rospy.Time.now() - self.scan_route_started).to_sec()
                       if self.scan_route_started is not None else 0.0)
            plan_failed = (
                not self.path_world and
                self.replan_fail_count >= self.scan_replan_failures)
            if plan_failed or elapsed >= self.scan_point_timeout:
                reason = ("planner repeatedly rejected scan point"
                          if plan_failed else "scan point reach timeout")
                self._select_alternative_scan_point(reason)
                return
        super(FixedPointContinuousOcrSweepRouteTest, self).control_loop(event)

    def check_goal(self):
        if (not self.continuous_sweep_ready or not self._is_scan_point() or
                self.route_index in self.scanned_route_indexes or
                self.distance_to_active_goal() >
                self.current_route_point()["tolerance"]):
            previous_index = self.route_index
            result = super(FixedPointContinuousOcrSweepRouteTest,
                           self).check_goal()
            if self.route_index != previous_index:
                self.scan_route_started = rospy.Time.now()
            return result
        self.publish_zero("SCAN_POINT_REACHED")
        return self._begin_scan()

    def shutdown(self):
        try:
            self._set_ocr(False)
        except Exception:
            pass
        super(FixedPointContinuousOcrSweepRouteTest, self).shutdown()


if __name__ == "__main__":
    FixedPointContinuousOcrSweepRouteTest().spin()

