#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Continuous search that laterally acquires any plausible workshop sign."""

import json
import math

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from factory_room_continuous_handoff_core_v1 import canonical_workshop
from factory_room_continuous_scan_search_v3 import (
    CandidateGatedContinuousScanSearchV3,
)
from factory_room_continuous_scan_search_v4 import (
    EdgeRecenteringContinuousScanSearchV4,
)
from factory_room_suspect_lateral_core_v11 import (
    lateral_acquisition_exit,
    near_scan_micro_command,
    orthogonal_alignment_command,
    orthogonal_handoff_ready,
    physical_view_ignore_decision,
    suspect_lateral_command,
)
from continuous_yaw_sweep_core_v1 import directed_yaw_increment
from factory_room_wall_handoff_core_v6 import (
    estimated_sign_heading,
    fit_oriented_front_wall,
    nearest_orthogonal_heading,
    norm_angle,
)
from fixed_point_continuous_ocr_sweep_route_test_v2 import (
    FixedPointContinuousOcrSweepRouteTestV2,
)


class SuspectLateralContinuousScanSearchV5(
        EdgeRecenteringContinuousScanSearchV4):
    """Square up to a suspected wall before classification and translation."""

    ACQUIRE_IDLE = "idle"
    SUSPECT_CONFIRM = "suspect_confirm"
    WALL_ALIGN = "wall_align"
    ORTHOGONAL_VERIFY = "orthogonal_verify"
    LATERAL_ALIGN = "lateral_align"

    def __init__(self):
        # The inherited navigation timer starts before construction returns.
        self.suspect_lateral_active = False
        self.suspect_lateral_label = ""
        self.suspect_lateral_bbox = None
        self.suspect_lateral_image_width = 0.0
        self.suspect_lateral_command = 0.0
        self.suspect_lateral_last_seen = None
        self.suspect_lateral_start_pose = None
        self.suspect_lateral_best_displacement = 0.0
        self.suspect_lateral_last_progress = None
        self.suspect_lateral_best_error = float("inf")
        self.suspect_lateral_odom_xy = None
        self.suspect_lateral_cooldowns = {}
        self.suspect_lateral_center_tolerance = 70.0
        self.suspect_lateral_command_sign = -1.0
        self.suspect_lateral_gain = 0.10
        self.suspect_lateral_min_speed = 0.025
        self.suspect_lateral_max_speed = 0.055
        self.suspect_lateral_max_shift = 0.18
        self.suspect_lateral_fresh_s = 0.70
        self.suspect_lateral_progress_m = 0.015
        self.suspect_lateral_stuck_s = 1.20
        self.suspect_lateral_cooldown_s = 1.20
        self.suspect_lateral_pixel_progress = 6.0
        self.non_target_view_label = ""
        self.non_target_view_route_index = -1
        self.non_target_view_anchor_yaw = None
        self.non_target_view_min_until = None
        self.non_target_view_max_until = None
        self.non_target_view_ignore_min_s = 2.8
        self.non_target_view_ignore_max_s = 8.0
        self.non_target_view_release_yaw = math.radians(22.0)
        self.acquire_phase = self.ACQUIRE_IDLE
        self.acquire_candidate_label = ""
        self.acquire_candidate_count = 0
        self.acquire_candidate_last_seen = None
        self.acquire_candidate_bbox = None
        self.acquire_candidate_width = 0.0
        self.acquire_started = None
        self.acquire_settle_until = None
        self.acquire_verify_deadline = None
        self.acquire_fallback_map_yaw = None
        self.acquire_wall_stable = 0
        self.acquire_best_wall_error = float("inf")
        self.acquire_last_wall_progress = None
        self.acquire_last_wall_error = None
        self.acquire_scan_anchor = None
        self.acquire_post_lateral = False
        self.acquire_scan_phase_before = "idle"
        self.handoff_gate_ready = False
        self.acquire_confirm_frames = 2
        self.acquire_confirm_fresh_s = 0.65
        self.acquire_wall_tolerance = math.radians(3.0)
        self.acquire_handoff_wall_tolerance = math.radians(4.0)
        self.acquire_wall_stable_frames = 3
        self.acquire_wall_timeout_s = 5.5
        self.acquire_wall_stuck_s = 1.8
        self.acquire_wall_kp = 1.35
        self.acquire_wall_min_wz = 0.10
        self.acquire_wall_max_wz = 0.28
        self.acquire_lateral_heading_max_wz = 0.12
        self.acquire_verify_settle_s = 0.30
        self.acquire_verify_timeout_s = 2.8
        self.acquire_camera_hfov = math.radians(70.0)
        self.acquire_laser_yaw = -0.07
        self.acquire_wall_sector = math.radians(72.0)
        self.acquire_wall_min_range = 0.20
        self.acquire_wall_max_range = 1.65
        self.acquire_wall_inlier = 0.035
        self.acquire_wall_min_inliers = 12
        self.acquire_wall_min_span = 0.22
        self.acquire_wall_max_heading = math.radians(55.0)
        self.near_micro_route_index = -1
        self.near_micro_started = None
        self.near_micro_last_progress = None
        self.near_micro_best_distance = float("inf")
        self.near_micro_best_yaw_error = float("inf")
        self.near_micro_goal_override = None
        self.near_micro_goal_route_index = -1
        self.near_micro_cooldown_until = None
        self.near_micro_trigger_distance = 0.34
        self.near_micro_timeout_s = 5.0
        self.near_micro_stuck_s = 2.0
        self.near_micro_progress_m = 0.010
        self.near_micro_yaw_progress = math.radians(3.0)
        self.near_micro_forward_gain = 0.80
        self.near_micro_lateral_gain = 0.80
        self.near_micro_yaw_gain = 0.90
        self.near_micro_max_forward = 0.10
        self.near_micro_max_lateral = 0.045
        self.near_micro_max_yaw = 0.20
        self.near_micro_yaw_slow = math.radians(20.0)
        super(SuspectLateralContinuousScanSearchV5, self).__init__()

        self.suspect_lateral_center_tolerance = float(rospy.get_param(
            "~suspect_lateral_center_tolerance_px", 70.0))
        self.suspect_lateral_command_sign = float(rospy.get_param(
            "~suspect_lateral_command_sign", -1.0))
        self.suspect_lateral_gain = float(rospy.get_param(
            "~suspect_lateral_gain", 0.10))
        self.suspect_lateral_min_speed = float(rospy.get_param(
            "~suspect_lateral_min_mps", 0.025))
        self.suspect_lateral_max_speed = float(rospy.get_param(
            "~suspect_lateral_max_mps", 0.055))
        self.suspect_lateral_max_shift = float(rospy.get_param(
            "~suspect_lateral_max_shift_m", 0.18))
        self.suspect_lateral_fresh_s = float(rospy.get_param(
            "~suspect_lateral_fresh_s", 0.70))
        self.suspect_lateral_progress_m = float(rospy.get_param(
            "~suspect_lateral_progress_m", 0.015))
        self.suspect_lateral_stuck_s = float(rospy.get_param(
            "~suspect_lateral_stuck_s", 1.20))
        self.suspect_lateral_cooldown_s = float(rospy.get_param(
            "~suspect_lateral_cooldown_s", 1.20))
        self.suspect_lateral_pixel_progress = float(rospy.get_param(
            "~suspect_lateral_pixel_progress_px", 6.0))
        self.non_target_view_ignore_min_s = float(rospy.get_param(
            "~non_target_view_ignore_min_s", 2.8))
        self.non_target_view_ignore_max_s = float(rospy.get_param(
            "~non_target_view_ignore_max_s", 8.0))
        self.non_target_view_release_yaw = math.radians(float(rospy.get_param(
            "~non_target_view_release_yaw_deg", 22.0)))
        self.acquire_confirm_frames = max(2, int(rospy.get_param(
            "~suspect_confirm_frames", 2)))
        self.acquire_confirm_fresh_s = float(rospy.get_param(
            "~suspect_confirm_fresh_s", 0.65))
        self.acquire_wall_tolerance = math.radians(float(rospy.get_param(
            "~suspect_wall_tolerance_deg", 3.0)))
        self.acquire_handoff_wall_tolerance = math.radians(float(
            rospy.get_param("~suspect_handoff_wall_tolerance_deg", 4.0)))
        self.acquire_wall_stable_frames = max(2, int(rospy.get_param(
            "~suspect_wall_stable_frames", 3)))
        self.acquire_wall_timeout_s = float(rospy.get_param(
            "~suspect_wall_timeout_s", 5.5))
        self.acquire_wall_stuck_s = float(rospy.get_param(
            "~suspect_wall_stuck_s", 1.8))
        self.acquire_wall_kp = float(rospy.get_param(
            "~suspect_wall_kp", 1.35))
        self.acquire_wall_min_wz = float(rospy.get_param(
            "~suspect_wall_min_rps", 0.10))
        self.acquire_wall_max_wz = float(rospy.get_param(
            "~suspect_wall_max_rps", 0.28))
        self.acquire_lateral_heading_max_wz = float(rospy.get_param(
            "~suspect_lateral_heading_max_rps", 0.12))
        self.acquire_verify_settle_s = float(rospy.get_param(
            "~suspect_orthogonal_settle_s", 0.30))
        self.acquire_verify_timeout_s = float(rospy.get_param(
            "~suspect_orthogonal_verify_timeout_s", 2.8))
        self.acquire_camera_hfov = math.radians(float(rospy.get_param(
            "~suspect_camera_hfov_deg", 70.0)))
        self.acquire_laser_yaw = float(rospy.get_param(
            "~suspect_laser_yaw_rad", -0.07))
        self.acquire_wall_sector = math.radians(float(rospy.get_param(
            "~suspect_wall_sector_deg", 72.0)))
        self.acquire_wall_min_range = float(rospy.get_param(
            "~suspect_wall_min_range_m", 0.20))
        self.acquire_wall_max_range = float(rospy.get_param(
            "~suspect_wall_max_range_m", 1.65))
        self.acquire_wall_inlier = float(rospy.get_param(
            "~suspect_wall_inlier_m", 0.035))
        self.acquire_wall_min_inliers = max(8, int(rospy.get_param(
            "~suspect_wall_min_inliers", 12)))
        self.acquire_wall_min_span = float(rospy.get_param(
            "~suspect_wall_min_span_m", 0.22))
        self.acquire_wall_max_heading = math.radians(float(rospy.get_param(
            "~suspect_wall_max_heading_deg", 55.0)))
        self.near_micro_trigger_distance = float(rospy.get_param(
            "~near_scan_micro_trigger_m", 0.34))
        self.near_micro_timeout_s = float(rospy.get_param(
            "~near_scan_micro_timeout_s", 5.0))
        self.near_micro_stuck_s = float(rospy.get_param(
            "~near_scan_micro_stuck_s", 2.0))
        self.near_micro_progress_m = float(rospy.get_param(
            "~near_scan_micro_progress_m", 0.010))
        self.near_micro_yaw_progress = math.radians(float(rospy.get_param(
            "~near_scan_micro_yaw_progress_deg", 3.0)))
        self.near_micro_forward_gain = float(rospy.get_param(
            "~near_scan_micro_forward_gain", 0.80))
        self.near_micro_lateral_gain = float(rospy.get_param(
            "~near_scan_micro_lateral_gain", 0.80))
        self.near_micro_yaw_gain = float(rospy.get_param(
            "~near_scan_micro_yaw_gain", 0.90))
        self.near_micro_max_forward = float(rospy.get_param(
            "~near_scan_micro_max_forward_mps", 0.10))
        self.near_micro_max_lateral = float(rospy.get_param(
            "~near_scan_micro_max_lateral_mps", 0.045))
        self.near_micro_max_yaw = float(rospy.get_param(
            "~near_scan_micro_max_yaw_rps", 0.20))
        self.near_micro_yaw_slow = math.radians(float(rospy.get_param(
            "~near_scan_micro_yaw_slow_deg", 20.0)))
        rospy.Subscriber(
            rospy.get_param("~odom_topic", "/odom"), Odometry,
            self._suspect_odom_callback, queue_size=3)
        rospy.logwarn(
            "FACTORY_CONTINUOUS_SEARCH_V5_READY suspect_lateral=true "
            "speed=(%.3f,%.3f)mps center=%.1fpx max_shift=%.2fm "
            "side_lidar_filter=true wall_first=true verify_after_motion=true",
            self.suspect_lateral_min_speed,
            self.suspect_lateral_max_speed,
            self.suspect_lateral_center_tolerance,
            self.suspect_lateral_max_shift)

    def _clear_non_target_view_ignore(self):
        self.non_target_view_label = ""
        self.non_target_view_route_index = -1
        self.non_target_view_anchor_yaw = None
        self.non_target_view_min_until = None
        self.non_target_view_max_until = None

    def _arm_non_target_view_ignore(self, label):
        now = rospy.Time.now()
        self.non_target_view_label = label
        self.non_target_view_route_index = self.route_index
        self.non_target_view_anchor_yaw = (
            None if self.pose is None else self.pose[2])
        self.non_target_view_min_until = (
            now + rospy.Duration(max(0.0, self.non_target_view_ignore_min_s)))
        self.non_target_view_max_until = (
            now + rospy.Duration(max(
                self.non_target_view_ignore_min_s,
                self.non_target_view_ignore_max_s)))
        rospy.logwarn(
            "OCR_NON_TARGET_VIEW_IGNORE_ARMED label=%s point=%s yaw=%s "
            "minimum=%.1fs release=%.1fdeg hard=%.1fs",
            label, self.current_route_point()["name"],
            "n/a" if self.non_target_view_anchor_yaw is None else
            "%.1fdeg" % math.degrees(self.non_target_view_anchor_yaw),
            self.non_target_view_ignore_min_s,
            math.degrees(self.non_target_view_release_yaw),
            self.non_target_view_ignore_max_s)

    def _delegate_ocr(self, msg, payload=None):
        stable_label = ""
        target = ""
        before = False
        if isinstance(payload, dict) and payload.get("stable"):
            stable_label = canonical_workshop(payload.get("label", ""))
            with self.integration_lock:
                target = self.target_warehouse
                before = stable_label in self.ocr_confirmed_non_targets
        result = super(SuspectLateralContinuousScanSearchV5,
                       self).ocr_callback(msg)
        if stable_label and stable_label != target and not before:
            with self.integration_lock:
                confirmed = stable_label in self.ocr_confirmed_non_targets
            if confirmed:
                self._arm_non_target_view_ignore(stable_label)
        return result

    def _non_target_view_blocks_ocr(self, msg):
        if not self.non_target_view_label:
            return False
        now = rospy.Time.now()
        current_yaw = None if self.pose is None else self.pose[2]
        same_route = self.route_index == self.non_target_view_route_index
        blocked, yaw_delta = physical_view_ignore_decision(
            now.to_sec(),
            (0.0 if self.non_target_view_min_until is None else
             self.non_target_view_min_until.to_sec()),
            (0.0 if self.non_target_view_max_until is None else
             self.non_target_view_max_until.to_sec()),
            current_yaw, self.non_target_view_anchor_yaw, same_route,
            self.non_target_view_release_yaw)
        if blocked:
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            rospy.logwarn_throttle(
                0.35,
                "OCR_NON_TARGET_VIEW_IGNORED original=%s yaw_delta=%s "
                "all_labels_blocked=true sweep_continues=true",
                self.non_target_view_label,
                "n/a" if not math.isfinite(yaw_delta) else
                "%.1fdeg" % math.degrees(yaw_delta))
            return True

        old_label = self.non_target_view_label
        self._clear_non_target_view_ignore()
        # Discard votes accumulated while the same physical sign was ignored.
        self.ocr_control_pub.publish(String(data="reset"))
        rospy.logwarn(
            "OCR_NON_TARGET_VIEW_RELEASED original=%s yaw_delta=%s "
            "votes_cleared=true",
            old_label,
            "n/a" if not math.isfinite(yaw_delta) else
            "%.1fdeg" % math.degrees(yaw_delta))
        return True

    def _suspect_odom_callback(self, msg):
        self.suspect_lateral_odom_xy = (
            msg.pose.pose.position.x, msg.pose.pose.position.y)

    def _current_xy(self):
        if self.suspect_lateral_odom_xy is not None:
            return self.suspect_lateral_odom_xy
        if self.pose is None:
            return None
        return self.pose[0], self.pose[1]

    def _lateral_displacement(self):
        current = self._current_xy()
        if current is None or self.suspect_lateral_start_pose is None:
            return 0.0
        return math.hypot(
            current[0] - self.suspect_lateral_start_pose[0],
            current[1] - self.suspect_lateral_start_pose[1])

    @staticmethod
    def _bbox_center_error(payload):
        bbox = payload.get("bbox") if isinstance(payload, dict) else None
        width = payload.get("image_width", 0) if isinstance(payload, dict) else 0
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            x0, x1 = float(bbox[0]), float(bbox[2])
            width = float(width)
        except (TypeError, ValueError):
            return None
        if (not all(math.isfinite(value) for value in (x0, x1, width)) or
                width <= 1.0 or x1 <= x0):
            return None
        return 0.5 * (x0 + x1) - 0.5 * width

    def _fallback_heading_for_payload(self, payload):
        if self.pose is None:
            return None
        error_px = self._bbox_center_error(payload)
        width = payload.get("image_width", 0)
        if error_px is None:
            estimated = self.pose[2]
        else:
            estimated = estimated_sign_heading(
                self.pose[2], error_px, float(width),
                self.acquire_camera_hfov)
        return nearest_orthogonal_heading(estimated)

    def _fit_search_wall(self):
        scan = getattr(self, "scan", None)
        if scan is None or not self.scan_fresh():
            return None
        points = []
        cos_yaw = math.cos(self.acquire_laser_yaw)
        sin_yaw = math.sin(self.acquire_laser_yaw)
        for index, distance in enumerate(scan.ranges):
            if (not math.isfinite(distance) or distance < scan.range_min or
                    distance > scan.range_max or
                    distance < self.acquire_wall_min_range or
                    distance > self.acquire_wall_max_range):
                continue
            angle = scan.angle_min + index * scan.angle_increment
            if abs(angle) > self.acquire_wall_sector:
                continue
            x_laser = distance * math.cos(angle)
            y_laser = distance * math.sin(angle)
            points.append((
                cos_yaw * x_laser - sin_yaw * y_laser,
                sin_yaw * x_laser + cos_yaw * y_laser))
        return fit_oriented_front_wall(
            points, self.acquire_wall_inlier,
            self.acquire_wall_min_inliers, self.acquire_wall_min_span,
            self.acquire_wall_max_heading)

    def _current_wall_error(self):
        model = self._fit_search_wall()
        if model is not None:
            return norm_angle(model["heading_error"]), model
        if self.pose is None or self.acquire_fallback_map_yaw is None:
            return None, None
        return norm_angle(self.acquire_fallback_map_yaw - self.pose[2]), None

    def _reanchor_current_scan(self):
        if self.pose is None or not self.scan_active:
            return
        current = (self.pose[0], self.pose[1])
        point = self.current_route_point()
        point["x"], point["y"] = current
        # Candidate relocation must be centered on the translated pose, not on
        # the original d1/d2/d3 coordinate.
        point["nominal_x"], point["nominal_y"] = current
        self.goal_x, self.goal_y = current
        self.acquire_scan_anchor = current
        self.scan_last_yaw = self.pose[2]
        rospy.logwarn(
            "CONTINUOUS_SCAN_REANCHORED point=%s anchor=(%.3f,%.3f) "
            "old_scan_point_reacquire=false",
            point["name"], current[0], current[1])

    def _resume_scan_from_current_heading(self, previous_scan_phase, reason):
        if (previous_scan_phase != "align" or not self.scan_active or
                self.pose is None or not self._is_scan_point()):
            return False
        spec = self.scan_specs[self._scan_point_name()]
        current_yaw = self.pose[2]
        directed = directed_yaw_increment(
            spec["start"], current_yaw, spec["direction"])
        if directed <= spec["travel"] + math.radians(10.0):
            self.scan_progress = min(spec["travel"], directed)
            mode = "preserve_original_endpoint"
        else:
            # The wall correction moved outside the nominal arc. Continue in
            # the requested sweep direction from here instead of turning back.
            self.scan_progress = 0.0
            mode = "shift_arc_to_current_heading"
        now = rospy.Time.now()
        self.scan_phase = "sweep"
        self.scan_command_wz = 0.0
        self.scan_stable = 0
        self.scan_last_yaw = current_yaw
        self.scan_last_tick = now
        self.scan_last_motion = now
        rospy.logwarn(
            "CONTINUOUS_SCAN_RESUME_CURRENT_HEADING point=%s yaw=%.1fdeg "
            "progress=%.1f/%.1fdeg mode=%s reason=%s "
            "return_to_old_start=false",
            self._scan_point_name(), math.degrees(current_yaw),
            math.degrees(self.scan_progress), math.degrees(spec["travel"]),
            mode, reason)
        return True

    def _clear_acquisition(self, reason="", cooldown=False):
        old_phase = self.acquire_phase
        old_label = self.acquire_candidate_label or self.suspect_lateral_label
        old_scan_phase = self.acquire_scan_phase_before
        self.acquire_phase = self.ACQUIRE_IDLE
        self._clear_suspect_lateral(reason, cooldown=cooldown)
        if cooldown and old_label:
            self.suspect_lateral_cooldowns[old_label] = (
                rospy.Time.now() + rospy.Duration(
                    max(0.0, self.suspect_lateral_cooldown_s)))
        self._cancel_candidate_hold()
        self.acquire_candidate_label = ""
        self.acquire_candidate_count = 0
        self.acquire_candidate_last_seen = None
        self.acquire_candidate_bbox = None
        self.acquire_candidate_width = 0.0
        self.acquire_started = None
        self.acquire_settle_until = None
        self.acquire_verify_deadline = None
        self.acquire_fallback_map_yaw = None
        self.acquire_wall_stable = 0
        self.acquire_best_wall_error = float("inf")
        self.acquire_last_wall_progress = None
        self.acquire_last_wall_error = None
        self.acquire_post_lateral = False
        self.acquire_scan_phase_before = "idle"
        self.handoff_gate_ready = False
        if old_phase != self.ACQUIRE_IDLE:
            self._publish_direct_lateral(0.0, 0.0)
            rospy.logwarn(
                "OCR_ORTHOGONAL_ACQUIRE_END phase=%s label=%s reason=%s "
                "scan_anchor=current_pose",
                old_phase, old_label, reason or "cleared")
        if reason not in (
                "parking_handoff", "scan_complete", "shutdown",
                "search_start"):
            self._resume_scan_from_current_heading(old_scan_phase, reason)

    def _start_suspect_confirmation(self, label, payload):
        now = rospy.Time.now()
        self.acquire_candidate_label = label
        self.acquire_candidate_count = 1
        self.acquire_candidate_last_seen = now
        self.acquire_candidate_bbox = payload.get("bbox")
        self.acquire_candidate_width = payload.get("image_width", 0)
        self.acquire_started = now
        self.acquire_fallback_map_yaw = self._fallback_heading_for_payload(
            payload)
        self.acquire_scan_anchor = self._current_xy()
        self.acquire_scan_phase_before = self.scan_phase
        self.handoff_gate_ready = False
        self._clear_edge_recenter()
        self._cancel_candidate_hold()
        self.acquire_phase = self.SUSPECT_CONFIRM
        rospy.logwarn(
            "OCR_SUSPECT_FREEZE label=%s count=1/%d bbox=%s "
            "next=wall_normal_alignment",
            label, self.acquire_confirm_frames, payload.get("bbox"))

    def _begin_wall_alignment(self, payload=None, post_lateral=False):
        now = rospy.Time.now()
        if payload is not None:
            fallback = self._fallback_heading_for_payload(payload)
            if fallback is not None:
                self.acquire_fallback_map_yaw = fallback
        self.acquire_started = now
        self.acquire_wall_stable = 0
        self.acquire_best_wall_error = float("inf")
        self.acquire_last_wall_progress = now
        self.acquire_post_lateral = bool(post_lateral)
        self.handoff_gate_ready = False
        self.ocr_control_pub.publish(String(data="reset"))
        self.acquire_phase = self.WALL_ALIGN
        rospy.logwarn(
            "OCR_WALL_NORMAL_ALIGN_START label=%s fallback=%s "
            "post_lateral=%s parking_handoff_blocked=true",
            self.acquire_candidate_label,
            "n/a" if self.acquire_fallback_map_yaw is None else
            "%.1fdeg" % math.degrees(self.acquire_fallback_map_yaw),
            self.acquire_post_lateral)

    def _begin_orthogonal_verify(self, source, wall_error,
                                 post_lateral=False):
        now = rospy.Time.now()
        self.acquire_settle_until = now + rospy.Duration(
            max(0.0, self.acquire_verify_settle_s))
        self.acquire_verify_deadline = now + rospy.Duration(
            max(self.acquire_verify_settle_s + 0.4,
                self.acquire_verify_timeout_s))
        self.acquire_post_lateral = bool(post_lateral)
        self.acquire_last_wall_error = wall_error
        self.acquire_wall_stable = 0
        self.handoff_gate_ready = False
        self._cancel_candidate_hold()
        self.ocr_control_pub.publish(String(data="reset"))
        self.acquire_phase = self.ORTHOGONAL_VERIFY
        rospy.logwarn(
            "OCR_ORTHOGONAL_VERIFY_START label=%s source=%s "
            "wall_error=%.2fdeg settle=%.2fs votes_reset=true "
            "post_lateral=%s",
            self.acquire_candidate_label, source,
            math.degrees(wall_error), self.acquire_verify_settle_s,
            self.acquire_post_lateral)

    def _cancel_candidate_hold(self):
        with self.integration_lock:
            self.ocr_candidate_hold_until = None
            self.ocr_candidate_label = ""

    def _publish_direct_lateral(self, vy, wz=0.0):
        # FrontFirstStableDynamicRouteTest intentionally suppresses normal
        # requested_vy in the room. Publish directly to /cmd_vel_raw here;
        # the existing DirectionalOmniSafetyMonitor still filters this command.
        msg = Twist()
        msg.linear.y = max(-self.suspect_lateral_max_speed,
                           min(self.suspect_lateral_max_speed, float(vy)))
        msg.angular.z = max(-self.acquire_lateral_heading_max_wz,
                            min(self.acquire_lateral_heading_max_wz,
                                float(wz)))
        self.cmd_pub.publish(msg)

    def _freeze_scan_clocks(self, now):
        self.scan_command_wz = 0.0
        self.scan_last_tick = now
        self.scan_last_motion = now
        if self.pose is not None:
            self.scan_last_yaw = self.pose[2]

    def _clear_suspect_lateral(self, reason="", cooldown=False):
        label = self.suspect_lateral_label
        if cooldown and label:
            self.suspect_lateral_cooldowns[label] = (
                rospy.Time.now() + rospy.Duration(
                    max(0.0, self.suspect_lateral_cooldown_s)))
        was_active = self.suspect_lateral_active
        self.suspect_lateral_active = False
        self.suspect_lateral_label = ""
        self.suspect_lateral_bbox = None
        self.suspect_lateral_image_width = 0.0
        self.suspect_lateral_command = 0.0
        self.suspect_lateral_last_seen = None
        self.suspect_lateral_start_pose = None
        self.suspect_lateral_best_displacement = 0.0
        self.suspect_lateral_last_progress = None
        self.suspect_lateral_best_error = float("inf")
        if was_active:
            self._publish_direct_lateral(0.0)
            rospy.logwarn(
                "OCR_SUSPECT_LATERAL_END label=%s reason=%s "
                "normal_scan_resumes=%s",
                label, reason or "cleared",
                "true" if cooldown else "false")

    def _cooldown_active(self, label, now):
        until = self.suspect_lateral_cooldowns.get(label)
        if until is None:
            return False
        if now < until:
            return True
        self.suspect_lateral_cooldowns.pop(label, None)
        return False

    def _start_or_update_lateral(self, label, payload, command, error_px):
        now = rospy.Time.now()
        if not self.suspect_lateral_active:
            # Publish the active flag last. The navigation timer runs in a
            # different thread and must never observe a half-built state.
            self.suspect_lateral_label = label
            self.suspect_lateral_start_pose = self._current_xy()
            self.suspect_lateral_best_displacement = 0.0
            self.suspect_lateral_last_progress = now
            self.suspect_lateral_best_error = abs(error_px)
            self.suspect_lateral_bbox = payload.get("bbox")
            self.suspect_lateral_image_width = payload.get("image_width", 0)
            self.suspect_lateral_command = command
            self.suspect_lateral_last_seen = now
            self._clear_edge_recenter()
            self._cancel_candidate_hold()
            self.suspect_lateral_active = True
            rospy.logwarn(
                "OCR_SUSPECT_LATERAL_START label=%s bbox=%s error=%.1fpx "
                "cmd_vy=%.3f max_shift=%.2fm ocr=enabled",
                label, payload.get("bbox"), error_px, command,
                self.suspect_lateral_max_shift)
        elif label != self.suspect_lateral_label:
            # Keep following the locked sign until it is stale. A one-frame
            # OCR flip must not reverse the robot's lateral direction.
            return False
        if (abs(error_px) <= self.suspect_lateral_best_error -
                self.suspect_lateral_pixel_progress):
            self.suspect_lateral_best_error = abs(error_px)
            self.suspect_lateral_last_progress = now
        self.suspect_lateral_bbox = payload.get("bbox")
        self.suspect_lateral_image_width = payload.get("image_width", 0)
        self.suspect_lateral_command = command
        self.suspect_lateral_last_seen = now
        return True

    def ocr_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return self._delegate_ocr(msg)

        with self.integration_lock:
            active = (self.integration_phase == self.ACTIVE and
                      self.scan_active)
        if not active:
            return self._delegate_ocr(msg, payload)

        if self._non_target_view_blocks_ocr(msg):
            return

        frame_label = canonical_workshop(payload.get("frame_label", ""))
        stable_label = (canonical_workshop(payload.get("label", ""))
                        if payload.get("stable") else "")
        now = rospy.Time.now()

        if self.acquire_phase == self.LATERAL_ALIGN:
            if frame_label != self.suspect_lateral_label:
                FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(
                    self, msg)
                return
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            state, command, error_px = suspect_lateral_command(
                payload.get("bbox"), payload.get("image_width", 0),
                self.suspect_lateral_center_tolerance,
                self.suspect_lateral_command_sign,
                self.suspect_lateral_gain,
                self.suspect_lateral_min_speed,
                self.suspect_lateral_max_speed)
            if state == "move":
                self._start_or_update_lateral(
                    frame_label, payload, command, error_px)
                return
            if state == "centered":
                label = self.suspect_lateral_label
                moved = self._lateral_displacement()
                self._clear_suspect_lateral("centered", cooldown=False)
                self._reanchor_current_scan()
                self.acquire_candidate_label = label
                self.acquire_candidate_bbox = payload.get("bbox")
                self.acquire_candidate_width = payload.get("image_width", 0)
                rospy.logwarn(
                    "OCR_SUSPECT_CENTERED label=%s error=%.1fpx moved=%.3fm "
                    "new_scan_anchor=%s; wall_realign_before_reverify=true",
                    label, error_px, moved, self.acquire_scan_anchor)
                self._begin_wall_alignment(payload, post_lateral=True)
            return

        if self.acquire_phase == self.ORTHOGONAL_VERIFY:
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            if (self.acquire_settle_until is not None and
                    now < self.acquire_settle_until):
                return
            if not stable_label:
                return
            if stable_label != self.target_warehouse:
                self._delegate_ocr(msg, payload)
                self._clear_acquisition(
                    "orthogonal_non_target_%s" % stable_label)
                return
            state, command, error_px = suspect_lateral_command(
                payload.get("bbox"), payload.get("image_width", 0),
                self.suspect_lateral_center_tolerance,
                self.suspect_lateral_command_sign,
                self.suspect_lateral_gain,
                self.suspect_lateral_min_speed,
                self.suspect_lateral_max_speed)
            wall_error, model = self._current_wall_error()
            if state == "move":
                self.acquire_candidate_label = stable_label
                self.acquire_candidate_bbox = payload.get("bbox")
                self.acquire_candidate_width = payload.get("image_width", 0)
                if self._start_or_update_lateral(
                        stable_label, payload, command, error_px):
                    self.acquire_phase = self.LATERAL_ALIGN
                    rospy.logwarn(
                        "OCR_ORTHOGONAL_TARGET_NEEDS_LATERAL label=%s "
                        "error=%.1fpx wall=%s cmd_vy=%.3f",
                        stable_label, error_px,
                        "n/a" if wall_error is None else
                        "%.2fdeg" % math.degrees(wall_error), command)
                return
            if state != "centered" or wall_error is None:
                return
            if not orthogonal_handoff_ready(
                    wall_error, self.acquire_handoff_wall_tolerance,
                    stable_label, self.target_warehouse, error_px,
                    self.suspect_lateral_center_tolerance):
                self._begin_wall_alignment(
                    payload, post_lateral=self.acquire_post_lateral)
                return
            self.handoff_gate_ready = True
            rospy.logwarn(
                "OCR_ORTHOGONAL_TARGET_VERIFIED label=%s source=%s "
                "wall=%.2fdeg center=%.1fpx parking_handoff=true",
                stable_label, "lidar" if model is not None else "cardinal",
                math.degrees(wall_error), error_px)
            return self._delegate_ocr(msg, payload)

        if self.acquire_phase == self.SUSPECT_CONFIRM:
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            if not frame_label:
                return
            if frame_label != self.acquire_candidate_label:
                self.acquire_candidate_label = frame_label
                self.acquire_candidate_count = 1
                self.acquire_candidate_bbox = payload.get("bbox")
                self.acquire_candidate_width = payload.get("image_width", 0)
                self.acquire_candidate_last_seen = now
                self.acquire_fallback_map_yaw = (
                    self._fallback_heading_for_payload(payload))
                return
            self.acquire_candidate_count += 1
            self.acquire_candidate_last_seen = now
            self.acquire_candidate_bbox = payload.get("bbox")
            self.acquire_candidate_width = payload.get("image_width", 0)
            if self.acquire_candidate_count >= self.acquire_confirm_frames:
                self._begin_wall_alignment(payload, post_lateral=False)
            return

        if self.acquire_phase == self.WALL_ALIGN:
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            if frame_label:
                self.acquire_candidate_bbox = payload.get("bbox")
                self.acquire_candidate_width = payload.get("image_width", 0)
            return

        if not frame_label:
            return self._delegate_ocr(msg, payload)
        with self.integration_lock:
            already_rejected = frame_label in self.ocr_confirmed_non_targets
        if already_rejected:
            return self._delegate_ocr(msg, payload)
        if self._cooldown_active(frame_label, now):
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            return
        FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
        self._start_suspect_confirmation(frame_label, payload)

    def _wall_alignment_control(self):
        now = rospy.Time.now()
        self._freeze_scan_clocks(now)
        wall_error, model = self._current_wall_error()
        fallback_error = None
        if self.pose is not None and self.acquire_fallback_map_yaw is not None:
            fallback_error = norm_angle(
                self.acquire_fallback_map_yaw - self.pose[2])
        state, command, error, source = orthogonal_alignment_command(
            wall_error if model is not None else None, fallback_error,
            self.acquire_wall_tolerance, self.acquire_wall_kp,
            self.acquire_wall_min_wz, self.acquire_wall_max_wz)
        if state == "invalid":
            self._clear_acquisition("wall_alignment_no_pose", cooldown=True)
            return super(SuspectLateralContinuousScanSearchV5,
                         self)._scan_control_loop()
        abs_error = abs(error)
        if abs_error + math.radians(0.35) < self.acquire_best_wall_error:
            self.acquire_best_wall_error = abs_error
            self.acquire_last_wall_progress = now
        if state == "aligned":
            self.acquire_wall_stable += 1
        else:
            self.acquire_wall_stable = 0
        elapsed = (0.0 if self.acquire_started is None else
                   (now - self.acquire_started).to_sec())
        no_progress = (0.0 if self.acquire_last_wall_progress is None else
                       (now - self.acquire_last_wall_progress).to_sec())
        if self.acquire_wall_stable >= self.acquire_wall_stable_frames:
            self._publish_direct_lateral(0.0, 0.0)
            self._begin_orthogonal_verify(
                source, error, post_lateral=self.acquire_post_lateral)
            return
        if (elapsed >= self.acquire_wall_timeout_s or
                (no_progress >= self.acquire_wall_stuck_s and
                 abs(command) >= self.acquire_wall_min_wz)):
            reason = ("wall_alignment_timeout" if
                      elapsed >= self.acquire_wall_timeout_s else
                      "wall_alignment_blocked")
            self._clear_acquisition(reason, cooldown=True)
            return super(SuspectLateralContinuousScanSearchV5,
                         self)._scan_control_loop()
        self.publish_cmd(0.0, command)
        rospy.logwarn_throttle(
            0.25,
            "OCR_WALL_NORMAL_ALIGN label=%s source=%s error=%.2fdeg "
            "cmd_wz=%.3f stable=%d/%d handoff_blocked=true",
            self.acquire_candidate_label, source, math.degrees(error),
            command, self.acquire_wall_stable,
            self.acquire_wall_stable_frames)

    def _orthogonal_verify_control(self):
        now = rospy.Time.now()
        self._freeze_scan_clocks(now)
        self._publish_direct_lateral(0.0, 0.0)
        if (self.acquire_verify_deadline is not None and
                now >= self.acquire_verify_deadline):
            self._clear_acquisition("orthogonal_verify_timeout", cooldown=True)
            self.ocr_control_pub.publish(String(data="reset"))
            return super(SuspectLateralContinuousScanSearchV5,
                         self)._scan_control_loop()
        remaining = (0.0 if self.acquire_verify_deadline is None else
                     max(0.0, (self.acquire_verify_deadline - now).to_sec()))
        rospy.logwarn_throttle(
            0.35,
            "OCR_ORTHOGONAL_VERIFY_WAIT label=%s remaining=%.2fs "
            "fresh_votes_only=true",
            self.acquire_candidate_label, remaining)

    def _lateral_alignment_control(self):
        now = rospy.Time.now()
        self._freeze_scan_clocks(now)
        displacement = self._lateral_displacement()
        if (displacement >= self.suspect_lateral_best_displacement +
                self.suspect_lateral_progress_m):
            self.suspect_lateral_best_displacement = displacement
            self.suspect_lateral_last_progress = now
        age = (float("inf") if self.suspect_lateral_last_seen is None
               else (now - self.suspect_lateral_last_seen).to_sec())
        no_progress = (
            float("inf") if self.suspect_lateral_last_progress is None
            else (now - self.suspect_lateral_last_progress).to_sec())
        reason = lateral_acquisition_exit(
            displacement, self.suspect_lateral_max_shift,
            age, self.suspect_lateral_fresh_s,
            no_progress, self.suspect_lateral_stuck_s)
        if reason:
            self._clear_acquisition(reason, cooldown=True)
            return super(SuspectLateralContinuousScanSearchV5,
                         self)._scan_control_loop()

        wall_error, model = self._current_wall_error()
        heading_wz = 0.0
        if wall_error is not None:
            state, heading_wz, _, _ = orthogonal_alignment_command(
                wall_error if model is not None else None, wall_error,
                self.acquire_wall_tolerance, self.acquire_wall_kp,
                min(self.acquire_wall_min_wz,
                    self.acquire_lateral_heading_max_wz),
                self.acquire_lateral_heading_max_wz)
            if state == "aligned":
                heading_wz = 0.0
            if abs(wall_error) > math.radians(10.0):
                self._publish_direct_lateral(0.0, 0.0)
                self._begin_wall_alignment(post_lateral=True)
                return
        self._publish_direct_lateral(
            self.suspect_lateral_command, heading_wz)
        rospy.logwarn_throttle(
            0.30,
            "OCR_SUSPECT_LATERAL_MOVING label=%s cmd_vy=%.3f "
            "heading_wz=%.3f wall=%s moved=%.3f/%.3fm age=%.2fs",
            self.suspect_lateral_label,
            self.suspect_lateral_command, heading_wz,
            "n/a" if wall_error is None else
            "%.2fdeg" % math.degrees(wall_error),
            displacement, self.suspect_lateral_max_shift, age)

    def _scan_control_loop(self):
        if self.acquire_phase == self.SUSPECT_CONFIRM:
            now = rospy.Time.now()
            self._freeze_scan_clocks(now)
            self._publish_direct_lateral(0.0, 0.0)
            age = (float("inf") if self.acquire_candidate_last_seen is None
                   else (now - self.acquire_candidate_last_seen).to_sec())
            if age > self.acquire_confirm_fresh_s:
                self._clear_acquisition("suspect_not_repeated", cooldown=True)
                return super(SuspectLateralContinuousScanSearchV5,
                             self)._scan_control_loop()
            return
        if self.acquire_phase == self.WALL_ALIGN:
            return self._wall_alignment_control()
        if self.acquire_phase == self.ORTHOGONAL_VERIFY:
            return self._orthogonal_verify_control()
        if self.acquire_phase == self.LATERAL_ALIGN:
            return self._lateral_alignment_control()
        super(SuspectLateralContinuousScanSearchV5,
              self)._scan_control_loop()

    def _clear_near_micro(self, stop=False):
        was_active = self.near_micro_route_index >= 0
        self.near_micro_route_index = -1
        self.near_micro_started = None
        self.near_micro_last_progress = None
        self.near_micro_best_distance = float("inf")
        self.near_micro_best_yaw_error = float("inf")
        self.near_micro_goal_override = None
        self.near_micro_goal_route_index = -1
        if stop and was_active:
            self._publish_local_micro(0.0, 0.0, 0.0)

    def _near_scan_micro_goal(self):
        if (self.near_micro_goal_override is not None and
                self.near_micro_goal_route_index == self.route_index):
            return self.near_micro_goal_override
        point = self.current_route_point()
        return (float(point["x"]), float(point["y"]))

    def _handoff_candidate_final_leg_to_near_micro(self):
        if (getattr(self, "candidate_gate_phase", "") != "to_candidate" or
                getattr(self, "candidate_gate_final", None) is None or
                self.pose is None or self.scan_active or
                not getattr(self, "integration_ready", False) or
                not self._is_scan_point() or
                self.route_index in self.scanned_route_indexes or
                getattr(self, "candidate_waiting", False) or
                not self.pose_fresh() or not self.scan_fresh()):
            return False
        with self.integration_lock:
            if self.integration_phase != self.ACTIVE:
                return False
        now = rospy.Time.now()
        hold_until = getattr(self, "route_transition_hold_until", None)
        if hold_until is not None and now < hold_until:
            return False
        if (self.near_micro_cooldown_until is not None and
                now < self.near_micro_cooldown_until):
            return False
        _, final_distance, _ = self._candidate_gate_distances()
        if final_distance is None:
            return False
        point = self.current_route_point()
        if (final_distance > self.near_micro_trigger_distance or
                final_distance <= float(point["tolerance"])):
            return False

        candidate_kind = self.candidate_gate_kind
        candidate_goal = tuple(self.candidate_gate_final)
        self._clear_candidate_gate()
        self.near_micro_goal_override = candidate_goal
        self.near_micro_goal_route_index = self.route_index
        self.path_world = []
        self.path_index = 0
        if hasattr(self, "_reset_translation_progress"):
            self._reset_translation_progress()
        rospy.logwarn(
            "NEAR_SCAN_CANDIDATE_HANDOFF point=%s kind=%s distance=%.3fm "
            "global_arrival_guidance=false controls=vx_vy_small_wz",
            point["name"], candidate_kind, final_distance)
        return True

    def _publish_local_micro(self, vx, vy, wz):
        command = Twist()
        command.linear.x = max(
            0.0, min(self.near_micro_max_forward, float(vx)))
        command.linear.y = max(
            -self.near_micro_max_lateral,
            min(self.near_micro_max_lateral, float(vy)))
        command.angular.z = max(
            -self.near_micro_max_yaw,
            min(self.near_micro_max_yaw, float(wz)))
        self.cmd_pub.publish(command)

    def _near_scan_micro_eligible(self):
        if (not getattr(self, "integration_ready", False) or
                self.pose is None or self.scan_active or
                not self._is_scan_point() or
                self.route_index in self.scanned_route_indexes or
                getattr(self, "candidate_waiting", False) or
                getattr(self, "candidate_gate_phase", "")):
            return False
        with self.integration_lock:
            if self.integration_phase != self.ACTIVE:
                return False
        now = rospy.Time.now()
        hold_until = getattr(self, "route_transition_hold_until", None)
        if hold_until is not None and now < hold_until:
            return False
        if (self.near_micro_cooldown_until is not None and
                now < self.near_micro_cooldown_until):
            return False
        point = self.current_route_point()
        goal_x, goal_y = self._near_scan_micro_goal()
        distance = math.hypot(
            goal_x - self.pose[0], goal_y - self.pose[1])
        return (distance <= self.near_micro_trigger_distance and
                distance > float(point["tolerance"]))

    def _near_scan_micro_control(self):
        if not self.pose_fresh() or not self.scan_fresh():
            self._publish_local_micro(0.0, 0.0, 0.0)
            return True
        now = rospy.Time.now()
        point = self.current_route_point()
        goal_x, goal_y = self._near_scan_micro_goal()
        dx = goal_x - self.pose[0]
        dy = goal_y - self.pose[1]
        distance = math.hypot(dx, dy)
        cos_yaw = math.cos(self.pose[2])
        sin_yaw = math.sin(self.pose[2])
        forward_error = cos_yaw * dx + sin_yaw * dy
        lateral_error = -sin_yaw * dx + cos_yaw * dy
        target_yaw = self.scan_specs[point["name"]]["start"]
        yaw_error = norm_angle(target_yaw - self.pose[2])
        absolute_yaw_error = abs(yaw_error)
        if self.near_micro_route_index != self.route_index:
            self.near_micro_route_index = self.route_index
            self.near_micro_started = now
            self.near_micro_last_progress = now
            self.near_micro_best_distance = distance
            self.near_micro_best_yaw_error = absolute_yaw_error
            self.path_world = []
            self.path_index = 0
            rospy.logwarn(
                "NEAR_SCAN_MICRO_START point=%s distance=%.3fm "
                "global_replan_suspended=true controls=vx_vy_small_wz",
                point["name"], distance)
        if distance <= self.near_micro_best_distance - self.near_micro_progress_m:
            self.near_micro_best_distance = distance
            self.near_micro_last_progress = now
        if (absolute_yaw_error <=
                self.near_micro_best_yaw_error - self.near_micro_yaw_progress):
            self.near_micro_best_yaw_error = absolute_yaw_error
            self.near_micro_last_progress = now
        elapsed = (now - self.near_micro_started).to_sec()
        stalled = (now - self.near_micro_last_progress).to_sec()
        if (elapsed >= self.near_micro_timeout_s or
                stalled >= self.near_micro_stuck_s):
            reason = "timeout" if elapsed >= self.near_micro_timeout_s else "stuck"
            self.near_micro_cooldown_until = now + rospy.Duration(2.0)
            self._clear_near_micro(stop=True)
            rospy.logwarn(
                "NEAR_SCAN_MICRO_RELEASE point=%s reason=%s distance=%.3fm "
                "planner_fallback=true",
                point["name"], reason, distance)
            return False

        state, vx, vy, wz = near_scan_micro_command(
            forward_error, lateral_error, yaw_error,
            float(point["tolerance"]), self.near_micro_yaw_slow,
            self.near_micro_forward_gain, self.near_micro_lateral_gain,
            self.near_micro_yaw_gain, self.near_micro_max_forward,
            self.near_micro_max_lateral, self.near_micro_max_yaw)
        if state == "reached":
            self._clear_near_micro(stop=True)
            self.publish_zero("NEAR_SCAN_MICRO_REACHED")
            return bool(self._begin_scan())
        if state == "invalid":
            self._clear_near_micro(stop=True)
            return False
        if math.isfinite(getattr(self, "front", float("inf"))) and self.front < 0.23:
            vx = 0.0
        self._publish_local_micro(vx, vy, wz)
        rospy.logwarn_throttle(
            0.30,
            "NEAR_SCAN_MICRO point=%s dist=%.3f body=(%.3f,%.3f) "
            "yaw_error=%.1fdeg cmd=(%.3f,%.3f,%.3f) "
            "external_safety_active=true",
            point["name"], distance, forward_error, lateral_error,
            math.degrees(yaw_error), vx, vy, wz)
        return True

    def control_loop(self, event):
        self._handoff_candidate_final_leg_to_near_micro()
        if self._near_scan_micro_eligible():
            if self._near_scan_micro_control():
                return
        elif self.near_micro_route_index >= 0:
            self._clear_near_micro(stop=True)
        return super(SuspectLateralContinuousScanSearchV5,
                     self).control_loop(event)

    def check_goal(self):
        if (getattr(self, "candidate_gate_phase", "") == "to_approach" and
                self.pose is not None):
            approach_distance, final_distance, travelled = (
                self._candidate_gate_distances())
            if (final_distance is not None and
                    final_distance <= self.candidate_gate_final_tolerance and
                    travelled >= self.candidate_gate_minimum_travel):
                point = self.current_route_point()["name"]
                self._clear_candidate_gate()
                rospy.logwarn(
                    "CONTINUOUS_SCAN_CANDIDATE_FINAL_REACHED point=%s "
                    "final=%.3fm travelled=%.3fm approach=%s; "
                    "approach_gate_bypassed=true",
                    point, final_distance, travelled,
                    "n/a" if approach_distance is None else
                    "%.3fm" % approach_distance)
                return super(CandidateGatedContinuousScanSearchV3,
                             self).check_goal()
        return super(SuspectLateralContinuousScanSearchV5,
                     self).check_goal()

    def _start_search(self):
        self._clear_acquisition("search_start")
        self.suspect_lateral_cooldowns.clear()
        self._clear_non_target_view_ignore()
        return super(SuspectLateralContinuousScanSearchV5,
                      self)._start_search()

    def _handoff_to_parking(self, payload):
        if not self.handoff_gate_ready:
            rospy.logwarn_throttle(
                0.4,
                "OCR_PARKING_HANDOFF_BLOCKED reason=orthogonal_gate_not_ready "
                "phase=%s label=%s",
                self.acquire_phase, payload.get("label", ""))
            return
        self._clear_acquisition("parking_handoff")
        self._clear_non_target_view_ignore()
        return super(SuspectLateralContinuousScanSearchV5,
                      self)._handoff_to_parking(payload)

    def _finish_scan(self):
        self._clear_acquisition("scan_complete")
        self._clear_non_target_view_ignore()
        return super(SuspectLateralContinuousScanSearchV5,
                      self)._finish_scan()

    def shutdown(self):
        self._clear_acquisition("shutdown")
        self._clear_non_target_view_ignore()
        return super(SuspectLateralContinuousScanSearchV5, self).shutdown()


if __name__ == "__main__":
    SuspectLateralContinuousScanSearchV5().spin()
