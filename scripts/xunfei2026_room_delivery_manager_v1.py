#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Large-room delivery using local lidar, move_base, OCR and center parking."""

import json
import math
import os
import signal
import subprocess
import threading
import time

import actionlib
import rospy
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetPlan
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, String
from std_srvs.srv import Empty, SetBool
from tf.transformations import euler_from_quaternion, quaternion_from_euler


def clamp(value, low, high):
    return max(low, min(high, value))


def norm_angle(value):
    return math.atan2(math.sin(value), math.cos(value))


def canonical_workshop(value):
    text = str(value or "").replace(" ", "")
    if "食品" in text:
        return "食品加工车间"
    if "日用" in text or "生活用品" in text:
        return "日用品加工车间"
    if "电子" in text:
        return "电子产品生产车间"
    return text


class Xunfei2026RoomDeliveryManager(object):
    def __init__(self):
        rospy.init_node("xunfei2026_room_delivery_manager")
        self.result_topic = rospy.get_param(
            "~result_topic", "/factory/subtask1_result")
        self.tts_topic = rospy.get_param("~tts_topic", "/factory/tts_text")
        self.ocr_topic = rospy.get_param(
            "~ocr_topic", "/factory_room/ocr_result")
        self.ocr_control_topic = rospy.get_param(
            "~ocr_control_topic", "/factory_room/ocr_control")
        self.ocr_health_topic = rospy.get_param(
            "~ocr_health_topic", "/factory_room/ocr_health")
        self.status_topic = rospy.get_param(
            "~status_topic", "/factory_room/xunfei2026_delivery_status")
        self.cone_control_topic = rospy.get_param(
            "~cone_control_topic", "/factory_room/navigation_active")
        self.parking_state_topic = rospy.get_param(
            "~parking_state_topic", "/factory/sign_center_parking_state")
        self.pose_topic = rospy.get_param("~pose_topic", "/amcl_pose")
        self.odom_topic = rospy.get_param("~odom_topic", "/odom")
        self.scan_topic = rospy.get_param("~scan_topic", "/scan")
        self.cmd_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")

        self.room_launch_pkg = rospy.get_param(
            "~room_launch_pkg", "iden_controller")
        self.room_launch_file = rospy.get_param(
            "~room_launch_file", "xunfei2026_room_move_base_v1.launch")
        self.reuse_first_stage_move_base = bool(rospy.get_param(
            "~reuse_first_stage_move_base", True))
        self.ocr_launch_file = rospy.get_param(
            "~ocr_launch_file", "xunfei2026_factory_ocr_v1.launch")
        self.parking_launch_file = rospy.get_param(
            "~parking_launch_file", "xunfei2026_centerline_parking_v1.launch")

        self.wait_after_tts = float(rospy.get_param("~wait_after_tts_s", 1.0))
        self.entry_approach_x = float(rospy.get_param(
            "~entry_approach_x", -0.85))
        self.entry_approach_y = float(rospy.get_param(
            "~entry_approach_y", -1.41))
        self.entry_x = float(rospy.get_param("~entry_x", -0.85))
        self.entry_y = float(rospy.get_param("~entry_y", -1.90))
        self.entry_yaw = float(rospy.get_param(
            "~entry_yaw", -math.pi / 2.0))
        self.entry_trigger_y = float(rospy.get_param(
            "~entry_trigger_y", -1.75))
        self.entry_goal_timeout = float(rospy.get_param(
            "~entry_goal_timeout_s", 42.0))
        self.move_base_start_timeout = float(rospy.get_param(
            "~move_base_start_timeout_s", 20.0))
        self.goal_timeout = float(rospy.get_param("~goal_timeout_s", 55.0))
        self.scan_timeout = float(rospy.get_param("~scan_timeout_s", 24.0))
        self.sweep_angular_speed = abs(float(rospy.get_param(
            "~sweep_angular_speed", 0.35)))
        self.sweep_min_angular_speed = abs(float(rospy.get_param(
            "~sweep_min_angular_speed", 0.13)))
        self.sweep_slow_angle = math.radians(float(rospy.get_param(
            "~sweep_slow_angle_deg", 24.0)))
        self.sweep_tolerance = math.radians(float(rospy.get_param(
            "~sweep_tolerance_deg", 3.0)))
        self.sweep_settle_s = float(rospy.get_param(
            "~sweep_settle_s", 1.0))
        self.sweep_direction = (1.0 if float(rospy.get_param(
            "~sweep_direction", -1.0)) >= 0.0 else -1.0)
        self.sweep_rotation_clearance = float(rospy.get_param(
            "~sweep_rotation_clearance_m", 0.008))
        self.sweep_slow_clearance = float(rospy.get_param(
            "~sweep_slow_clearance_m", 0.06))
        self.sweep_block_timeout = float(rospy.get_param(
            "~sweep_block_timeout_s", 1.2))
        self.sweep_sensor_fresh_s = float(rospy.get_param(
            "~sweep_sensor_fresh_s", 0.6))
        self.prepark_align_timeout = float(rospy.get_param(
            "~prepark_align_timeout_s", 10.0))
        self.prepark_center_tolerance = float(rospy.get_param(
            "~prepark_center_tolerance_px", 24.0))
        self.prepark_center_stable_frames = int(rospy.get_param(
            "~prepark_center_stable_frames", 2))
        self.prepark_heading_kp = float(rospy.get_param(
            "~prepark_heading_kp", 1.4))
        self.prepark_min_wz = float(rospy.get_param(
            "~prepark_min_wz", 0.10))
        self.prepark_max_wz = float(rospy.get_param(
            "~prepark_max_wz", 0.28))
        self.laser_offset_x = float(rospy.get_param(
            "~laser_offset_x_m", 0.11))
        self.laser_offset_y = float(rospy.get_param(
            "~laser_offset_y_m", 0.0))
        self.laser_yaw = float(rospy.get_param(
            "~laser_yaw_rad", -0.07))
        self.laser_yaw_cos = math.cos(self.laser_yaw)
        self.laser_yaw_sin = math.sin(self.laser_yaw)
        self.robot_half_length = float(rospy.get_param(
            "~robot_half_length_m", 0.171))
        self.robot_half_width = float(rospy.get_param(
            "~robot_half_width_m", 0.128))
        self.rotation_radius = math.hypot(
            self.robot_half_length, self.robot_half_width)
        self.stuck_timeout = float(rospy.get_param("~stuck_timeout_s", 6.0))
        self.progress_distance = float(rospy.get_param(
            "~progress_distance_m", 0.045))
        self.progress_yaw = math.radians(float(rospy.get_param(
            "~progress_yaw_deg", 4.0)))
        self.goal_retries = int(rospy.get_param("~goal_retries", 2))
        self.required_ocr_votes = int(rospy.get_param(
            "~required_ocr_votes", 8))
        self.candidate_ocr_votes = int(rospy.get_param(
            "~candidate_ocr_votes", 5))
        self.candidate_hold_s = float(rospy.get_param(
            "~candidate_hold_s", 1.8))
        self.candidate_cooldown_s = float(rospy.get_param(
            "~candidate_cooldown_s", 1.5))
        self.target_confirm_frames = int(rospy.get_param(
            "~target_confirm_frames", 3))
        self.target_edge_margin_ratio = float(rospy.get_param(
            "~target_edge_margin_ratio", 0.05))
        self.target_vertical_edge_margin_ratio = float(rospy.get_param(
            "~target_vertical_edge_margin_ratio", 0.02))
        self.target_handoff_center_ratio = float(rospy.get_param(
            "~target_handoff_center_ratio", 0.24))
        self.ocr_image_height = float(rospy.get_param(
            "~ocr_image_height", 600.0))
        self.non_target_release_distance = float(rospy.get_param(
            "~non_target_release_distance_m", 0.24))
        self.non_target_hard_release_distance = float(rospy.get_param(
            "~non_target_hard_release_distance_m", 0.52))
        self.non_target_release_yaw = math.radians(float(rospy.get_param(
            "~non_target_release_yaw_deg", 16.0)))
        self.non_target_hard_release_yaw = math.radians(float(rospy.get_param(
            "~non_target_hard_release_yaw_deg", 35.0)))
        self.non_target_blank_s = float(rospy.get_param(
            "~non_target_blank_s", 0.45))
        self.non_target_target_override_frames = int(rospy.get_param(
            "~non_target_target_override_frames", 5))
        self.ocr_fresh_s = float(rospy.get_param("~ocr_fresh_s", 1.2))
        self.ocr_ready_timeout = float(rospy.get_param(
            "~ocr_ready_timeout_s", 12.0))
        self.ocr_max_restarts = int(rospy.get_param(
            "~ocr_max_restarts", 2))
        self.camera_hfov = math.radians(float(rospy.get_param(
            "~camera_hfov_deg", 70.0)))
        self.camera_bearing_sign = float(rospy.get_param(
            "~camera_bearing_sign", -1.0))
        self.wall_standoff = float(rospy.get_param("~wall_standoff_m", 0.58))
        self.parking_timeout = float(rospy.get_param(
            "~parking_timeout_s", 55.0))
        self.parking_target_reacquire_s = float(rospy.get_param(
            "~parking_target_reacquire_s", 8.0))
        self.parking_retries = int(rospy.get_param("~parking_retries", 1))
        # Parking runs in this manager.  Keeping OCR, lidar and move_base alive
        # removes the multi-second roslaunch handoff which used to lose the sign.
        self.parking_wall_distance = float(rospy.get_param(
            "~parking_wall_distance_m", 0.171))
        self.parking_wall_tolerance = float(rospy.get_param(
            "~parking_wall_tolerance_m", 0.012))
        self.parking_heading_tolerance = math.radians(float(rospy.get_param(
            "~parking_heading_tolerance_deg", 2.8)))
        self.parking_center_tolerance = float(rospy.get_param(
            "~parking_center_tolerance_px", 20.0))
        self.parking_stable_frames = int(rospy.get_param(
            "~parking_stable_frames", 3))
        self.parking_heading_kp = float(rospy.get_param(
            "~parking_heading_kp", 1.35))
        self.parking_max_wz = abs(float(rospy.get_param(
            "~parking_max_wz_rps", 0.28)))
        self.parking_min_wz = abs(float(rospy.get_param(
            "~parking_min_wz_rps", 0.115)))
        self.parking_lateral_kp = float(rospy.get_param(
            "~parking_lateral_kp", 0.28))
        self.parking_lateral_sign = float(rospy.get_param(
            "~parking_lateral_sign", -1.0))
        self.parking_max_vy = abs(float(rospy.get_param(
            "~parking_max_vy_mps", 0.11)))
        self.parking_min_vy = abs(float(rospy.get_param(
            "~parking_min_vy_mps", 0.025)))
        self.parking_fast_vx = abs(float(rospy.get_param(
            "~parking_fast_vx_mps", 0.14)))
        self.parking_slow_vx = abs(float(rospy.get_param(
            "~parking_slow_vx_mps", 0.035)))
        self.parking_slow_distance = float(rospy.get_param(
            "~parking_slow_distance_m", 0.10))
        self.parking_lateral_hard_clearance = float(rospy.get_param(
            "~parking_lateral_hard_clearance_m", 0.20))
        self.parking_lateral_slow_clearance = float(rospy.get_param(
            "~parking_lateral_slow_clearance_m", 0.34))
        self.parking_corner_hard_gap = float(rospy.get_param(
            "~parking_corner_hard_gap_m", 0.045))
        self.parking_corner_slow_gap = float(rospy.get_param(
            "~parking_corner_slow_gap_m", 0.14))
        self.parking_corner_turn_hard_gap = float(rospy.get_param(
            "~parking_corner_turn_hard_gap_m", 0.030))
        self.parking_corner_turn_slow_gap = float(rospy.get_param(
            "~parking_corner_turn_slow_gap_m", 0.080))
        self.parking_rear_clear_nudge_speed = abs(float(rospy.get_param(
            "~parking_rear_clear_nudge_speed_mps", 0.055)))
        self.parking_rear_clear_front_min = float(rospy.get_param(
            "~parking_rear_clear_front_min_m", 0.32))
        self.parking_rear_clear_wall_min = float(rospy.get_param(
            "~parking_rear_clear_wall_min_m", 0.42))
        self.reverse_hard_gap = float(rospy.get_param(
            "~wall_reverse_hard_gap_m", 0.040))
        self.reverse_slow_gap = float(rospy.get_param(
            "~wall_reverse_slow_gap_m", 0.120))
        self.side_body_hard_gap = float(rospy.get_param(
            "~wall_side_body_hard_gap_m", 0.025))
        self.side_body_slow_gap = float(rospy.get_param(
            "~wall_side_body_slow_gap_m", 0.080))
        self.wall_return_release_margin = float(rospy.get_param(
            "~wall_return_release_margin_m", 0.080))
        self.side_escape_speed = abs(float(rospy.get_param(
            "~wall_side_escape_speed_mps", 0.045)))
        self.side_flow_hold_s = float(rospy.get_param(
            "~wall_side_flow_hold_s", 0.9))
        self.side_flow_lateral_speed = abs(float(rospy.get_param(
            "~wall_side_flow_lateral_speed_mps", 0.035)))
        self.side_body_confirm_scans = int(rospy.get_param(
            "~wall_side_body_confirm_scans", 3))
        self.motion_prediction_horizon = float(rospy.get_param(
            "~motion_prediction_horizon_s", 0.60))
        self.motion_prediction_margin = float(rospy.get_param(
            "~motion_prediction_margin_m", 0.012))
        self.motion_prediction_steps = int(rospy.get_param(
            "~motion_prediction_steps", 6))
        self.predictive_escape_speed = abs(float(rospy.get_param(
            "~predictive_escape_speed_mps", 0.035)))
        self.side_corridor_margin = float(rospy.get_param(
            "~wall_side_corridor_margin_m", 0.015))
        self.side_corridor_cluster_gap = float(rospy.get_param(
            "~wall_side_corridor_cluster_gap_m", 0.10))
        self.side_corridor_max_cluster_span = float(rospy.get_param(
            "~wall_side_corridor_max_cluster_span_m", 0.24))
        self.side_corridor_center_tolerance = float(rospy.get_param(
            "~wall_side_corridor_center_tolerance_m", 0.025))
        self.side_corridor_center_kp = float(rospy.get_param(
            "~wall_side_corridor_center_kp", 0.8))
        self.side_corridor_max_vx = abs(float(rospy.get_param(
            "~wall_side_corridor_max_vx_mps", 0.07)))
        self.side_corridor_vy = abs(float(rospy.get_param(
            "~wall_side_corridor_vy_mps", 0.09)))
        self.parking_recenter_threshold = float(rospy.get_param(
            "~parking_recenter_threshold_px", 48.0))
        self.parking_front_emergency = float(rospy.get_param(
            "~parking_front_emergency_m", 0.105))
        self.parking_target_memory_s = float(rospy.get_param(
            "~parking_target_memory_s", 3.0))
        self.parking_center_filter_alpha = float(rospy.get_param(
            "~parking_center_filter_alpha", 0.45))
        self.parking_center_max_step = float(rospy.get_param(
            "~parking_center_max_step_px", 70.0))
        self.parking_approach_max_vy = abs(float(rospy.get_param(
            "~parking_approach_max_vy_mps", 0.060)))
        self.parking_translation_heading_gate = math.radians(float(
            rospy.get_param("~parking_translation_heading_gate_deg", 6.0)))
        self.parking_recovery_trigger_vy = abs(float(rospy.get_param(
            "~parking_recovery_trigger_vy_mps", 0.015)))
        self.parking_escape_lateral_speed = abs(float(rospy.get_param(
            "~parking_escape_lateral_speed_mps", 0.035)))
        self.parking_forward_escape_margin = float(rospy.get_param(
            "~parking_forward_escape_margin_m", 0.020))
        self.mission_timeout = float(rospy.get_param(
            "~mission_timeout_s", 420.0))

        # The rectangle is a route hint, not a set of poses that must be
        # reached exactly.  Once near b1, lidar wall fitting closes the loop.
        self.wall_route_speed = abs(float(rospy.get_param(
            "~wall_route_speed_mps", 0.14)))
        self.wall_route_min_speed = abs(float(rospy.get_param(
            "~wall_route_min_speed_mps", 0.055)))
        self.wall_route_max_vx = abs(float(rospy.get_param(
            "~wall_route_max_vx_mps", 0.10)))
        self.wall_route_max_wz = abs(float(rospy.get_param(
            "~wall_route_max_wz_rps", 0.30)))
        self.wall_distance_kp = float(rospy.get_param(
            "~wall_distance_kp", 0.72))
        self.wall_heading_kp = float(rospy.get_param(
            "~wall_heading_kp", 1.15))
        self.wall_nominal_yaw_kp = float(rospy.get_param(
            "~wall_nominal_yaw_kp", 0.45))
        self.wall_target_default = float(rospy.get_param(
            "~wall_target_default_m", 0.75))
        self.wall_target_min = float(rospy.get_param(
            "~wall_target_min_m", 0.48))
        self.wall_target_max = float(rospy.get_param(
            "~wall_target_max_m", 0.98))
        self.wall_fit_half_angle = math.radians(float(rospy.get_param(
            "~wall_fit_half_angle_deg", 38.0)))
        self.wall_fit_max_range = float(rospy.get_param(
            "~wall_fit_max_range_m", 1.35))
        self.wall_fit_inlier = float(rospy.get_param(
            "~wall_fit_inlier_m", 0.13))
        self.wall_fit_min_points = int(rospy.get_param(
            "~wall_fit_min_points", 9))
        self.wall_sensor_lost_s = float(rospy.get_param(
            "~wall_sensor_lost_s", 2.0))
        self.segment_end_tolerance = float(rospy.get_param(
            "~wall_segment_end_tolerance_m", 0.10))
        self.segment_timeout_scale = float(rospy.get_param(
            "~wall_segment_timeout_scale", 3.2))
        self.corner_early_ratio = float(rospy.get_param(
            "~wall_corner_early_ratio", 0.50))
        self.corner_side_distance = float(rospy.get_param(
            "~wall_corner_side_distance_m", 0.30))
        self.corner_wall_span = math.radians(float(rospy.get_param(
            "~wall_corner_span_deg", 28.0)))
        self.corner_turn_tolerance = math.radians(float(rospy.get_param(
            "~wall_corner_turn_tolerance_deg", 4.0)))
        self.corner_turn_min_wz = abs(float(rospy.get_param(
            "~wall_corner_turn_min_wz_rps", 0.12)))
        self.corner_turn_max_wz = abs(float(rospy.get_param(
            "~wall_corner_turn_max_wz_rps", 0.34)))
        self.corner_rotation_clearance = float(rospy.get_param(
            "~wall_corner_rotation_clearance_m", 0.010))
        self.cone_side_slow = float(rospy.get_param(
            "~wall_cone_side_slow_m", 0.34))
        self.cone_side_stop = float(rospy.get_param(
            "~wall_cone_side_stop_m", 0.24))
        self.cone_side_hard = float(rospy.get_param(
            "~wall_cone_side_hard_m", 0.17))
        self.cone_avoid_max_vx = abs(float(rospy.get_param(
            "~wall_cone_avoid_max_vx_mps", 0.075)))
        self.detour_away_distance = float(rospy.get_param(
            "~wall_detour_away_m", 0.18))
        self.detour_min_away = float(rospy.get_param(
            "~wall_detour_min_away_m", 0.08))
        self.detour_pass_distance = float(rospy.get_param(
            "~wall_detour_pass_m", 0.44))
        self.detour_max_away = float(rospy.get_param(
            "~wall_detour_max_away_m", 0.24))
        self.detour_speed = abs(float(rospy.get_param(
            "~wall_detour_speed_mps", 0.10)))
        self.detour_hard_clearance = float(rospy.get_param(
            "~wall_detour_hard_clearance_m", 0.19))
        # Match the proven first-stage TEB acceleration envelope.  The base
        # driver applies its existing alpha=0.6 low-pass filter afterwards.
        self.direct_accel_x = abs(float(rospy.get_param(
            "~wall_direct_accel_x_mps2", 0.50)))
        self.direct_accel_y = abs(float(rospy.get_param(
            "~wall_direct_accel_y_mps2", 0.50)))
        self.direct_accel_wz = abs(float(rospy.get_param(
            "~wall_direct_accel_wz_rps2", 1.30)))
        self.direct_smooth_stop_s = float(rospy.get_param(
            "~wall_direct_smooth_stop_s", 0.35))

        self.room_min_x = float(rospy.get_param("~room_min_x", -2.23))
        self.room_max_x = float(rospy.get_param("~room_max_x", 2.80))
        self.room_min_y = float(rospy.get_param("~room_min_y", -3.28))
        self.room_max_y = float(rospy.get_param("~room_max_y", -1.18))

        self.wall_route_points = [
            ("b1", float(rospy.get_param("~wall_b1_x", -0.43)),
             float(rospy.get_param("~wall_b1_y", -2.54))),
            ("a3", float(rospy.get_param("~wall_a3_x", -1.43)),
             float(rospy.get_param("~wall_a3_y", -2.54))),
            ("a4", float(rospy.get_param("~wall_a4_x", -1.43)),
             float(rospy.get_param("~wall_a4_y", -2.04))),
            ("a1", float(rospy.get_param("~wall_a1_x", 2.04)),
             float(rospy.get_param("~wall_a1_y", -2.04))),
            ("a2", float(rospy.get_param("~wall_a2_x", 2.04)),
             float(rospy.get_param("~wall_a2_y", -2.54))),
            ("b2", float(rospy.get_param("~wall_b2_x", 1.04)),
             float(rospy.get_param("~wall_b2_y", -2.54))),
        ]
        # Facing yaw for each segment.  Every segment moves toward body-right
        # while the camera/front lidar continue looking at the wall.
        self.wall_route_yaws = [
            -math.pi / 2.0, math.pi, math.pi / 2.0, 0.0, -math.pi / 2.0,
        ]

        self.lock = threading.RLock()
        self.started = False
        self.finished = False
        self.target_warehouse = ""
        self.selected_item = ""
        self.latest_ocr = None
        self.ocr_health = ""
        self.ocr_health_stamp = 0.0
        self.ocr_restart_count = 0
        self.pose = None
        self.odom_yaw = None
        self.odom_pose = None
        self.odom_stamp = 0.0
        self.rotation_clearance = float("inf")
        self.scan_samples = []
        self.scan_stamp = 0.0
        self.target_event = threading.Event()
        self.candidate_event = threading.Event()
        self.candidate_last_seen = 0.0
        self.candidate_cooldown_until = 0.0
        self.target_confirm_count = 0
        self.non_target_view_label = ""
        self.non_target_view_anchor = None
        self.non_target_odom_anchor = None
        self.non_target_target_frames = 0
        self.non_target_blank_since = None
        self.target_snapshot = None
        self.parking_target_ocr = None
        self.parking_target_stamp = 0.0
        self.parking_center_filtered = None
        self.parking_center_width = 0.0
        self.parking_center_source_stamp = 0.0
        self.parking_center_aligned = False
        self.parking_active = False
        self.parking_wrong_label = ""
        self.parking_wrong_event = threading.Event()
        self.active_wall_segment_index = 0
        self.parking_side_blocker = None
        self.parking_lateral_requested = 0.0
        self.parking_lateral_guarded = 0.0
        self.room_search_active = False
        self.parking_state = ""
        self.parking_payload = None
        self.room_process = None
        self.ocr_process = None
        self.parking_process = None
        self.move_base = None
        self.make_plan = None
        self.clear_costmaps = None
        self.shutdown_started = False
        self.direct_cmd = [0.0, 0.0, 0.0]
        self.direct_cmd_stamp = time.monotonic()

        self.tts_pub = rospy.Publisher(self.tts_topic, String, queue_size=3)
        self.ocr_control_pub = rospy.Publisher(
            self.ocr_control_topic, String, queue_size=3, latch=True)
        self.cone_control_pub = rospy.Publisher(
            self.cone_control_topic, Bool, queue_size=1, latch=True)
        self.status_pub = rospy.Publisher(
            self.status_topic, String, queue_size=10, latch=True)
        self.cmd_pub = rospy.Publisher(self.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(self.result_topic, String, self.result_callback,
                         queue_size=5)
        rospy.Subscriber(self.ocr_topic, String, self.ocr_callback, queue_size=10)
        rospy.Subscriber(self.ocr_health_topic, String,
                         self.ocr_health_callback, queue_size=5)
        rospy.Subscriber(self.pose_topic, PoseWithCovarianceStamped,
                         self.pose_callback, queue_size=10)
        rospy.Subscriber(self.odom_topic, Odometry,
                         self.odom_callback, queue_size=20)
        rospy.Subscriber(self.scan_topic, LaserScan,
                         self.scan_callback, queue_size=5)
        rospy.Subscriber(self.parking_state_topic, String,
                         self.parking_state_callback, queue_size=10)
        rospy.on_shutdown(self.shutdown)
        self.publish_state("WAITING_SUBTASK1_RESULT")

    def publish_state(self, state, **values):
        payload = {
            "state": state,
            "stamp": time.time(),
            "item": self.selected_item,
            "warehouse": self.target_warehouse,
        }
        payload.update(values)
        self.status_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        rospy.logwarn("XUNFEI2026_ROOM_STATE %s",
                      json.dumps(payload, ensure_ascii=False))

    def result_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if str(payload.get("status", "")).lower() != "success":
            return
        item = str(payload.get("selected_item", "")).strip()
        warehouse = canonical_workshop(payload.get("target_warehouse", ""))
        if not item or not warehouse:
            return
        with self.lock:
            if self.started:
                return
            self.started = True
            self.selected_item = item
            self.target_warehouse = warehouse
        rospy.logwarn("XUNFEI2026_ROOM_TASK_ACCEPTED item=%s warehouse=%s",
                      item, warehouse)
        worker = threading.Thread(target=self.mission_thread)
        worker.daemon = True
        worker.start()

    def pose_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        with self.lock:
            self.pose = (msg.pose.pose.position.x,
                         msg.pose.pose.position.y, yaw, time.monotonic())

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = euler_from_quaternion((q.x, q.y, q.z, q.w))[2]
        with self.lock:
            self.odom_yaw = yaw
            self.odom_pose = (msg.pose.pose.position.x,
                              msg.pose.pose.position.y, yaw)
            self.odom_stamp = time.monotonic()

    def laser_to_base(self, x_laser, y_laser):
        """Apply the same planar transform published by ydlidar.launch."""
        return (
            self.laser_offset_x + self.laser_yaw_cos * x_laser -
            self.laser_yaw_sin * y_laser,
            self.laser_offset_y + self.laser_yaw_sin * x_laser +
            self.laser_yaw_cos * y_laser,
        )

    def base_to_laser(self, x_base, y_base):
        """Inverse of laser_to_base for lidar-frame wall calculations."""
        dx = x_base - self.laser_offset_x
        dy = y_base - self.laser_offset_y
        return (
            self.laser_yaw_cos * dx + self.laser_yaw_sin * dy,
            -self.laser_yaw_sin * dx + self.laser_yaw_cos * dy,
        )

    def scan_callback(self, msg):
        clearance = float("inf")
        samples = []
        for index, distance in enumerate(msg.ranges):
            if (not math.isfinite(distance) or distance < msg.range_min or
                    distance > msg.range_max):
                continue
            angle = msg.angle_min + index * msg.angle_increment
            x_laser = distance * math.cos(angle)
            y_laser = distance * math.sin(angle)
            x_base, y_base = self.laser_to_base(x_laser, y_laser)
            base_distance = math.hypot(x_base, y_base)
            base_angle = math.atan2(y_base, x_base)
            samples.append((base_angle, base_distance, x_base, y_base))
            clearance = min(
                clearance,
                base_distance - self.rotation_radius)
        with self.lock:
            self.rotation_clearance = clearance
            self.scan_samples = samples
            self.scan_stamp = time.monotonic()

    def ocr_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        payload["label"] = canonical_workshop(payload.get("label", ""))
        payload["frame_label"] = canonical_workshop(
            payload.get("frame_label", ""))
        payload["received_monotonic"] = time.monotonic()
        with self.lock:
            self.latest_ocr = payload
            target = self.target_warehouse
            pose = self.pose
            odom_pose = self.odom_pose
        stable = bool(payload.get("stable", False))
        votes = int(payload.get("votes", 0) or 0)
        now = time.monotonic()
        frame_label = payload.get("frame_label", "")
        stable_label = payload.get("label", "") if stable else ""

        if not self.room_search_active or not target or pose is None:
            return

        # A confidently identified different workshop is a completed negative
        # observation.  Suppress every class until this physical view has been
        # left, otherwise OCR's sliding vote window can briefly rename the same
        # edge-clipped sign as the requested workshop.
        confirmed_non_target = (
            stable_label and stable_label != target and
            frame_label == stable_label and votes >= self.required_ocr_votes)
        if confirmed_non_target:
            with self.lock:
                # A stable contradictory workshop invalidates the parking
                # target during every parking phase.  Previously this was only
                # honoured before image centering; a later contradiction could
                # therefore leave the robot approaching the wrong wall using a
                # stale target center.
                parking_abort = self.parking_active
                if self.non_target_view_label != stable_label:
                    self.non_target_view_label = stable_label
                    self.non_target_view_anchor = tuple(pose)
                    self.non_target_odom_anchor = (
                        None if odom_pose is None else tuple(odom_pose))
                    self.non_target_target_frames = 0
                    rospy.logwarn(
                        "OCR_NON_TARGET_VIEW_LOCKED label=%s target=%s; "
                        "scan_continues=true", stable_label, target)
                self.non_target_blank_since = None
                self.target_confirm_count = 0
                self.parking_target_ocr = None
                self.parking_target_stamp = 0.0
                if parking_abort:
                    self.parking_wrong_label = stable_label
                    self.parking_wrong_event.set()
            self.candidate_event.clear()
            if parking_abort:
                rospy.logwarn(
                    "OCR_PARKING_WRONG_WORKSHOP label=%s target=%s; "
                    "parking_aborts_and_wall_scan_resumes=true",
                    stable_label, target)
            return

        with self.lock:
            ignored_label = self.non_target_view_label
            anchor = self.non_target_view_anchor
            odom_anchor = self.non_target_odom_anchor
            blank_since = self.non_target_blank_since
            target_frames = self.non_target_target_frames
        if ignored_label:
            if frame_label:
                blank_since = None
            elif blank_since is None:
                blank_since = now
            moved = (0.0 if anchor is None else
                     math.hypot(pose[0] - anchor[0], pose[1] - anchor[1]))
            yaw_delta = (0.0 if anchor is None else
                         abs(norm_angle(pose[2] - anchor[2])))
            odom_moved = (0.0 if odom_anchor is None or odom_pose is None else
                          math.hypot(odom_pose[0] - odom_anchor[0],
                                     odom_pose[1] - odom_anchor[1]))
            odom_yaw_delta = (
                0.0 if odom_anchor is None or odom_pose is None else
                abs(norm_angle(odom_pose[2] - odom_anchor[2])))
            target_override_frame = (
                frame_label == target and votes >= self.candidate_ocr_votes)
            target_frames = target_frames + 1 if target_override_frame else 0
            target_override = (
                target_frames >= self.non_target_target_override_frames)
            blank_long_enough = (
                blank_since is not None and now - blank_since >=
                self.non_target_blank_s)
            released = (
                moved >= self.non_target_hard_release_distance or
                yaw_delta >= self.non_target_hard_release_yaw or
                odom_moved >= self.non_target_hard_release_distance or
                odom_yaw_delta >= self.non_target_hard_release_yaw or
                target_override or
                (blank_long_enough and
                 (moved >= self.non_target_release_distance or
                  yaw_delta >= self.non_target_release_yaw or
                  odom_moved >= self.non_target_release_distance or
                  odom_yaw_delta >= self.non_target_release_yaw)))
            with self.lock:
                self.non_target_blank_since = blank_since
                self.target_confirm_count = 0
                self.non_target_target_frames = target_frames
            self.candidate_event.clear()
            if not released:
                rospy.logwarn_throttle(
                    0.5, "OCR_NON_TARGET_VIEW_IGNORED label=%s frame=%s "
                    "moved=%.2f/%.2f yaw=%.1f/%.1fdeg target_frames=%d/%d "
                    "scan_continues=true",
                    ignored_label, frame_label or "none", moved,
                    odom_moved, math.degrees(yaw_delta),
                    math.degrees(odom_yaw_delta), target_frames,
                    self.non_target_target_override_frames)
                return
            with self.lock:
                self.non_target_view_label = ""
                self.non_target_view_anchor = None
                self.non_target_odom_anchor = None
                self.non_target_blank_since = None
                self.non_target_target_frames = 0
            if not target_override:
                self.ocr_control_pub.publish(String(data="reset"))
            rospy.logwarn(
                "OCR_NON_TARGET_VIEW_RELEASED label=%s moved=%.2f/%.2f "
                "yaw=%.1f/%.1fdeg target_override=%s votes_reset=%s",
                ignored_label, moved, odom_moved, math.degrees(yaw_delta),
                math.degrees(odom_yaw_delta), str(target_override),
                str(not target_override))
            if not target_override:
                return

        bbox = payload.get("bbox")
        width = float(payload.get("image_width", 0) or 0)
        height = float(payload.get("image_height", self.ocr_image_height) or
                       self.ocr_image_height)
        bbox_visible = False
        bbox_centered = False
        if (isinstance(bbox, (list, tuple)) and len(bbox) >= 4 and
                width > 1.0 and height > 1.0):
            margin_x = self.target_edge_margin_ratio * width
            margin_y = self.target_vertical_edge_margin_ratio * height
            center_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
            bbox_visible = (float(bbox[0]) >= margin_x and
                            float(bbox[2]) <= width - margin_x and
                            float(bbox[1]) >= margin_y and
                            float(bbox[3]) <= height - margin_y)
            bbox_centered = (
                abs(center_x - 0.5 * width) <=
                self.target_handoff_center_ratio * width)
        target_label_ready = (
            stable and stable_label == target and frame_label == target and
            votes >= self.required_ocr_votes)
        if target_label_ready and (not bbox_visible or not bbox_centered):
            rospy.logwarn_throttle(
                0.5, "OCR_TARGET_FRAMING_WAIT bbox=%s visible=%s centered=%s "
                "continue_motion=true", str(bbox), str(bbox_visible),
                str(bbox_centered))
        current_target = (
            target_label_ready and bbox_visible and bbox_centered)
        with self.lock:
            self.target_confirm_count = (
                self.target_confirm_count + 1 if current_target else 0)
            confirm_count = self.target_confirm_count

        # Pause only for a stable, current-frame, fully visible target.  Votes
        # belonging to a previous non-target can no longer trigger this hold.
        candidate_confirm_frames = max(2, self.target_confirm_frames - 1)
        if (current_target and confirm_count >= candidate_confirm_frames and
                now >= self.candidate_cooldown_until):
            with self.lock:
                self.candidate_last_seen = now
                # Lock only OCR belonging to the requested workshop.  A stable
                # different workshop is handled above as a parking veto.
                self.parking_target_ocr = dict(payload)
                self.parking_target_stamp = now
            self.candidate_event.set()
        elif not current_target:
            self.candidate_event.clear()
        if current_target and confirm_count >= self.target_confirm_frames:
            snapshot = {"ocr": dict(payload), "pose": tuple(pose)}
            with self.lock:
                self.target_snapshot = snapshot
                self.parking_target_ocr = dict(payload)
                self.parking_target_stamp = now
            self.target_event.set()
            rospy.logwarn_throttle(
                0.5, "XUNFEI2026_TARGET_OCR label=%s votes=%d "
                "fresh_frames=%d bbox=%s",
                target, votes, confirm_count, str(payload.get("bbox")))

    def ocr_health_callback(self, msg):
        with self.lock:
            self.ocr_health = str(msg.data or "").strip().lower()
            self.ocr_health_stamp = time.monotonic()

    def parking_state_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self.lock:
            self.parking_state = str(payload.get("state", ""))
            self.parking_payload = payload

    @staticmethod
    def start_process(command):
        rospy.logwarn("XUNFEI2026_START_PROCESS %s", " ".join(command))
        return subprocess.Popen(command, preexec_fn=os.setsid)

    @staticmethod
    def stop_process(process, name):
        if process is None or process.poll() is not None:
            return
        rospy.logwarn("XUNFEI2026_STOP_PROCESS %s", name)
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGINT)
            process.wait(timeout=3.0)
        except Exception:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except Exception:
                pass

    def stop_robot(self, repeats=10):
        if self.move_base is not None:
            try:
                self.move_base.cancel_all_goals()
            except Exception:
                pass
        for _ in range(repeats):
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.025)
        self.direct_cmd = [0.0, 0.0, 0.0]
        self.direct_cmd_stamp = time.monotonic()

    @staticmethod
    def slew_value(current, target, limit):
        return current + clamp(target - current, -limit, limit)

    def publish_direct_command(self, vx=0.0, vy=0.0, wz=0.0):
        now = time.monotonic()
        dt = clamp(now - self.direct_cmd_stamp, 0.02, 0.12)
        next_vx = self.slew_value(
            self.direct_cmd[0], float(vx), self.direct_accel_x * dt)
        next_vy = self.slew_value(
            self.direct_cmd[1], float(vy), self.direct_accel_y * dt)
        next_wz = self.slew_value(
            self.direct_cmd[2], float(wz), self.direct_accel_wz * dt)
        command = Twist()
        command.linear.x = next_vx
        command.linear.y = next_vy
        command.angular.z = next_wz
        self.cmd_pub.publish(command)
        self.direct_cmd = [next_vx, next_vy, next_wz]
        self.direct_cmd_stamp = now
        return command

    def smooth_stop_robot(self):
        deadline = time.monotonic() + max(0.08, self.direct_smooth_stop_s)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            command = self.publish_direct_command()
            if (abs(command.linear.x) < 0.006 and
                    abs(command.linear.y) < 0.006 and
                    abs(command.angular.z) < 0.015):
                break
            rospy.sleep(0.025)
        self.cmd_pub.publish(Twist())
        self.direct_cmd = [0.0, 0.0, 0.0]
        self.direct_cmd_stamp = time.monotonic()

    def restore_camera_for_ocr(self):
        try:
            rospy.wait_for_service("/ucar_camera/set_exposure_profile", timeout=2.0)
            result = rospy.ServiceProxy(
                "/ucar_camera/set_exposure_profile", SetBool)(False)
            rospy.logwarn("XUNFEI2026_CAMERA_OCR_EXPOSURE success=%s message=%s",
                          str(result.success), result.message)
        except Exception as exc:
            rospy.logwarn("camera exposure restore unavailable: %s", exc)

    def start_ocr(self):
        self.restore_camera_for_ocr()
        with self.lock:
            self.ocr_health = ""
            self.ocr_health_stamp = 0.0
        self.ocr_process = self.start_process([
            "roslaunch", self.room_launch_pkg, self.ocr_launch_file])
        deadline = time.monotonic() + self.ocr_ready_timeout
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.ocr_process.poll() is not None:
                raise RuntimeError("OCR launch exited before ready")
            with self.lock:
                health = self.ocr_health
            if health == "ready":
                break
            if health.startswith("runtime_error"):
                raise RuntimeError("OCR reported {}".format(health))
            rospy.sleep(0.05)
        else:
            raise RuntimeError("OCR ready timeout")
        self.ocr_control_pub.publish(String(data="reset"))
        self.ocr_control_pub.publish(String(data="enable"))
        self.publish_state("OCR_RUNTIME_READY", health="ready",
                           restart_count=self.ocr_restart_count)

    def ensure_ocr_running(self):
        with self.lock:
            health = self.ocr_health
        process_alive = (
            self.ocr_process is not None and self.ocr_process.poll() is None)
        if process_alive and health == "ready":
            return True
        if self.ocr_restart_count >= self.ocr_max_restarts:
            self.publish_state(
                "OCR_RUNTIME_UNAVAILABLE", health=health,
                restart_count=self.ocr_restart_count)
            return False
        self.stop_robot(8)
        self.publish_state(
            "OCR_RUNTIME_RESTARTING", health=health,
            restart_count=self.ocr_restart_count + 1)
        self.stop_process(self.ocr_process, "OCR watchdog restart")
        self.ocr_process = None
        self.ocr_restart_count += 1
        try:
            self.start_ocr()
            return True
        except Exception as exc:
            rospy.logerr("OCR watchdog restart failed: %s", exc)
            return False

    def room_entry_crossed(self):
        pose = self.current_pose()
        return pose is not None and pose[1] <= self.entry_trigger_y

    def clear_existing_costmaps(self, reason):
        try:
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
            rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
            rospy.logwarn("FIRST_STAGE_COSTMAPS_CLEARED reason=%s", reason)
        except Exception as exc:
            rospy.logwarn("first-stage clear costmaps unavailable: %s", exc)
        rospy.sleep(0.35)

    def enter_room_with_first_stage_navigation(self):
        self.publish_state(
            "FIRST_STAGE_ROOM_ENTRY_START",
            approach_x=self.entry_approach_x,
            approach_y=self.entry_approach_y,
            entry_x=self.entry_x, entry_y=self.entry_y,
            trigger_y=self.entry_trigger_y)
        self.move_base = actionlib.SimpleActionClient(
            "/move_base", MoveBaseAction)
        if not self.move_base.wait_for_server(rospy.Duration(10.0)):
            raise RuntimeError("first-stage move_base action server unavailable")
        if self.room_entry_crossed():
            self.publish_state("FIRST_STAGE_ROOM_ENTRY_ALREADY_CROSSED")
            return True

        approach_result = self.send_goal(
            self.entry_approach_x, self.entry_approach_y, self.entry_yaw,
            "doorway_approach_first_stage", self.entry_goal_timeout,
            watch_target=False)
        if self.room_entry_crossed():
            self.publish_state("FIRST_STAGE_ROOM_ENTRY_CROSSED_DURING_APPROACH")
            return True
        if approach_result != "SUCCEEDED":
            self.clear_existing_costmaps(
                "doorway approach {}".format(approach_result))

        # The entry point itself is deliberately loose.  Crossing y=-1.75 is
        # sufficient; the robot must not waste time trying to settle exactly
        # on a waypoint in the narrow doorway.
        for attempt, offset_x in enumerate((0.0, 0.08, -0.08), 1):
            result = self.send_goal(
                self.entry_x + offset_x, self.entry_y, self.entry_yaw,
                "room_entry_first_stage_{}".format(attempt),
                self.entry_goal_timeout, watch_target=False,
                success_y_below=self.entry_trigger_y)
            if result in ("SUCCEEDED", "SUCCEEDED_THRESHOLD") or \
                    self.room_entry_crossed():
                self.stop_robot(12)
                pose = self.current_pose()
                self.publish_state(
                    "FIRST_STAGE_ROOM_ENTRY_COMPLETE", attempt=attempt,
                    pose_y=None if pose is None else pose[1])
                return True
            self.clear_existing_costmaps(
                "room entry attempt {} {}".format(attempt, result))
        return False

    def replace_move_base(self):
        if self.reuse_first_stage_move_base:
            self.publish_state("REUSING_FIRST_STAGE_NAVIGATION")
            if self.move_base is None:
                self.move_base = actionlib.SimpleActionClient(
                    "/move_base", MoveBaseAction)
            if not self.move_base.wait_for_server(rospy.Duration(3.0)):
                raise RuntimeError("existing first-stage move_base unavailable")
            rospy.wait_for_service("/move_base/make_plan", timeout=3.0)
            rospy.wait_for_service("/move_base/clear_costmaps", timeout=3.0)
            self.make_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
            self.clear_costmaps = rospy.ServiceProxy(
                "/move_base/clear_costmaps", Empty)
            self.cone_control_pub.publish(Bool(data=True))
            self.publish_state("ROOM_NAVIGATION_READY", reused=True)
            return
        self.publish_state("SWITCHING_TO_ROOM_NAVIGATION")
        self.stop_robot(8)
        subprocess.call(
            ["rosnode", "kill", "/move_base"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rospy.sleep(1.0)
        self.room_process = self.start_process([
            "roslaunch", self.room_launch_pkg, self.room_launch_file])
        self.move_base = actionlib.SimpleActionClient(
            "/move_base", MoveBaseAction)
        if not self.move_base.wait_for_server(
                rospy.Duration(self.move_base_start_timeout)):
            raise RuntimeError("room move_base action server unavailable")
        rospy.wait_for_service("/move_base/make_plan", timeout=8.0)
        rospy.wait_for_service("/move_base/clear_costmaps", timeout=8.0)
        self.make_plan = rospy.ServiceProxy("/move_base/make_plan", GetPlan)
        self.clear_costmaps = rospy.ServiceProxy(
            "/move_base/clear_costmaps", Empty)
        self.cone_control_pub.publish(Bool(data=True))
        rospy.sleep(0.7)
        self.publish_state("ROOM_NAVIGATION_READY")

    @staticmethod
    def pose_message(x, y, yaw):
        message = PoseStamped()
        message.header.frame_id = "map"
        message.header.stamp = rospy.Time.now()
        message.pose.position.x = x
        message.pose.position.y = y
        quaternion = quaternion_from_euler(0.0, 0.0, yaw)
        message.pose.orientation.x = quaternion[0]
        message.pose.orientation.y = quaternion[1]
        message.pose.orientation.z = quaternion[2]
        message.pose.orientation.w = quaternion[3]
        return message

    def current_pose(self):
        with self.lock:
            return None if self.pose is None else tuple(self.pose)

    def make_plan_exists(self, x, y, yaw):
        pose = self.current_pose()
        if pose is None or self.make_plan is None:
            return False
        try:
            start = self.pose_message(pose[0], pose[1], pose[2])
            goal = self.pose_message(x, y, yaw)
            response = self.make_plan(start=start, goal=goal, tolerance=0.06)
            return len(response.plan.poses) >= 2
        except Exception as exc:
            rospy.logwarn_throttle(2.0, "make_plan check failed: %s", exc)
            return False

    def nearest_reachable(self, x, y, yaw):
        candidates = [(x, y)]
        for radius, samples in ((0.12, 4), (0.24, 8), (0.36, 8)):
            for index in range(samples):
                angle = 2.0 * math.pi * index / samples
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                if (self.room_min_x + 0.12 <= px <= self.room_max_x - 0.12 and
                        self.room_min_y + 0.12 <= py <= self.room_max_y - 0.12):
                    candidates.append((px, py))
        for px, py in candidates:
            if self.make_plan_exists(px, py, yaw):
                if math.hypot(px - x, py - y) > 0.02:
                    rospy.logwarn(
                        "ROOM_NEAREST_REACHABLE requested=(%.2f,%.2f) selected=(%.2f,%.2f)",
                        x, y, px, py)
                return px, py, yaw
        return None

    def send_goal(self, x, y, yaw, name, timeout, watch_target=True,
                  success_y_below=None):
        goal = MoveBaseGoal()
        goal.target_pose = self.pose_message(x, y, yaw)
        self.move_base.send_goal(goal)
        self.publish_state("NAVIGATING", goal=name, x=x, y=y, yaw=yaw)
        started = time.monotonic()
        last_progress = started
        last_pose = self.current_pose()
        rate = rospy.Rate(12)
        while not rospy.is_shutdown():
            if watch_target and self.target_event.is_set():
                self.move_base.cancel_goal()
                self.stop_robot(6)
                return "TARGET"
            if watch_target and self.candidate_event.is_set():
                self.move_base.cancel_goal()
                self.stop_robot(8)
                self.publish_state(
                    "OCR_TARGET_CANDIDATE_HOLD", goal=name,
                    hold_s=self.candidate_hold_s)
                hold_deadline = time.monotonic() + self.candidate_hold_s
                while (not rospy.is_shutdown() and
                       time.monotonic() < hold_deadline):
                    if self.target_event.is_set():
                        self.stop_robot(6)
                        return "TARGET"
                    rospy.sleep(0.04)
                self.candidate_event.clear()
                with self.lock:
                    self.candidate_cooldown_until = (
                        time.monotonic() + self.candidate_cooldown_s)
                self.publish_state(
                    "OCR_TARGET_CANDIDATE_RESUME", goal=name)
                self.move_base.send_goal(goal)
                last_progress = time.monotonic()
                last_pose = self.current_pose()
                rate.sleep()
                continue
            state = self.move_base.get_state()
            if state == GoalStatus.SUCCEEDED:
                return "SUCCEEDED"
            if state not in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                return "FAILED_{}".format(state)
            now = time.monotonic()
            pose = self.current_pose()
            if (success_y_below is not None and pose is not None and
                    pose[1] <= success_y_below):
                self.move_base.cancel_goal()
                self.stop_robot(8)
                return "SUCCEEDED_THRESHOLD"
            if pose is not None and last_pose is not None:
                moved = math.hypot(pose[0] - last_pose[0], pose[1] - last_pose[1])
                turned = abs(norm_angle(pose[2] - last_pose[2]))
                if moved >= self.progress_distance or turned >= self.progress_yaw:
                    last_progress = now
                    last_pose = pose
                remaining = (math.hypot(x - pose[0], y - pose[1]) +
                             0.20 * abs(norm_angle(yaw - pose[2])))
                if remaining > 0.12 and now - last_progress >= self.stuck_timeout:
                    self.move_base.cancel_goal()
                    self.stop_robot(6)
                    self.publish_state("GOAL_STUCK_REPLAN", goal=name,
                                       remaining=remaining)
                    return "STUCK"
            if now - started >= timeout:
                self.move_base.cancel_goal()
                self.stop_robot(6)
                return "TIMEOUT"
            rate.sleep()
        return "SHUTDOWN"

    def clear_and_wait(self, reason):
        try:
            self.clear_costmaps()
            rospy.logwarn("ROOM_COSTMAPS_CLEARED reason=%s", reason)
        except Exception as exc:
            rospy.logwarn("clear costmaps failed: %s", exc)
        rospy.sleep(0.6)

    @staticmethod
    def median(values):
        ordered = sorted(values)
        count = len(ordered)
        if count == 0:
            return None
        middle = count // 2
        if count % 2:
            return ordered[middle]
        return 0.5 * (ordered[middle - 1] + ordered[middle])

    def scan_snapshot(self):
        with self.lock:
            return list(self.scan_samples), self.scan_stamp

    def sector_clearance(self, center_angle, half_angle):
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        distances = [
            distance for angle, distance, _, _ in samples
            if abs(norm_angle(angle - center_angle)) <= half_angle
        ]
        return min(distances) if distances else None

    def rear_footprint_clearance(self):
        """Closest scan return to the rectangular rear half and its corners."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        closest = None
        for _, _, x_base, y_base in samples:
            if x_base >= 0.0:
                continue
            dx = max(abs(x_base) - self.robot_half_length, 0.0)
            dy = max(abs(y_base) - self.robot_half_width, 0.0)
            gap = math.hypot(dx, dy)
            if closest is None or gap < closest[0]:
                closest = (gap, x_base, y_base)
        return closest

    def side_footprint_clearance(self, vy, front_wall_distance=None):
        """Closest point to the full moving side of the rectangular chassis."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        closest = None
        for _, _, x_base, y_base in samples:
            if (vy > 0.0 and y_base <= 0.0) or (vy < 0.0 and y_base >= 0.0):
                continue
            if front_wall_distance is not None:
                x_laser, _ = self.base_to_laser(x_base, y_base)
                if abs(x_laser - front_wall_distance) <= 0.05:
                    continue
            dx = max(abs(x_base) - self.robot_half_length, 0.0)
            dy = max(abs(y_base) - self.robot_half_width, 0.0)
            gap = math.hypot(dx, dy)
            if closest is None or gap < closest[0]:
                closest = (gap, x_base, y_base)
        return closest

    def predicted_footprint_clearance(self, vx, vy, wz,
                                      front_wall_distance=None):
        """Closest scan point to the rectangular footprint over a short motion."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        horizon = max(0.10, self.motion_prediction_horizon)
        steps = max(2, self.motion_prediction_steps)
        margin = max(0.0, self.motion_prediction_margin)
        reach_x = (self.robot_half_length + margin +
                   abs(vx) * horizon + self.rotation_radius *
                   abs(wz) * horizon)
        reach_y = (self.robot_half_width + margin +
                   abs(vy) * horizon + self.rotation_radius *
                   abs(wz) * horizon)
        closest = None
        for _, _, x_base, y_base in samples:
            if front_wall_distance is not None:
                x_laser, _ = self.base_to_laser(x_base, y_base)
                if abs(x_laser - front_wall_distance) <= 0.055:
                    continue
            if abs(x_base) > reach_x + 0.08 or abs(y_base) > reach_y + 0.08:
                continue
            for step in range(1, steps + 1):
                elapsed = horizon * float(step) / float(steps)
                translated_x = x_base - vx * elapsed
                translated_y = y_base - vy * elapsed
                angle = -wz * elapsed
                cosine = math.cos(angle)
                sine = math.sin(angle)
                future_x = cosine * translated_x - sine * translated_y
                future_y = sine * translated_x + cosine * translated_y
                dx = max(abs(future_x) -
                         (self.robot_half_length + margin), 0.0)
                dy = max(abs(future_y) -
                         (self.robot_half_width + margin), 0.0)
                gap = math.hypot(dx, dy)
                if closest is None or gap < closest[0]:
                    closest = (gap, elapsed, x_base, y_base)
        return closest

    def predictive_motion_guard(self, vx, vy, wz, wall_distance, context):
        """Keep the largest safe lateral command before a cone reaches the body."""
        threat = self.predicted_footprint_clearance(
            vx, vy, wz, front_wall_distance=wall_distance)
        if threat is None or threat[0] > 1.0e-6:
            return vx, vy, wz

        # First remove longitudinal approach.  This is especially important in
        # parking, where forward motion and individually-safe lateral motion can
        # combine into a front-corner collision.
        stopped_x = self.predicted_footprint_clearance(
            0.0, vy, 0.0, front_wall_distance=wall_distance)
        if stopped_x is None or stopped_x[0] > 1.0e-6:
            rospy.logwarn_throttle(
                0.35, "%s_PREDICTIVE_GUARD stop_vx threat=(%.3f,%.3f) "
                "time=%.2f cmd=(%.3f,%.3f,%.3f)->(0.000,%.3f,0.000)",
                context, threat[2], threat[3], threat[1], vx, vy, wz, vy)
            return 0.0, vy, 0.0

        if abs(vy) <= 1.0e-4:
            return 0.0, 0.0, 0.0

        # Move longitudinally away from the threatening front/rear corner and
        # retain the largest lateral fraction that remains collision-free.
        escape_vx = (-self.predictive_escape_speed if threat[2] >= 0.0 else
                     self.predictive_escape_speed)
        if escape_vx < 0.0:
            escape_vx = self.guard_reverse_velocity(
                escape_vx, "{}_PREDICTIVE".format(context))
        else:
            front = self.sector_clearance(0.0, math.radians(58.0))
            if front is None or front < self.detour_hard_clearance:
                escape_vx = 0.0

        low = 0.0
        high = 1.0
        safe_scale = 0.0
        for _ in range(5):
            scale = 0.5 * (low + high)
            candidate = self.predicted_footprint_clearance(
                escape_vx, vy * scale, 0.0,
                front_wall_distance=wall_distance)
            if candidate is None or candidate[0] > 1.0e-6:
                safe_scale = scale
                low = scale
            else:
                high = scale
        guarded_vy = vy * safe_scale
        rospy.logwarn_throttle(
            0.35, "%s_PREDICTIVE_GLIDE threat=(%.3f,%.3f) time=%.2f "
            "scale=%.2f cmd=(%.3f,%.3f,%.3f)->(%.3f,%.3f,0.000)",
            context, threat[2], threat[3], threat[1], safe_scale,
            vx, vy, wz, escape_vx, guarded_vy)
        return escape_vx, guarded_vy, 0.0

    def side_escape_velocity(self, threat, context):
        if threat is None or abs(threat[1]) < 0.03:
            return 0.0
        if threat[1] < 0.0:
            # Obstacle is beside the rear half: move the chassis forward if
            # the complete front hemisphere is open.
            front = self.sector_clearance(0.0, math.radians(58.0))
            if front is not None and front >= self.detour_hard_clearance:
                rospy.logwarn_throttle(
                    0.4, "%s_SIDE_ESCAPE_FORWARD gap=%.3f point=(%.3f,%.3f)",
                    context, threat[0], threat[1], threat[2])
                return self.side_escape_speed
            return 0.0
        # Obstacle is beside the front half: retreat only when the whole rear
        # hemisphere and rectangular rear corners agree that it is safe.
        rear = self.sector_clearance(math.pi, math.radians(58.0))
        if rear is None or rear < self.detour_hard_clearance:
            return 0.0
        guarded = self.guard_reverse_velocity(-self.side_escape_speed, context)
        if guarded < -1.0e-4:
            rospy.logwarn_throttle(
                0.4, "%s_SIDE_ESCAPE_REVERSE gap=%.3f point=(%.3f,%.3f) "
                "vx=%.3f", context, threat[0], threat[1], threat[2], guarded)
        return guarded

    def side_cone_corridor(self, vy, front_wall_distance=None):
        """Find a compact front/rear cone pair with room for sideways passage."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        values = []
        max_side_gap = self.cone_side_slow
        for _, _, x_base, y_base in samples:
            if (vy > 0.0 and y_base <= 0.0) or (vy < 0.0 and y_base >= 0.0):
                continue
            if abs(x_base) > 0.80:
                continue
            if max(abs(y_base) - self.robot_half_width, 0.0) > max_side_gap:
                continue
            if front_wall_distance is not None:
                x_laser, _ = self.base_to_laser(x_base, y_base)
                if abs(x_laser - front_wall_distance) <= 0.06:
                    continue
            values.append(x_base)
        if len(values) < 4:
            return None
        values.sort()
        clusters = []
        current = [values[0]]
        for value in values[1:]:
            if value - current[-1] <= self.side_corridor_cluster_gap:
                current.append(value)
            else:
                clusters.append(current)
                current = [value]
        clusters.append(current)
        compact = [
            (cluster[0], cluster[-1], len(cluster)) for cluster in clusters
            if len(cluster) >= 2 and
            cluster[-1] - cluster[0] <= self.side_corridor_max_cluster_span
        ]
        rear = [cluster for cluster in compact if cluster[1] < -0.02]
        front = [cluster for cluster in compact if cluster[0] > 0.02]
        if not rear or not front:
            return None
        rear_cluster = max(rear, key=lambda cluster: cluster[1])
        front_cluster = min(front, key=lambda cluster: cluster[0])
        free_gap = front_cluster[0] - rear_cluster[1]
        required_gap = (2.0 * self.robot_half_length +
                        2.0 * self.side_corridor_margin)
        if free_gap < required_gap:
            return None
        lower = (rear_cluster[1] + self.robot_half_length +
                 self.side_corridor_margin)
        upper = (front_cluster[0] - self.robot_half_length -
                 self.side_corridor_margin)
        center_offset = 0.5 * (lower + upper)
        return {
            "rear": rear_cluster,
            "front": front_cluster,
            "free_gap": free_gap,
            "required_gap": required_gap,
            "center_offset": center_offset,
            "lower": lower,
            "upper": upper,
        }

    def guard_reverse_velocity(self, vx, context):
        if vx >= -1.0e-4:
            return vx
        rear = self.rear_footprint_clearance()
        if rear is None:
            rospy.logwarn_throttle(
                0.5, "%s_REVERSE_BLOCKED rear_scan=stale", context)
            return 0.0
        ratio = clamp(
            (rear[0] - self.reverse_hard_gap) /
            max(0.02, self.reverse_slow_gap - self.reverse_hard_gap),
            0.0, 1.0)
        guarded = vx * ratio
        if ratio < 0.999:
            rospy.logwarn_throttle(
                0.4, "%s_REVERSE_GUARD gap=%.3f point=(%.3f,%.3f) "
                "vx=%.3f->%.3f", context, rear[0], rear[1], rear[2],
                vx, guarded)
        return guarded

    def side_obstacle_span(self, distance_limit):
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return 0.0
        angles = [
            angle for angle, distance, _, _ in samples
            if abs(norm_angle(angle + math.pi / 2.0)) <= math.radians(38.0)
            and distance <= distance_limit
        ]
        if len(angles) < 2:
            return 0.0
        return max(angles) - min(angles)

    def side_obstacle_is_wall(self, distance_limit):
        """Distinguish a long flat room boundary from a compact cone return."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return False, 0.0, 0
        points = [
            (x_base, y_base) for angle, distance, x_base, y_base in samples
            if abs(norm_angle(angle + math.pi / 2.0)) <= math.radians(42.0)
            and distance <= distance_limit
        ]
        if len(points) < 12:
            return False, 0.0, len(points)
        median_y = self.median([point[1] for point in points])
        inliers = [point for point in points
                   if abs(point[1] - median_y) <= 0.035]
        if len(inliers) < 12:
            return False, 0.0, len(inliers)
        linear_span = max(point[0] for point in inliers) - min(
            point[0] for point in inliers)
        return linear_span >= 0.20, linear_span, len(inliers)

    def front_wall_estimate(self):
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        candidates = [
            (x_base, y_base) for angle, distance, x_base, y_base in samples
            if abs(angle) <= self.wall_fit_half_angle
            and 0.15 <= x_base <= self.wall_fit_max_range
            and distance <= self.wall_fit_max_range
        ]
        if len(candidates) < self.wall_fit_min_points:
            return None
        median_x = self.median([value[0] for value in candidates])
        inliers = [
            value for value in candidates
            if abs(value[0] - median_x) <= self.wall_fit_inlier
        ]
        if len(inliers) < self.wall_fit_min_points:
            return None
        mean_y = sum(value[1] for value in inliers) / len(inliers)
        mean_x = sum(value[0] for value in inliers) / len(inliers)
        denominator = sum((value[1] - mean_y) ** 2 for value in inliers)
        if denominator < 1.0e-5:
            return None
        slope = sum(
            (value[1] - mean_y) * (value[0] - mean_x)
            for value in inliers) / denominator
        intercept = mean_x - slope * mean_y
        if not (0.12 <= intercept <= self.wall_fit_max_range):
            return None
        return intercept, math.atan(slope), len(inliers)

    def acquire_wall_target(self, segment_name):
        deadline = time.monotonic() + 1.2
        distances = []
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            estimate = self.front_wall_estimate()
            if estimate is not None:
                distances.append(estimate[0])
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.05)
        measured = self.median(distances)
        if measured is None:
            target = self.wall_target_default
            source = "default"
        else:
            target = clamp(
                measured, self.wall_target_min, self.wall_target_max)
            source = "lidar"
        self.publish_state(
            "WALL_TARGET_ACQUIRED", segment=segment_name,
            target_distance=target, measured_distance=measured,
            source=source)
        return target

    def motion_interrupted_by_ocr(self, motion_name):
        if not self.ensure_ocr_running():
            self.stop_robot(8)
            return "OCR_UNAVAILABLE"
        if self.target_event.is_set():
            self.stop_robot(8)
            return "TARGET"
        if self.candidate_event.is_set():
            if self.hold_ocr_candidate(motion_name):
                return "TARGET"
        return None

    def drive_body_distance(self, name, vx, vy, distance, target_yaw,
                            direction_angle, hard_clearance):
        start_pose = self.current_pose()
        if start_pose is None:
            return "NO_POSE"
        speed = max(0.02, math.hypot(vx, vy))
        deadline = time.monotonic() + max(2.0, distance / speed * 2.8 + 0.8)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            interruption_started = time.monotonic()
            interrupted = self.motion_interrupted_by_ocr(name)
            interruption_elapsed = time.monotonic() - interruption_started
            if interruption_elapsed > 0.10:
                # OCR inspection is intentional stopped time, not failed motion.
                # Without this compensation a 1.8 s candidate hold consumed
                # almost the complete short corner-relocation deadline.
                deadline += interruption_elapsed
            if interrupted is not None:
                return interrupted
            pose = self.current_pose()
            if pose is None:
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue
            moved = math.hypot(pose[0] - start_pose[0], pose[1] - start_pose[1])
            if moved >= distance:
                self.smooth_stop_robot()
                return "SUCCEEDED"
            clearance = self.sector_clearance(direction_angle, math.radians(42.0))
            if clearance is None:
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue
            if clearance < hard_clearance:
                self.stop_robot(8)
                self.publish_state(
                    "WALL_DIRECT_MOTION_BLOCKED", motion=name,
                    clearance=clearance, moved=moved)
                return "BLOCKED"
            yaw_error = norm_angle(target_yaw - pose[2])
            command_wz = clamp(
                self.wall_nominal_yaw_kp * yaw_error,
                -self.wall_route_max_wz, self.wall_route_max_wz)
            self.publish_direct_command(vx, vy, command_wz)
            rate.sleep()
        self.stop_robot(8)
        return "TIMEOUT"

    def make_rotation_space(self, target_yaw):
        samples, stamp = self.scan_snapshot()
        if (not samples or
                time.monotonic() - stamp > self.sweep_sensor_fresh_s):
            return False
        nearest = min(samples, key=lambda value: value[1])
        obstacle_angle = nearest[0]
        escape_angle = norm_angle(obstacle_angle + math.pi)
        speed = min(0.075, self.detour_speed)
        with self.lock:
            current_clearance = self.rotation_clearance
        escape_distance = clamp(
            self.corner_rotation_clearance - current_clearance + 0.03,
            0.05, 0.16)
        result = self.drive_body_distance(
            "corner_rotation_space", speed * math.cos(escape_angle),
            speed * math.sin(escape_angle), escape_distance, target_yaw,
            escape_angle, self.detour_hard_clearance)
        return result in ("SUCCEEDED", "TARGET")

    def turn_to_wall(self, name, target_yaw, watch_ocr=True):
        self.stop_robot(8)
        self.publish_state("WALL_CORNER_TURN", corner=name,
                           target_yaw=target_yaw, feedback="odom")
        started = time.monotonic()
        relocation_attempted = False
        target_odom_yaw = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() - started < 12.0:
            if watch_ocr:
                interrupted = self.motion_interrupted_by_ocr(
                    "{}_corner_turn".format(name))
                if interrupted is not None:
                    return interrupted
            pose = self.current_pose()
            with self.lock:
                clearance = self.rotation_clearance
                scan_stamp = self.scan_stamp
                odom_yaw = self.odom_yaw
                odom_stamp = self.odom_stamp
            now = time.monotonic()
            if (pose is None or odom_yaw is None or
                    now - scan_stamp > self.sweep_sensor_fresh_s or
                    now - odom_stamp > self.sweep_sensor_fresh_s):
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue
            if target_odom_yaw is None:
                target_odom_yaw = norm_angle(
                    odom_yaw + norm_angle(target_yaw - pose[2]))
            error = norm_angle(target_odom_yaw - odom_yaw)
            if abs(error) <= self.corner_turn_tolerance:
                self.smooth_stop_robot()
                self.publish_state("WALL_CORNER_TURN_COMPLETE", corner=name,
                                   yaw_error=error,
                                   map_yaw_error=norm_angle(
                                       target_yaw - pose[2]))
                return "SUCCEEDED"
            if clearance < self.corner_rotation_clearance:
                self.stop_robot(8)
                if relocation_attempted or not watch_ocr:
                    self.publish_state(
                        "WALL_CORNER_TURN_BLOCKED", corner=name,
                        clearance=clearance)
                    return "BLOCKED"
                relocation_attempted = True
                if not self.make_rotation_space(target_yaw):
                    return "BLOCKED"
                started = time.monotonic()
                target_odom_yaw = None
                continue
            speed = clamp(
                0.85 * abs(error), self.corner_turn_min_wz,
                self.corner_turn_max_wz)
            command = self.publish_direct_command(
                wz=math.copysign(speed, error))
            rospy.logwarn_throttle(
                0.4,
                "WALL_CORNER_CMD corner=%s odom_error=%.1fdeg map_error=%.1fdeg "
                "wz=%.3f clear=%.3f",
                name, math.degrees(error),
                math.degrees(norm_angle(target_yaw - pose[2])),
                command.angular.z, clearance)
            rate.sleep()
        self.stop_robot(10)
        return "TIMEOUT"

    def perform_cone_detour(self, segment_name, target_yaw, wall_target):
        self.stop_robot(8)
        side_clearance = self.sector_clearance(
            -math.pi / 2.0, math.radians(34.0))
        if side_clearance is None:
            away = self.detour_away_distance
        else:
            # Back away only far enough to recover useful side clearance.  The
            # old fixed 24 cm retreat wasted time even for a cone barely inside
            # the trigger band.
            required = self.cone_side_stop - side_clearance + 0.04
            away = clamp(required, self.detour_min_away,
                         self.detour_away_distance)
        self.publish_state(
            "WALL_CONE_DETOUR_START", segment=segment_name,
            side_clearance=side_clearance, away_distance=away,
            max_away_distance=self.detour_max_away,
            pass_distance=self.detour_pass_distance)
        result = self.drive_body_distance(
            "{}_detour_away".format(segment_name), -self.detour_speed, 0.0,
            away, target_yaw, math.pi, self.detour_hard_clearance)
        if result != "SUCCEEDED":
            return result

        result = self.drive_body_distance(
            "{}_detour_pass".format(segment_name), 0.0, -self.detour_speed,
            self.detour_pass_distance, target_yaw, -math.pi / 2.0,
            self.detour_hard_clearance)
        if result == "BLOCKED" and away < self.detour_max_away:
            extra = self.detour_max_away - away
            result = self.drive_body_distance(
                "{}_detour_extra_away".format(segment_name),
                -self.detour_speed, 0.0, extra, target_yaw, math.pi,
                self.detour_hard_clearance)
            if result == "SUCCEEDED":
                result = self.drive_body_distance(
                    "{}_detour_pass_retry".format(segment_name),
                    0.0, -self.detour_speed, self.detour_pass_distance,
                    target_yaw, -math.pi / 2.0,
                    self.detour_hard_clearance)
                away = self.detour_max_away
        if result != "SUCCEEDED":
            return result

        return_started = time.monotonic()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() - return_started < 5.0:
            interrupted = self.motion_interrupted_by_ocr(
                "{}_detour_return".format(segment_name))
            if interrupted is not None:
                return interrupted
            estimate = self.front_wall_estimate()
            pose = self.current_pose()
            if estimate is not None and estimate[0] <= wall_target + 0.05:
                self.stop_robot(8)
                self.publish_state("WALL_CONE_DETOUR_COMPLETE",
                                   segment=segment_name)
                return "SUCCEEDED"
            if pose is None:
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue
            front_clearance = self.sector_clearance(0.0, math.radians(30.0))
            if (front_clearance is None or
                    front_clearance < self.detour_hard_clearance):
                self.stop_robot(8)
                return "BLOCKED"
            command_wz = clamp(
                self.wall_nominal_yaw_kp * norm_angle(target_yaw - pose[2]),
                -self.wall_route_max_wz, self.wall_route_max_wz)
            self.publish_direct_command(
                vx=min(self.detour_speed, 0.085), wz=command_wz)
            rate.sleep()
        self.stop_robot(8)
        # A missing wall estimate must not trap the mission.  Returning to the
        # approximate pre-detour offset is a bounded fallback.
        result = self.drive_body_distance(
            "{}_detour_return_fallback".format(segment_name),
            min(self.detour_speed, 0.075), 0.0, away, target_yaw, 0.0,
            self.detour_hard_clearance)
        return result

    def wall_segment_fallback(self, name, direction_x, direction_y,
                              remaining, target_yaw):
        pose = self.current_pose()
        if pose is None:
            return "NO_POSE"
        fallback_distance = clamp(remaining, 0.18, 0.65)
        requested_x = pose[0] + direction_x * fallback_distance
        requested_y = pose[1] + direction_y * fallback_distance
        selected = self.nearest_reachable(requested_x, requested_y, target_yaw)
        if selected is None:
            return "UNREACHABLE"
        self.publish_state(
            "WALL_SEGMENT_LOCAL_REPLAN", segment=name,
            remaining=remaining, goal_x=selected[0], goal_y=selected[1])
        return self.send_goal(
            selected[0], selected[1], selected[2],
            "{}_local_replan".format(name), min(18.0, self.goal_timeout))

    def follow_wall_segment(self, index, resume_from_current=False):
        start_name, start_x, start_y = self.wall_route_points[index]
        end_name, end_x, end_y = self.wall_route_points[index + 1]
        target_yaw = self.wall_route_yaws[index]
        segment_name = "{}_to_{}".format(start_name, end_name)
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.hypot(dx, dy)
        direction_x = dx / max(1.0e-6, length)
        direction_y = dy / max(1.0e-6, length)

        reference_length = length
        if resume_from_current:
            resume_pose = self.current_pose()
            if resume_pose is None:
                return "NO_POSE"
            nominal_progress = (
                (resume_pose[0] - start_x) * direction_x +
                (resume_pose[1] - start_y) * direction_y)
            nominal_progress = clamp(nominal_progress, 0.0, reference_length)
            length = max(0.0, reference_length - nominal_progress)
            if length <= self.segment_end_tolerance:
                self.publish_state(
                    "WALL_SEGMENT_RESUME_ALREADY_COMPLETE",
                    segment=segment_name, nominal_progress=nominal_progress,
                    reference_length=reference_length)
                return "SUCCEEDED"
            self.publish_state(
                "WALL_SEGMENT_RESUME_FROM_CURRENT", segment=segment_name,
                nominal_progress=nominal_progress,
                remaining_length=length)

        turn_result = self.turn_to_wall(start_name, target_yaw)
        if turn_result != "SUCCEEDED":
            return (turn_result if turn_result == "TARGET" else
                    "TURN_{}".format(turn_result))
        wall_target = self.acquire_wall_target(segment_name)
        actual_start = self.current_pose()
        if actual_start is None:
            return "NO_POSE"
        deadline = time.monotonic() + max(
            12.0, length / max(0.03, self.wall_route_speed) *
            self.segment_timeout_scale)
        wall_lost_since = None
        side_body_count = 0
        avoid_state = "DIRECT"
        avoid_clear_scans = 0
        avoid_axial_sign = -1.0
        avoid_blocked_since = None
        pass_start_pose = None
        pass_start_odom = None
        rate = rospy.Rate(20)
        self.publish_state(
            "WALL_SEGMENT_START", segment=segment_name,
            reference_length=length, target_yaw=target_yaw,
            wall_target=wall_target, resumed=resume_from_current)

        while not rospy.is_shutdown() and time.monotonic() < deadline:
            interrupted = self.motion_interrupted_by_ocr(segment_name)
            if interrupted is not None:
                return interrupted
            pose = self.current_pose()
            if pose is None:
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue
            progress = ((pose[0] - actual_start[0]) * direction_x +
                        (pose[1] - actual_start[1]) * direction_y)
            remaining = max(0.0, length - progress)
            if remaining <= self.segment_end_tolerance:
                self.smooth_stop_robot()
                self.publish_state(
                    "WALL_SEGMENT_COMPLETE", segment=segment_name,
                    progress=progress, reason="reference_length")
                return "SUCCEEDED"

            side_clearance = self.sector_clearance(
                -math.pi / 2.0, math.radians(48.0))
            if side_clearance is None:
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue
            side_span = self.side_obstacle_span(
                self.corner_side_distance + 0.08)
            side_is_wall, side_linear_span, side_wall_points = (
                self.side_obstacle_is_wall(self.corner_side_distance + 0.08))
            if (side_clearance < self.cone_side_stop and
                    progress >= self.corner_early_ratio * length and
                    side_span >= self.corner_wall_span and side_is_wall):
                self.smooth_stop_robot()
                self.publish_state(
                    "WALL_SEGMENT_COMPLETE", segment=segment_name,
                    progress=progress, reason="physical_corner",
                    side_clearance=side_clearance, side_span=side_span,
                    side_linear_span=side_linear_span,
                    side_wall_points=side_wall_points)
                return "SUCCEEDED"

            estimate = self.front_wall_estimate()
            now = time.monotonic()
            if estimate is None:
                if wall_lost_since is None:
                    wall_lost_since = now
                if now - wall_lost_since >= self.wall_sensor_lost_s:
                    self.stop_robot(8)
                    fallback = self.wall_segment_fallback(
                        segment_name, direction_x, direction_y,
                        remaining, target_yaw)
                    if fallback == "TARGET":
                        return fallback
                    wall_lost_since = None
                    deadline = max(
                        deadline, time.monotonic() +
                        max(6.0, remaining / max(0.03, self.wall_route_speed)))
                    continue
                wall_distance = wall_target
                wall_heading_error = 0.0
            else:
                wall_lost_since = None
                wall_distance, wall_heading_error, _ = estimate

            side_body = self.side_footprint_clearance(
                -1.0, front_wall_distance=wall_distance)
            body_hard_now = (
                side_body is not None and
                side_body[0] <= self.side_body_hard_gap)
            side_body_count = side_body_count + 1 if body_hard_now else 0
            right_open = (
                side_body is None or
                side_body[0] > self.side_body_hard_gap)
            right_wide_open = (
                side_body is None or
                side_body[0] >= self.side_body_slow_gap)

            # Keep one simple manoeuvre active until the same cone has really
            # passed the chassis.  The former 0.9 s timer repeatedly switched
            # back to normal motion while the cone was still beside the front
            # corner, producing an endless forward/backward oscillation.
            if (avoid_state == "DIRECT" and body_hard_now and
                    side_body_count >= max(1, self.side_body_confirm_scans)):
                avoid_state = "BACK_CLEAR"
                # A cone beside the front half is cleared by backing up.  If it
                # is already beside the rear half, moving forward avoids
                # backing the opposite corner into it.
                avoid_axial_sign = -1.0 if side_body[1] >= 0.0 else 1.0
                avoid_blocked_since = time.monotonic()
                avoid_clear_scans = 0
                deadline += 10.0
                self.publish_state(
                    "WALL_SIMPLE_CONE_AVOID_START", segment=segment_name,
                    gap=side_body[0], point_x=side_body[1],
                    point_y=side_body[2], axial_sign=avoid_axial_sign)
            elif avoid_state == "BACK_CLEAR":
                avoid_clear_scans = (avoid_clear_scans + 1
                                     if right_wide_open else 0)
                if avoid_clear_scans >= 2:
                    avoid_state = "PASS_RIGHT"
                    avoid_clear_scans = 0
                    pass_start_pose = pose
                    with self.lock:
                        pass_start_odom = (
                            None if self.odom_pose is None else
                            tuple(self.odom_pose))
                    avoid_blocked_since = None
                    self.publish_state(
                        "WALL_SIMPLE_CONE_RIGHT_OPEN", segment=segment_name)
            elif avoid_state == "PASS_RIGHT":
                if body_hard_now:
                    # The right sweep closed again before the cone passed;
                    # back away a little more using live lidar feedback.
                    avoid_state = "BACK_CLEAR"
                    avoid_axial_sign = -1.0 if side_body[1] >= 0.0 else 1.0
                    avoid_blocked_since = time.monotonic()
                    avoid_clear_scans = 0
                    pass_start_pose = None
                    pass_start_odom = None
                else:
                    pass_progress = 0.0
                    with self.lock:
                        current_odom = (
                            None if self.odom_pose is None else
                            tuple(self.odom_pose))
                    if pass_start_odom is not None and current_odom is not None:
                        pass_progress = math.hypot(
                            current_odom[0] - pass_start_odom[0],
                            current_odom[1] - pass_start_odom[1])
                    elif pass_start_pose is not None:
                        pass_progress = (
                            (pose[0] - pass_start_pose[0]) * direction_x +
                            (pose[1] - pass_start_pose[1]) * direction_y)
                    avoid_clear_scans = (avoid_clear_scans + 1
                                         if right_wide_open else 0)
                    passed_by_scan = side_body is None
                    passed_by_motion = pass_progress >= max(
                        0.10, 0.85 * self.robot_half_width)
                    if (avoid_clear_scans >= 2 and
                            (passed_by_scan or passed_by_motion)):
                        avoid_state = "RETURN_WALL"
                        avoid_clear_scans = 0
                        self.publish_state(
                            "WALL_SIMPLE_CONE_PASSED", segment=segment_name,
                            pass_progress=pass_progress)
            elif avoid_state == "RETURN_WALL":
                # Do not spend several seconds converging exactly to the old
                # wall distance.  Normal wall following can close the final
                # small offset while continuing along the inspection route.
                if wall_distance <= (
                        wall_target + self.wall_return_release_margin):
                    avoid_state = "DIRECT"
                    side_body_count = 0
                    pass_start_pose = None
                    pass_start_odom = None
                    self.publish_state(
                        "WALL_SIMPLE_CONE_AVOID_COMPLETE",
                        segment=segment_name, wall_distance=wall_distance)

            lateral_speed = self.wall_route_speed
            if remaining < 0.28:
                lateral_speed *= clamp(remaining / 0.28, 0.45, 1.0)
            yaw_error = norm_angle(target_yaw - pose[2])
            command = Twist()
            avoid_ratio = clamp(
                (self.cone_side_slow - side_clearance) /
                max(0.02, self.cone_side_slow - self.cone_side_hard),
                0.0, 1.0)

            if avoid_state == "DIRECT":
                command.linear.x = clamp(
                    self.wall_distance_kp * (wall_distance - wall_target),
                    -self.wall_route_max_vx, self.wall_route_max_vx)
                command.linear.y = -lateral_speed if right_open else 0.0
            elif avoid_state == "BACK_CLEAR":
                if avoid_axial_sign < 0.0:
                    command.linear.x = self.guard_reverse_velocity(
                        -self.side_escape_speed, "WALL_SIMPLE_CONE")
                else:
                    front_clearance = self.sector_clearance(
                        0.0, math.radians(58.0))
                    if (front_clearance is not None and
                            front_clearance >= self.detour_hard_clearance):
                        command.linear.x = self.side_escape_speed
                if abs(command.linear.x) <= 1.0e-4:
                    if avoid_blocked_since is None:
                        avoid_blocked_since = time.monotonic()
                    elif time.monotonic() - avoid_blocked_since >= 0.7:
                        # The preferred axial direction is blocked.  Switch
                        # once to the other live-clearance direction instead
                        # of remaining stopped forever.
                        avoid_axial_sign *= -1.0
                        avoid_blocked_since = time.monotonic()
                        self.publish_state(
                            "WALL_SIMPLE_CONE_CLEAR_DIRECTION_SWITCH",
                            segment=segment_name,
                            axial_sign=avoid_axial_sign)
                else:
                    avoid_blocked_since = None
                command.linear.y = 0.0
            elif avoid_state == "PASS_RIGHT":
                command.linear.x = 0.0
                command.linear.y = -min(
                    self.wall_route_speed, self.side_corridor_vy)
            else:  # RETURN_WALL
                command.linear.x = clamp(
                    self.wall_distance_kp * (wall_distance - wall_target),
                    0.0, self.wall_route_max_vx)
                command.linear.y = -self.wall_route_min_speed

            if command.linear.x > 0.0:
                front_clearance = self.sector_clearance(
                    0.0, math.radians(32.0))
                if (front_clearance is None or
                        front_clearance < self.detour_hard_clearance):
                    command.linear.x = 0.0
            elif command.linear.x < 0.0:
                rear_clearance = self.sector_clearance(
                    math.pi, math.radians(32.0))
                if (rear_clearance is None or
                        rear_clearance < self.detour_hard_clearance):
                    command.linear.x = 0.0
                else:
                    command.linear.x = self.guard_reverse_velocity(
                        command.linear.x, "WALL_ROUTE")
            command.angular.z = clamp(
                self.wall_nominal_yaw_kp * yaw_error -
                self.wall_heading_kp * wall_heading_error,
                -self.wall_route_max_wz, self.wall_route_max_wz)
            if avoid_state == "DIRECT":
                (command.linear.x, command.linear.y, command.angular.z) = (
                    self.predictive_motion_guard(
                        command.linear.x, command.linear.y,
                        command.angular.z, wall_distance, "WALL_ROUTE"))
            command = self.publish_direct_command(
                command.linear.x, command.linear.y, command.angular.z)
            rospy.logwarn_throttle(
                0.5,
                "WALL_ROUTE segment=%s mode=%s progress=%.2f/%.2fm "
                "wall=%.2f/%.2fm side=%.2fm avoid=%.2f "
                "cmd=(%.2f,%.2f,%.2f)",
                segment_name, avoid_state, progress, length,
                wall_distance, wall_target,
                side_clearance, avoid_ratio, command.linear.x, command.linear.y,
                command.angular.z)
            rate.sleep()

        self.stop_robot(10)
        pose = self.current_pose()
        if pose is None:
            return "TIMEOUT"
        progress = ((pose[0] - actual_start[0]) * direction_x +
                    (pose[1] - actual_start[1]) * direction_y)
        remaining = max(0.0, length - progress)
        fallback = self.wall_segment_fallback(
            segment_name, direction_x, direction_y, remaining, target_yaw)
        return "TARGET" if fallback == "TARGET" else "SUCCEEDED_FALLBACK"

    def nearest_route_start(self, x, y, yaw):
        selected = self.nearest_reachable(x, y, yaw)
        if selected is not None:
            return selected
        for radius in (0.48, 0.60, 0.72):
            for index in range(16):
                angle = 2.0 * math.pi * index / 16.0
                px = x + radius * math.cos(angle)
                py = y + radius * math.sin(angle)
                if (self.room_min_x + 0.12 <= px <= self.room_max_x - 0.12 and
                        self.room_min_y + 0.12 <= py <= self.room_max_y - 0.12 and
                        self.make_plan_exists(px, py, yaw)):
                    self.publish_state(
                        "WALL_ROUTE_START_NEARBY", requested_x=x,
                        requested_y=y, selected_x=px, selected_y=py)
                    return px, py, yaw
        return None

    def run_wall_route(self, start_index=0, approach_start=True):
        segment_count = len(self.wall_route_points) - 1
        start_index = max(0, min(int(start_index), segment_count))
        if start_index >= segment_count:
            return "SUCCEEDED"

        if approach_start:
            start_name, start_x, start_y = self.wall_route_points[start_index]
            start_yaw = self.wall_route_yaws[start_index]
            selected = self.nearest_route_start(start_x, start_y, start_yaw)
            if selected is None:
                self.publish_state("WALL_ROUTE_START_UNREACHABLE")
                return "UNREACHABLE"
            result = self.send_goal(
                selected[0], selected[1], selected[2],
                "{}_approach".format(start_name), self.goal_timeout)
            if result == "TARGET":
                return result
            if result != "SUCCEEDED":
                self.clear_and_wait("wall route start {}".format(result))
                replacement = self.nearest_route_start(
                    start_x, start_y, start_yaw)
                if replacement is not None:
                    result = self.send_goal(
                        replacement[0], replacement[1], replacement[2],
                        "{}_approach_retry".format(start_name),
                        self.goal_timeout)
            if result == "TARGET":
                return result
            if result != "SUCCEEDED":
                # Being near the intended corridor is enough; direct wall
                # fitting can recover from a loose move_base endpoint.
                pose = self.current_pose()
                if pose is None or math.hypot(
                        pose[0] - start_x, pose[1] - start_y) > 0.75:
                    return result
                self.publish_state(
                    "WALL_ROUTE_START_ACCEPTED_LOOSE", result=result,
                    pose_x=pose[0], pose_y=pose[1])

        for index in range(start_index, segment_count):
            self.active_wall_segment_index = index
            result = self.follow_wall_segment(
                index, resume_from_current=(not approach_start and
                                            index == start_index))
            if result == "TARGET":
                return result
            if result.startswith("TURN_"):
                # A corner clearance adjustment is local.  Do not send a
                # global move_base goal back to an approximate map corner;
                # that was the apparent "escape" at a1.  Retry from the live
                # lidar pose after OCR candidate handling has released.
                self.publish_state(
                    "WALL_CORNER_LOCAL_RETRY", segment_index=index,
                    result=result)
                rospy.sleep(0.20)
                result = self.follow_wall_segment(
                    index, resume_from_current=(not approach_start and
                                                index == start_index))
                if result == "TARGET":
                    return result
                if result in ("SUCCEEDED", "SUCCEEDED_FALLBACK"):
                    continue
            if result not in ("SUCCEEDED", "SUCCEEDED_FALLBACK"):
                self.publish_state(
                    "WALL_SEGMENT_RECOVERY_START",
                    segment_index=index, result=result)
                start_name, start_x, start_y = self.wall_route_points[index]
                target_yaw = self.wall_route_yaws[index]
                recovery_name = "{}_segment_recovery".format(start_name)
                if result.startswith("TURN_"):
                    # The route coordinates are only hints.  Recover rotation
                    # around the live pose instead of driving back toward the
                    # nominal corner coordinate.
                    pose = self.current_pose()
                    if pose is not None:
                        start_x, start_y = pose[0], pose[1]
                        recovery_name = "{}_corner_pose_recovery".format(
                            start_name)
                replacement = self.nearest_route_start(
                    start_x, start_y, target_yaw)
                if replacement is not None:
                    recovery = self.send_goal(
                        replacement[0], replacement[1], replacement[2],
                        recovery_name,
                        min(20.0, self.goal_timeout))
                    if recovery == "TARGET":
                        return recovery
                result = self.follow_wall_segment(index)
                if result == "TARGET":
                    return result
                if result not in ("SUCCEEDED", "SUCCEEDED_FALLBACK"):
                    self.publish_state(
                        "WALL_SEGMENT_UNAVAILABLE_AFTER_RETRY",
                        segment_index=index, result=result)
        return "SUCCEEDED"

    def wall_route_resume_index(self):
        segment_count = len(self.wall_route_points) - 1
        index = max(0, min(self.active_wall_segment_index,
                           segment_count - 1))
        pose = self.current_pose()
        if pose is None:
            return index
        _, start_x, start_y = self.wall_route_points[index]
        _, end_x, end_y = self.wall_route_points[index + 1]
        dx = end_x - start_x
        dy = end_y - start_y
        length = math.hypot(dx, dy)
        if length <= 1.0e-6:
            return min(index + 1, segment_count)
        progress = ((pose[0] - start_x) * dx +
                    (pose[1] - start_y) * dy) / length
        if progress >= length - max(0.12, self.segment_end_tolerance):
            return min(index + 1, segment_count)
        return index

    def hold_ocr_candidate(self, goal_name):
        self.stop_robot(8)
        self.publish_state(
            "OCR_TARGET_CANDIDATE_HOLD", goal=goal_name,
            hold_s=self.candidate_hold_s)
        deadline = time.monotonic() + self.candidate_hold_s
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            if self.target_event.is_set():
                return True
            if not self.candidate_event.is_set():
                self.publish_state(
                    "OCR_TARGET_CANDIDATE_RELEASED_EARLY", goal=goal_name)
                return False
            rospy.sleep(0.04)
        self.candidate_event.clear()
        with self.lock:
            self.candidate_cooldown_until = (
                time.monotonic() + self.candidate_cooldown_s)
        self.publish_state("OCR_TARGET_CANDIDATE_RESUME", goal=goal_name)
        return self.target_event.is_set()

    @staticmethod
    def directed_sweep_angle(start_yaw, end_yaw, direction):
        if direction > 0.0:
            return (end_yaw - start_yaw) % (2.0 * math.pi)
        return (start_yaw - end_yaw) % (2.0 * math.pi)

    def controlled_ocr_sweep(self, name, start_yaw, end_yaw):
        goal_name = "{}_sweep".format(name)
        self.stop_robot(10)
        total_angle = self.directed_sweep_angle(
            start_yaw, end_yaw, self.sweep_direction)
        self.publish_state(
            "CONTINUOUS_OCR_SWEEP", point=name,
            start_yaw=start_yaw, end_yaw=end_yaw,
            direction=self.sweep_direction,
            angular_speed=self.sweep_angular_speed,
            sweep_angle=total_angle)

        sensor_deadline = time.monotonic() + 1.5
        current_yaw = None
        while not rospy.is_shutdown() and time.monotonic() < sensor_deadline:
            now = time.monotonic()
            with self.lock:
                odom_yaw = self.odom_yaw
                odom_stamp = self.odom_stamp
                scan_stamp = self.scan_stamp
            if (odom_yaw is not None and now - odom_stamp <= self.sweep_sensor_fresh_s and
                    now - scan_stamp <= self.sweep_sensor_fresh_s):
                current_yaw = odom_yaw
                break
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.04)
        if current_yaw is None:
            self.stop_robot(12)
            self.publish_state("SWEEP_SENSORS_STALE", point=name)
            return "SENSORS_STALE"

        progress = 0.0
        last_yaw = current_yaw
        started = time.monotonic()
        blocked_since = None
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            if self.target_event.is_set():
                self.stop_robot(10)
                return "TARGET"
            if self.candidate_event.is_set():
                if self.hold_ocr_candidate(goal_name):
                    return "TARGET"
                with self.lock:
                    last_yaw = self.odom_yaw

            now = time.monotonic()
            if now - started >= self.scan_timeout:
                self.stop_robot(12)
                self.publish_state("SWEEP_TIMEOUT", point=name,
                                   progress=progress, total=total_angle)
                return "TIMEOUT"
            with self.lock:
                odom_yaw = self.odom_yaw
                odom_stamp = self.odom_stamp
                clearance = self.rotation_clearance
                scan_stamp = self.scan_stamp
            if (odom_yaw is None or now - odom_stamp > self.sweep_sensor_fresh_s or
                    now - scan_stamp > self.sweep_sensor_fresh_s):
                self.cmd_pub.publish(Twist())
                rate.sleep()
                continue

            delta = norm_angle(odom_yaw - last_yaw)
            last_yaw = odom_yaw
            directed_delta = self.sweep_direction * delta
            if directed_delta > 0.0:
                progress += directed_delta
            remaining = max(0.0, total_angle - progress)
            if remaining <= self.sweep_tolerance:
                break

            if clearance < self.sweep_rotation_clearance:
                self.cmd_pub.publish(Twist())
                if blocked_since is None:
                    blocked_since = now
                rospy.logwarn_throttle(
                    0.5,
                    "OCR_SWEEP_ROTATION_BLOCKED point=%s clear=%.3fm required=%.3fm",
                    name, clearance, self.sweep_rotation_clearance)
                if now - blocked_since >= self.sweep_block_timeout:
                    self.stop_robot(12)
                    self.publish_state(
                        "SWEEP_ROTATION_BLOCKED", point=name,
                        clearance=clearance)
                    return "BLOCKED"
                rate.sleep()
                continue
            blocked_since = None

            command_speed = self.sweep_angular_speed
            if remaining < self.sweep_slow_angle:
                command_speed = max(
                    self.sweep_min_angular_speed,
                    self.sweep_angular_speed * remaining /
                    max(self.sweep_slow_angle, self.sweep_tolerance))
            if clearance < self.sweep_slow_clearance:
                clearance_ratio = clamp(
                    (clearance - self.sweep_rotation_clearance) /
                    max(0.01, self.sweep_slow_clearance -
                        self.sweep_rotation_clearance),
                    0.0, 1.0)
                command_speed = max(
                    self.sweep_min_angular_speed,
                    command_speed * clearance_ratio)
            command = self.publish_direct_command(
                wz=self.sweep_direction * command_speed)
            rospy.logwarn_throttle(
                0.5,
                "OCR_SWEEP point=%s progress=%.1f/%.1fdeg remaining=%.1fdeg "
                "wz=%.3f clear=%.3f",
                name, math.degrees(progress), math.degrees(total_angle),
                math.degrees(remaining), command.angular.z, clearance)
            rate.sleep()

        self.stop_robot(12)
        settle_deadline = time.monotonic() + self.sweep_settle_s
        while not rospy.is_shutdown() and time.monotonic() < settle_deadline:
            if self.target_event.is_set():
                return "TARGET"
            if self.candidate_event.is_set() and self.hold_ocr_candidate(goal_name):
                return "TARGET"
            rospy.sleep(0.04)
        self.publish_state("CONTINUOUS_OCR_SWEEP_COMPLETE", point=name,
                           swept_angle=progress)
        return "SUCCEEDED"

    def latest_target_center(self):
        with self.lock:
            payload = None if self.latest_ocr is None else dict(self.latest_ocr)
            target = self.target_warehouse
        if payload is None:
            return None
        if time.monotonic() - float(payload.get(
                "received_monotonic", 0.0)) > self.ocr_fresh_s:
            return None
        if (payload.get("label") != target and
                payload.get("frame_label") != target):
            return None
        width = float(payload.get("image_width", 800) or 800)
        center = payload.get("filtered_center_x")
        if center is None:
            bbox = payload.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None
            center = 0.5 * (float(bbox[0]) + float(bbox[2]))
        return float(center), width, payload

    def align_target_for_parking(self):
        self.stop_robot(12)
        self.publish_state(
            "PREPARK_TARGET_ALIGNMENT",
            tolerance_px=self.prepark_center_tolerance)
        started = time.monotonic()
        stable = 0
        blocked_since = None
        rate = rospy.Rate(20)
        while (not rospy.is_shutdown() and
               time.monotonic() - started < self.prepark_align_timeout):
            target_center = self.latest_target_center()
            now = time.monotonic()
            with self.lock:
                clearance = self.rotation_clearance
                scan_stamp = self.scan_stamp
                odom_stamp = self.odom_stamp
            if (target_center is None or
                    now - scan_stamp > self.sweep_sensor_fresh_s or
                    now - odom_stamp > self.sweep_sensor_fresh_s):
                self.cmd_pub.publish(Twist())
                stable = 0
                rospy.logwarn_throttle(
                    0.8, "PREPARK_ALIGN waiting fresh OCR/lidar/odom")
                rate.sleep()
                continue

            center, width, _ = target_center
            target = 0.5 * width
            pixel_error = center - target
            if abs(pixel_error) <= self.prepark_center_tolerance:
                stable += 1
                blocked_since = None
                self.cmd_pub.publish(Twist())
                command_wz = 0.0
            else:
                stable = 0
                if clearance < self.sweep_rotation_clearance:
                    self.cmd_pub.publish(Twist())
                    if blocked_since is None:
                        blocked_since = now
                    rospy.logwarn_throttle(
                        0.5,
                        "PREPARK_ALIGN_BLOCKED clear=%.3fm required=%.3fm",
                        clearance, self.sweep_rotation_clearance)
                    if now - blocked_since >= self.sweep_block_timeout:
                        self.stop_robot(12)
                        self.publish_state(
                            "PREPARK_ALIGNMENT_BLOCKED",
                            clearance=clearance)
                        return False
                    rate.sleep()
                    continue
                blocked_since = None
                normalized = pixel_error / max(1.0, 0.5 * width)
                angular_error = (self.camera_bearing_sign * normalized *
                                 0.5 * self.camera_hfov)
                command_wz = self.prepark_heading_kp * angular_error
                if abs(command_wz) < self.prepark_min_wz:
                    command_wz = math.copysign(
                        self.prepark_min_wz, angular_error)
                command_wz = clamp(
                    command_wz, -self.prepark_max_wz,
                    self.prepark_max_wz)
                self.publish_direct_command(wz=command_wz)
            rospy.logwarn_throttle(
                0.35,
                "PREPARK_ALIGN center=%.1f target=%.1f error=%.1fpx "
                "stable=%d/%d wz=%.3f clear=%.3f",
                center, target, pixel_error, stable,
                self.prepark_center_stable_frames, command_wz, clearance)
            if stable >= self.prepark_center_stable_frames:
                self.stop_robot(12)
                rospy.sleep(0.4)
                self.publish_state(
                    "PREPARK_TARGET_ALIGNED", pixel_error=pixel_error)
                return True
            rate.sleep()
        self.stop_robot(12)
        self.publish_state("PREPARK_ALIGNMENT_TIMEOUT")
        return False

    def visit_scan_point(self, point):
        name, x, y, start_yaw, end_yaw, fallback = point
        selected = self.nearest_reachable(x, y, start_yaw)
        if selected is None:
            self.publish_state("SCAN_POINT_UNREACHABLE", point=name,
                               fallback=fallback)
            return "UNREACHABLE"
        sx, sy, syaw = selected
        for attempt in range(1, self.goal_retries + 1):
            result = self.send_goal(
                sx, sy, syaw, "{}_approach".format(name), self.goal_timeout)
            if result in ("SUCCEEDED", "TARGET"):
                break
            self.clear_and_wait("{} approach {}".format(name, result))
            replacement = self.nearest_reachable(x, y, start_yaw)
            if replacement is not None:
                sx, sy, syaw = replacement
        if result != "SUCCEEDED":
            return result
        if self.target_event.is_set():
            return "TARGET"

        result = self.controlled_ocr_sweep(name, start_yaw, end_yaw)
        if result not in ("SUCCEEDED", "TARGET"):
            self.clear_and_wait("{} sweep {}".format(name, result))
        return result

    def ray_room_intersection(self, origin_x, origin_y, angle):
        dx = math.cos(angle)
        dy = math.sin(angle)
        candidates = []

        def add(t, x, y, normal):
            if (t > 0.02 and self.room_min_x - 0.02 <= x <= self.room_max_x + 0.02 and
                    self.room_min_y - 0.02 <= y <= self.room_max_y + 0.02):
                candidates.append((t, x, y, normal))

        if abs(dx) > 1.0e-5:
            t = (self.room_min_x - origin_x) / dx
            add(t, self.room_min_x, origin_y + t * dy, (1.0, 0.0))
            t = (self.room_max_x - origin_x) / dx
            add(t, self.room_max_x, origin_y + t * dy, (-1.0, 0.0))
        if abs(dy) > 1.0e-5:
            t = (self.room_min_y - origin_y) / dy
            add(t, origin_x + t * dx, self.room_min_y, (0.0, 1.0))
            t = (self.room_max_y - origin_y) / dy
            add(t, origin_x + t * dx, self.room_max_y, (0.0, -1.0))
        if not candidates:
            return None
        return min(candidates, key=lambda value: value[0])

    def target_approach_goal(self):
        with self.lock:
            snapshot = None if self.target_snapshot is None else dict(
                self.target_snapshot)
        if snapshot is None:
            return None
        pose = snapshot["pose"]
        ocr = snapshot["ocr"]
        bbox = ocr.get("bbox")
        width = float(ocr.get("image_width", 1280) or 1280)
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            pixel_x = 0.5 * (float(bbox[0]) + float(bbox[2]))
        else:
            pixel_x = 0.5 * width
        normalized = clamp((pixel_x - 0.5 * width) / max(1.0, 0.5 * width),
                           -1.0, 1.0)
        bearing = pose[2] + self.camera_bearing_sign * normalized * \
            (0.5 * self.camera_hfov)
        intersection = self.ray_room_intersection(pose[0], pose[1], bearing)
        if intersection is None:
            return None
        _, wall_x, wall_y, normal = intersection
        goal_x = wall_x + normal[0] * self.wall_standoff
        goal_y = wall_y + normal[1] * self.wall_standoff
        goal_yaw = math.atan2(-normal[1], -normal[0])
        self.publish_state(
            "TARGET_WALL_PROJECTED", pixel_x=pixel_x, image_width=width,
            wall_x=wall_x, wall_y=wall_y, approach_x=goal_x,
            approach_y=goal_y, approach_yaw=goal_yaw)
        return goal_x, goal_y, goal_yaw

    def prepare_target_wall_orientation(self):
        projected = self.target_approach_goal()
        pose = self.current_pose()
        if projected is not None:
            target_yaw = projected[2]
            source = "ocr_wall_projection"
        elif pose is not None:
            quarter_turn = math.pi / 2.0
            target_yaw = norm_angle(
                round(pose[2] / quarter_turn) * quarter_turn)
            source = "nearest_cardinal_fallback"
        else:
            self.publish_state("PREPARK_WALL_ORIENTATION_NO_POSE")
            return False

        self.publish_state(
            "PREPARK_WALL_ORIENTATION_START", target_yaw=target_yaw,
            source=source)
        result = self.turn_to_wall(
            "target_wall_prepark", target_yaw, watch_ocr=False)
        if result != "SUCCEEDED":
            self.publish_state(
                "PREPARK_WALL_ORIENTATION_FAILED", result=result,
                target_yaw=target_yaw)
            return False

        deadline = time.monotonic() + 1.5
        estimate = None
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            estimate = self.front_wall_estimate()
            if estimate is not None:
                break
            self.cmd_pub.publish(Twist())
            rospy.sleep(0.05)
        if estimate is None:
            self.publish_state(
                "PREPARK_WALL_MODEL_NOT_READY", target_yaw=target_yaw)
            return False
        self.publish_state(
            "PREPARK_WALL_ORIENTATION_READY", target_yaw=target_yaw,
            wall_distance=estimate[0],
            wall_heading_error=estimate[1], wall_points=estimate[2])
        return True

    def approach_target(self):
        self.stop_robot(12)
        projected = self.target_approach_goal()
        if projected is None:
            self.publish_state("TARGET_PROJECTION_UNAVAILABLE_PARK_DIRECT")
            return True
        selected = self.nearest_reachable(*projected)
        if selected is None:
            self.publish_state("TARGET_APPROACH_UNREACHABLE_PARK_DIRECT")
            return True
        result = self.send_goal(
            selected[0], selected[1], selected[2], "target_wall_approach",
            self.goal_timeout, watch_target=False)
        if result != "SUCCEEDED":
            self.clear_and_wait("target approach {}".format(result))
            result = self.send_goal(
                selected[0], selected[1], selected[2],
                "target_wall_approach_retry", self.goal_timeout,
                watch_target=False)
        self.stop_robot(12)
        return result == "SUCCEEDED"

    def handoff_directly_to_parking(self):
        # Keep the navigation stack and OCR process warm.  There is no roslaunch
        # boundary here: cancel the active goal and take /cmd_vel immediately.
        self.cone_control_pub.publish(Bool(data=False))
        self.stop_robot(4)
        self.ocr_control_pub.publish(String(data="enable"))
        self.publish_state("IN_PROCESS_CENTERLINE_PARKING_HANDOFF")

    @staticmethod
    def ocr_center(payload):
        if not isinstance(payload, dict):
            return None
        width = float(payload.get("image_width", 800) or 800)
        center = payload.get("filtered_center_x")
        if center is None:
            bbox = payload.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                return None
            center = 0.5 * (float(bbox[0]) + float(bbox[2]))
        return float(center), width

    def parking_center_observation_valid(self, payload):
        """Reject edge-clipped OCR boxes before they corrupt parking center."""
        if not isinstance(payload, dict):
            return False
        bbox = payload.get("bbox")
        width = float(payload.get("image_width", 0) or 0)
        height = float(payload.get("image_height", self.ocr_image_height) or
                       self.ocr_image_height)
        if (not isinstance(bbox, (list, tuple)) or len(bbox) < 4 or
                width <= 1.0 or height <= 1.0):
            return False
        margin_x = self.target_edge_margin_ratio * width
        margin_y = self.target_vertical_edge_margin_ratio * height
        return (float(bbox[0]) >= margin_x and
                float(bbox[2]) <= width - margin_x and
                float(bbox[1]) >= margin_y and
                float(bbox[3]) <= height - margin_y)

    def locked_parking_target_center(self):
        now = time.monotonic()
        with self.lock:
            live = None if self.latest_ocr is None else dict(self.latest_ocr)
            locked = (None if self.parking_target_ocr is None else
                      dict(self.parking_target_ocr))
            locked_stamp = self.parking_target_stamp
            snapshot = (None if self.target_snapshot is None else
                        dict(self.target_snapshot.get("ocr", {})))
            target = self.target_warehouse

        live_match = (live is not None and
                      bool(live.get("stable", False)) and
                      live.get("label") == target and
                      live.get("frame_label") == target and
                      self.parking_center_observation_valid(live) and
                      now - float(live.get("received_monotonic", 0.0)) <=
                      self.ocr_fresh_s)
        if live_match:
            center = self.ocr_center(live)
            if center is not None:
                source_stamp = float(live.get("received_monotonic", 0.0))
                with self.lock:
                    if source_stamp > self.parking_center_source_stamp + 1.0e-5:
                        if (self.parking_center_filtered is None or
                                abs(center[1] - self.parking_center_width) > 1.0):
                            filtered = center[0]
                        else:
                            raw_delta = center[0] - self.parking_center_filtered
                            limited_delta = clamp(
                                raw_delta, -self.parking_center_max_step,
                                self.parking_center_max_step)
                            filtered = (self.parking_center_filtered +
                                        self.parking_center_filter_alpha *
                                        limited_delta)
                            if abs(raw_delta) > self.parking_center_max_step:
                                rospy.logwarn_throttle(
                                    0.4, "PARKING_OCR_CENTER_JUMP_FILTERED "
                                    "raw_delta=%.1f limited=%.1f",
                                    raw_delta, limited_delta)
                        self.parking_center_filtered = filtered
                        self.parking_center_width = center[1]
                        self.parking_center_source_stamp = source_stamp
                    self.parking_target_ocr = dict(live)
                    self.parking_target_stamp = now
                    filtered = self.parking_center_filtered
                    filtered_width = self.parking_center_width
                return filtered, filtered_width, 0.0, True

        source = locked if locked is not None else snapshot
        center = self.ocr_center(source)
        if center is None:
            return None
        age = now - locked_stamp if locked_stamp > 0.0 else float("inf")
        with self.lock:
            if self.parking_center_filtered is None:
                self.parking_center_filtered = center[0]
                self.parking_center_width = center[1]
            filtered = self.parking_center_filtered
            filtered_width = self.parking_center_width
        return filtered, filtered_width, age, False

    def parking_abort_reason(self):
        if self.parking_wrong_event.is_set():
            with self.lock:
                label = self.parking_wrong_label
            return "WRONG_WORKSHOP:{}".format(label or "unknown")
        return ""

    def reset_target_and_resume_wall_scan(self, reason):
        self.smooth_stop_robot()
        resume_index = self.wall_route_resume_index()
        with self.lock:
            wrong_label = self.parking_wrong_label
            self.parking_active = False
            self.parking_wrong_label = ""
            self.target_snapshot = None
            self.parking_target_ocr = None
            self.parking_target_stamp = 0.0
            self.parking_center_filtered = None
            self.parking_center_width = 0.0
            self.parking_center_source_stamp = 0.0
            self.parking_center_aligned = False
            self.target_confirm_count = 0
            self.candidate_cooldown_until = (
                time.monotonic() + self.candidate_cooldown_s)
        self.parking_wrong_event.clear()
        self.target_event.clear()
        self.candidate_event.clear()
        self.cone_control_pub.publish(Bool(data=True))
        self.ocr_control_pub.publish(String(data="enable"))
        self.publish_state(
            "PARKING_ABORTED_RESUME_WALL_SCAN", reason=reason,
            rejected_label=wrong_label, resume_segment=resume_index)
        return resume_index

    def parking_footprint_side_clearance(self, vy, wall_distance):
        """Return the closest lidar point to the moving side of the rectangle."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        closest = None
        for _, _, x_base, y_base in samples:
            if (vy > 0.0 and y_base <= 0.0) or (vy < 0.0 and y_base >= 0.0):
                continue
            x_laser, _ = self.base_to_laser(x_base, y_base)
            # Exclude fitted front-wall returns while retaining compact cones
            # along the complete front and rear side edges.
            if abs(x_laser - wall_distance) <= 0.05:
                continue
            dx = max(abs(x_base) - self.robot_half_length, 0.0)
            dy = max(abs(y_base) - self.robot_half_width, 0.0)
            gap = math.hypot(dx, dy)
            if closest is None or gap < closest[0]:
                closest = (gap, x_base, y_base)
        return closest

    def parking_side_guard(self, vy, wall_distance):
        self.parking_side_blocker = None
        if abs(vy) < 1.0e-4:
            return 0.0
        side_angle = math.pi / 2.0 if vy > 0.0 else -math.pi / 2.0
        clearance = self.parking_raw_sector_clearance(
            side_angle, math.radians(32.0))
        corner = self.parking_footprint_side_clearance(vy, wall_distance)
        if corner is not None:
            self.parking_side_blocker = corner
        if clearance is None and corner is None:
            return 0.0
        radial_blocked = (clearance is not None and
                          clearance <= self.parking_lateral_hard_clearance)
        corner_blocked = (corner is not None and
                          corner[0] <= self.parking_corner_hard_gap)
        if radial_blocked or corner_blocked:
            rospy.logwarn_throttle(
                0.4, "IN_PROCESS_PARKING_SIDE_BLOCKED side=%s gap=%s "
                "point=%s vy=%.3f",
                "none" if clearance is None else "{:.3f}".format(clearance),
                "none" if corner is None else "{:.3f}".format(corner[0]),
                "none" if corner is None else
                "({:.3f},{:.3f})".format(corner[1], corner[2]), vy)
            return 0.0
        ratios = [1.0]
        if clearance is not None:
            ratios.append(clamp(
                (clearance - self.parking_lateral_hard_clearance) /
                max(0.02, self.parking_lateral_slow_clearance -
                    self.parking_lateral_hard_clearance), 0.0, 1.0))
        if corner is not None:
            ratios.append(clamp(
                (corner[0] - self.parking_corner_hard_gap) /
                max(0.02, self.parking_corner_slow_gap -
                    self.parking_corner_hard_gap), 0.0, 1.0))
        ratio = min(ratios)
        return vy * ratio

    def parking_rotation_guard(self, wz):
        """Slow only rotations that sweep the rear footprint into a scan hit."""
        if abs(wz) < 1.0e-4:
            return 0.0, None
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return 0.0, None
        direction = 1.0 if wz > 0.0 else -1.0
        threat = None
        for _, _, x_base, y_base in samples:
            if x_base >= -0.03:
                continue
            nearest_x = clamp(
                x_base, -self.robot_half_length, self.robot_half_length)
            nearest_y = clamp(
                y_base, -self.robot_half_width, self.robot_half_width)
            dx = x_base - nearest_x
            dy = y_base - nearest_y
            gap = math.hypot(dx, dy)
            if gap <= 1.0e-6:
                closing = 1.0
            else:
                velocity_x = -direction * nearest_y
                velocity_y = direction * nearest_x
                closing = (velocity_x * dx + velocity_y * dy) / gap
            if closing <= 0.0:
                continue
            if threat is None or gap < threat[0]:
                threat = (gap, x_base, y_base)
        if threat is None:
            return wz, None
        ratio = clamp(
            (threat[0] - self.parking_corner_turn_hard_gap) /
            max(0.02, self.parking_corner_turn_slow_gap -
                self.parking_corner_turn_hard_gap), 0.0, 1.0)
        guarded = wz * ratio
        if ratio < 0.999:
            rospy.logwarn_throttle(
                0.4, "IN_PROCESS_PARKING_REAR_TURN_GUARD gap=%.3f "
                "point=(%.3f,%.3f) wz=%.3f->%.3f",
                threat[0], threat[1], threat[2], wz, guarded)
        return guarded, threat

    def parking_raw_sector_clearance(self, center_angle, half_angle):
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        values = []
        for _, _, x_base, y_base in samples:
            x_laser, y_laser = self.base_to_laser(x_base, y_base)
            distance = math.hypot(x_laser, y_laser)
            angle = math.atan2(y_laser, x_laser)
            if abs(norm_angle(angle - center_angle)) <= half_angle:
                values.append(distance)
        return min(values) if values else None

    def parking_front_wall_estimate(self):
        """Fit the parking wall in the lidar frame, matching the proven test."""
        samples, stamp = self.scan_snapshot()
        if time.monotonic() - stamp > self.sweep_sensor_fresh_s:
            return None
        points = []
        sector = math.radians(55.0)
        for _, _, x_base, y_base in samples:
            x_laser, y_laser = self.base_to_laser(x_base, y_base)
            distance = math.hypot(x_laser, y_laser)
            angle = math.atan2(y_laser, x_laser)
            if abs(angle) <= sector and 0.10 <= distance <= 1.60:
                points.append((x_laser, y_laser))
        if len(points) < 12:
            return None

        stride = max(1, len(points) // 34)
        indices = list(range(0, len(points), stride))
        best = None
        best_score = -1.0
        threshold = 0.04
        for first_pos, first in enumerate(indices):
            x1, y1 = points[first]
            for second in indices[first_pos + 2:]:
                x2, y2 = points[second]
                dy = y2 - y1
                if abs(dy) < 0.12:
                    continue
                slope = (x2 - x1) / dy
                intercept = x1 - slope * y1
                scale = math.sqrt(1.0 + slope * slope)
                inliers = [
                    point for point in points
                    if abs(point[0] - slope * point[1] - intercept) /
                    scale <= threshold
                ]
                if len(inliers) < 12:
                    continue
                tangent_x = slope / scale
                tangent_y = 1.0 / scale
                projected = [
                    point[0] * tangent_x + point[1] * tangent_y
                    for point in inliers
                ]
                span = max(projected) - min(projected)
                if span < 0.20:
                    continue
                score = len(inliers) + 12.0 * span
                if score > best_score:
                    best_score = score
                    best = inliers
        if best is None:
            return None

        mean_y = sum(point[1] for point in best) / len(best)
        mean_x = sum(point[0] for point in best) / len(best)
        denominator = sum((point[1] - mean_y) ** 2 for point in best)
        if denominator < 1.0e-6:
            return None
        slope = sum(
            (point[1] - mean_y) * (point[0] - mean_x)
            for point in best) / denominator
        intercept = mean_x - slope * mean_y
        scale = math.sqrt(1.0 + slope * slope)
        distance = intercept / scale
        heading_error = -math.atan(slope)
        if (distance <= 0.0 or
                abs(heading_error) > math.radians(35.0)):
            return None
        return distance, heading_error, len(best)

    def parking_pose_text(self):
        with self.lock:
            pose = None if self.pose is None else tuple(self.pose)
        if pose is None:
            return "unavailable"
        return "(%.3f,%.3f,%.1fdeg)" % (
            pose[0], pose[1], math.degrees(pose[2]))

    def parking_pose_values(self):
        with self.lock:
            pose = None if self.pose is None else tuple(self.pose)
        if pose is None:
            return {}
        return {"map_x": pose[0], "map_y": pose[1], "map_yaw": pose[2]}

    def parking_commands(self, allow_forward):
        wall = self.parking_front_wall_estimate()
        if wall is None:
            return None
        wall_distance, heading_error, points = wall
        target = self.locked_parking_target_center()

        wz = clamp(self.parking_heading_kp * heading_error,
                   -self.parking_max_wz, self.parking_max_wz)
        if (abs(heading_error) > self.parking_heading_tolerance and
                abs(wz) < self.parking_min_wz):
            wz = math.copysign(self.parking_min_wz, heading_error)
        if abs(heading_error) <= self.parking_heading_tolerance:
            wz = 0.0

        pixel_error = 0.0
        target_live = False
        target_age = float("inf")
        vy = 0.0
        if target is not None:
            center, width, target_age, target_live = target
            pixel_error = center - 0.5 * width
            # A remembered center bridges brief OCR gaps, but must not drive a
            # long blind translation after the image has changed.
            if target_live or target_age <= self.parking_target_memory_s:
                normalized = pixel_error / max(1.0, 0.5 * width)
                vy = (self.parking_lateral_sign * self.parking_lateral_kp *
                      normalized)
                vy = clamp(vy, -self.parking_max_vy, self.parking_max_vy)
                active_center_tolerance = (
                    self.parking_recenter_threshold if allow_forward else
                    self.parking_center_tolerance)
                if (abs(pixel_error) > active_center_tolerance and
                        abs(vy) < self.parking_min_vy):
                    vy = math.copysign(self.parking_min_vy, vy)
                if abs(pixel_error) <= active_center_tolerance:
                    vy = 0.0
            # Once the wall is visible, yaw belongs exclusively to the lidar
            # wall fit.  Chasing horizontal OCR error with yaw made the robot
            # repeatedly lose its perpendicular pose and oscillate around the
            # sign.  An omni base can remove that error directly with vy while
            # preserving wall alignment.
        if allow_forward:
            vy = clamp(vy, -self.parking_approach_max_vy,
                       self.parking_approach_max_vy)
        requested_vy = vy
        vy = self.parking_side_guard(vy, wall_distance)
        self.parking_lateral_requested = requested_vy
        self.parking_lateral_guarded = vy
        requested_wz = wz
        wz, turn_blocker = self.parking_rotation_guard(wz)

        vx = 0.0
        distance_error = wall_distance - self.parking_wall_distance
        side_blocked = (abs(requested_vy) > 1.0e-4 and abs(vy) < 1.0e-4)
        blocker = self.parking_side_blocker
        if (not allow_forward and side_blocked and blocker is not None and
                blocker[1] < -0.35 * self.robot_half_length and
                wall_distance >= self.parking_rear_clear_wall_min):
            front = self.parking_raw_sector_clearance(
                0.0, math.radians(16.0))
            if front is not None and front >= self.parking_rear_clear_front_min:
                vx = self.parking_rear_clear_nudge_speed
                rospy.logwarn_throttle(
                    0.4, "IN_PROCESS_PARKING_REAR_CORNER_CLEAR "
                    "gap=%.3f point=(%.3f,%.3f) vx=%.3f",
                    blocker[0], blocker[1], blocker[2], vx)
        if (side_blocked and blocker is not None and
                blocker[1] > 0.03):
            escape = self.side_escape_velocity(blocker, "PARKING")
            if escape < -1.0e-4:
                vx = escape
        # A cone just outside a rectangular corner can reduce lateral velocity
        # to a few millimetres per second without making it exactly zero.  That
        # is a deadlock, not useful caution.  Flow around the compact obstacle:
        # move longitudinally according to whether it is beside the front or
        # rear half, while also moving laterally away from it.  Sensor feedback
        # ends the manoeuvre as soon as the side gap opens; no fixed retreat
        # distance is used.
        lateral_deadlock = (
            abs(requested_vy) >= self.parking_min_vy and
            abs(vy) < self.parking_recovery_trigger_vy and
            blocker is not None)
        if lateral_deadlock:
            original_blocker = blocker
            escape_vx = self.side_escape_velocity(
                original_blocker, "PARKING_CENTER")
            if (escape_vx > 0.0 and
                    wall_distance <= self.parking_wall_distance +
                    self.parking_forward_escape_margin):
                escape_vx = 0.0
            away_vy = -math.copysign(
                self.parking_escape_lateral_speed, original_blocker[2])
            away_vy = self.parking_side_guard(away_vy, wall_distance)
            self.parking_side_blocker = original_blocker
            vy = 0.0
            if abs(away_vy) >= 1.0e-4:
                vy = away_vy
            vx = escape_vx
            rospy.logwarn_throttle(
                0.4, "IN_PROCESS_PARKING_CONE_FLOW gap=%.3f "
                "point=(%.3f,%.3f) requested_vy=%.3f guarded_vy=%.3f "
                "escape=(%.3f,%.3f)", original_blocker[0],
                original_blocker[1], original_blocker[2], requested_vy,
                self.parking_lateral_guarded, vx, vy)
        turn_blocked = (abs(requested_wz) > 1.0e-4 and abs(wz) < 1.0e-4)
        if (not allow_forward and turn_blocked and turn_blocker is not None and
                wall_distance >= self.parking_rear_clear_wall_min):
            front = self.parking_raw_sector_clearance(
                0.0, math.radians(16.0))
            if front is not None and front >= self.parking_rear_clear_front_min:
                vx = max(vx, self.parking_rear_clear_nudge_speed)
        target_centered_for_motion = (
            target is not None and
            target_age <= self.parking_target_reacquire_s and
            abs(pixel_error) <= self.parking_recenter_threshold)
        if (allow_forward and target_centered_for_motion and
                abs(heading_error) <= math.radians(8.0)):
            front = self.parking_raw_sector_clearance(
                0.0, math.radians(12.0))
            if (front is not None and front <= self.parking_front_emergency):
                vx = 0.0
            elif distance_error > self.parking_wall_tolerance:
                ratio = clamp(distance_error / max(
                    self.parking_slow_distance, 1.0e-3), 0.0, 1.0)
                vx = (self.parking_slow_vx +
                      ratio * (self.parking_fast_vx - self.parking_slow_vx))
            elif distance_error < -self.parking_wall_tolerance:
                # Never reverse during final parking for a centimetre-scale
                # overshoot.  A blind correction can sweep the rear corners
                # into a cone that was safely behind the chassis.
                vx = 0.0

        # During initial acquisition a large yaw correction sweeps the long
        # rectangular corners.  Complete that rotation before translating.
        if (not allow_forward and
                abs(heading_error) > self.parking_translation_heading_gate):
            vx = 0.0
            vy = 0.0

        vx, vy, wz = self.predictive_motion_guard(
            vx, vy, wz, wall_distance, "IN_PROCESS_PARKING")

        return (vx, vy, wz, wall_distance, heading_error, points,
                pixel_error, target_live, target_age)

    def park(self):
        with self.lock:
            self.parking_active = True
            self.parking_wrong_label = ""
        self.parking_wrong_event.clear()
        self.publish_state("IN_PROCESS_PARKING_STARTED",
                           target=self.target_warehouse)
        deadline = time.monotonic() + self.parking_timeout
        self.parking_center_aligned = False

        # Phase 1: first become perpendicular and place the requested sign on
        # the camera centerline.  Never rush forward while a clipped sign is at
        # the image edge; that was the source of both the cone touch and the
        # out-of-box final pose.
        align_stable = 0
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            abort_reason = self.parking_abort_reason()
            if abort_reason:
                self.smooth_stop_robot()
                with self.lock:
                    self.parking_active = False
                self.publish_state(
                    "IN_PROCESS_PARKING_ABORTED", reason=abort_reason,
                    **self.parking_pose_values())
                return abort_reason
            values = self.parking_commands(allow_forward=False)
            if values is None:
                target = self.locked_parking_target_center()
                search_wz = 0.0
                if target is not None:
                    center, width, age, live = target
                    if live or age <= self.parking_target_memory_s:
                        normalized = ((center - 0.5 * width) /
                                      max(1.0, 0.5 * width))
                        angular_error = (self.camera_bearing_sign * normalized *
                                         0.5 * self.camera_hfov)
                        search_wz = clamp(
                            self.prepark_heading_kp * angular_error,
                            -self.parking_max_wz, self.parking_max_wz)
                self.publish_direct_command(wz=search_wz)
                align_stable = 0
                rospy.logwarn_throttle(
                    0.5, "IN_PROCESS_PARKING_ALIGN waiting wall wz=%.3f",
                    search_wz)
                rate.sleep()
                continue
            (vx, vy, wz, distance, heading, points, pixel_error,
             target_live, target_age) = values
            self.publish_direct_command(vx, vy, wz)
            aligned = (target_age <= self.parking_target_memory_s and
                       abs(pixel_error) <= self.parking_center_tolerance and
                       abs(heading) <= self.parking_heading_tolerance)
            align_stable = align_stable + 1 if aligned else 0
            rospy.logwarn_throttle(
                0.25,
                "IN_PROCESS_PARKING_ALIGN wall=%.3f heading=%.1fdeg "
                "pixel=%.1f live=%s pose=%s cmd=(%.3f,%.3f,%.3f) "
                "stable=%d/%d",
                distance, math.degrees(heading), pixel_error,
                str(target_live), self.parking_pose_text(), vx, vy, wz,
                align_stable,
                self.parking_stable_frames)
            if align_stable >= self.parking_stable_frames:
                self.parking_center_aligned = True
                self.publish_state(
                    "IN_PROCESS_PARKING_CENTERED", pixel_error=pixel_error,
                    wall_heading_error=heading,
                    **self.parking_pose_values())
                break
            rate.sleep()

        if not self.parking_center_aligned:
            self.smooth_stop_robot()
            with self.lock:
                self.parking_active = False
            self.publish_state("IN_PROCESS_PARKING_ALIGNMENT_TIMEOUT",
                               **self.parking_pose_values())
            return "TIMEOUT"

        # Phase 2: approach the wall while preserving the established center.
        stable = 0
        while not rospy.is_shutdown() and time.monotonic() < deadline:
            abort_reason = self.parking_abort_reason()
            if abort_reason:
                self.smooth_stop_robot()
                with self.lock:
                    self.parking_active = False
                self.publish_state(
                    "IN_PROCESS_PARKING_ABORTED", reason=abort_reason,
                    **self.parking_pose_values())
                return abort_reason
            values = self.parking_commands(allow_forward=True)
            if values is None:
                # If the wall is initially outside the fit angle, use the
                # already locked sign bearing to bring it into the lidar/front
                # field without invoking a map goal or a separate process.
                target = self.locked_parking_target_center()
                search_wz = 0.0
                if target is not None:
                    center, width, age, live = target
                    if live or age <= self.parking_target_memory_s:
                        normalized = ((center - 0.5 * width) /
                                      max(1.0, 0.5 * width))
                        angular_error = (self.camera_bearing_sign * normalized *
                                         0.5 * self.camera_hfov)
                        search_wz = clamp(
                            self.prepark_heading_kp * angular_error,
                            -self.parking_max_wz, self.parking_max_wz)
                self.publish_direct_command(wz=search_wz)
                stable = 0
                rospy.logwarn_throttle(
                    0.5, "IN_PROCESS_PARKING waiting wall model search_wz=%.3f",
                    search_wz)
                rate.sleep()
                continue
            (vx, vy, wz, distance, heading, points, pixel_error,
             target_live, target_age) = values
            self.publish_direct_command(vx, vy, wz)
            target = self.locked_parking_target_center()
            centered = (
                target is not None and
                target[2] <= self.parking_target_reacquire_s and
                abs(pixel_error) <= self.parking_recenter_threshold)
            complete = (abs(distance - self.parking_wall_distance) <=
                        self.parking_wall_tolerance and
                        abs(heading) <= self.parking_heading_tolerance and
                        centered)
            stable = stable + 1 if complete else 0
            rospy.logwarn_throttle(
                0.25,
                "IN_PROCESS_PARKING wall=%.3f/%.3f heading=%.1fdeg "
                "pixel=%.1f live=%s pose=%s cmd=(%.3f,%.3f,%.3f) "
                "stable=%d/%d",
                distance, self.parking_wall_distance, math.degrees(heading),
                pixel_error, str(target_live), self.parking_pose_text(),
                vx, vy, wz, stable,
                self.parking_stable_frames)
            if stable >= self.parking_stable_frames:
                self.smooth_stop_robot()
                with self.lock:
                    self.parking_active = False
                self.publish_state(
                    "PARKING_CONFIRMED", wall_distance=distance,
                    wall_heading_error=heading, pixel_error=pixel_error,
                    in_process=True, **self.parking_pose_values())
                return "SUCCEEDED"
            rate.sleep()
        self.smooth_stop_robot()
        with self.lock:
            self.parking_active = False
        self.publish_state("IN_PROCESS_PARKING_TIMEOUT",
                           **self.parking_pose_values())
        return "TIMEOUT"

    def final_announcement(self, parking_success, reason=""):
        self.stop_robot(20)
        text = "已将{}放入{}".format(
            self.selected_item, self.target_warehouse)
        self.tts_pub.publish(String(data=text))
        self.publish_state(
            "COMPLETE" if parking_success else "COMPLETE_SAFE_FALLBACK",
            parking_success=parking_success, reason=reason,
            announcement=text)
        self.finished = True

    def mission_thread(self):
        deadline = time.monotonic() + self.mission_timeout
        parking_success = False
        reason = ""
        ocr_start = {"error": None}
        ocr_thread = None

        def preload_ocr():
            try:
                self.start_ocr()
            except Exception as exc:
                ocr_start["error"] = exc

        try:
            self.publish_state("WAITING_INITIAL_TTS", wait_s=self.wait_after_tts)
            rospy.sleep(max(0.0, self.wait_after_tts))
            # Model loading is independent of chassis motion.  Start it while
            # the proven first-stage navigator crosses the doorway so the room
            # inspection can begin without a stationary roslaunch handoff.
            ocr_thread = threading.Thread(target=preload_ocr)
            ocr_thread.daemon = True
            ocr_thread.start()
            if not self.enter_room_with_first_stage_navigation():
                raise RuntimeError("first-stage room entry not crossed")
            ocr_thread.join(self.ocr_ready_timeout + 2.0)
            if ocr_thread.is_alive():
                raise RuntimeError("OCR preload did not finish")
            if ocr_start["error"] is not None:
                raise ocr_start["error"]
            self.replace_move_base()
            with self.lock:
                self.latest_ocr = None
                self.target_snapshot = None
                self.parking_target_ocr = None
                self.parking_target_stamp = 0.0
                self.parking_center_filtered = None
                self.parking_center_width = 0.0
                self.parking_center_source_stamp = 0.0
            self.target_event.clear()
            self.candidate_event.clear()
            self.room_search_active = True
            self.ocr_control_pub.publish(String(data="reset"))
            self.ocr_control_pub.publish(String(data="enable"))
            self.publish_state("ROOM_SEARCH_ACTIVE", seamless_handoff=True)
            resume_index = 0
            approach_start = True
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                route_result = self.run_wall_route(
                    start_index=resume_index,
                    approach_start=approach_start)
                found = (route_result == "TARGET" or
                         self.target_event.is_set())
                if not found:
                    reason = "wall inspection route completed without target"
                    break

                self.publish_state("TARGET_FOUND",
                                   target=self.target_warehouse)
                self.handoff_directly_to_parking()
                parking_result = self.park()
                if parking_result == "SUCCEEDED":
                    parking_success = True
                    break
                if parking_result.startswith("WRONG_WORKSHOP:"):
                    resume_index = self.reset_target_and_resume_wall_scan(
                        parking_result)
                    if resume_index >= len(self.wall_route_points) - 1:
                        reason = "wall inspection route completed after rejected target"
                        break
                    approach_start = False
                    continue
                reason = "in-process parking timeout"
                break
            if time.monotonic() >= deadline and not parking_success:
                reason = "mission timeout"
        except Exception as exc:
            reason = str(exc)
            rospy.logerr("XUNFEI2026_ROOM_EXCEPTION %s", exc)
        finally:
            if ocr_thread is not None and ocr_thread.is_alive():
                ocr_thread.join(self.ocr_ready_timeout + 2.0)
            self.room_search_active = False
            self.cone_control_pub.publish(Bool(data=False))
            self.ocr_control_pub.publish(String(data="disable"))
            self.final_announcement(parking_success, reason)

    def shutdown(self):
        if self.shutdown_started:
            return
        self.shutdown_started = True
        self.cone_control_pub.publish(Bool(data=False))
        self.stop_robot(8)
        self.stop_process(self.parking_process, "parking")
        self.stop_process(self.room_process, "room move_base")
        self.stop_process(self.ocr_process, "OCR")


if __name__ == "__main__":
    Xunfei2026RoomDeliveryManager()
    rospy.spin()
