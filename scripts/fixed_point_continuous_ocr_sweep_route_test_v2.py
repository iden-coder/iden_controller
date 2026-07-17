#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Faster continuous OCR sweeps with stable cone tracking and arrival yaw."""

import math

import rospy
from std_msgs.msg import Bool

from adaptive_scan_route_core_v2 import (
    approach_point,
    arrival_heading_correction,
    merge_groups,
)
from continuous_yaw_sweep_core_v1 import (
    braking_limited_speed,
    slew_rate,
    wrap_angle,
)
from fixed_point_continuous_ocr_sweep_route_test_v1 import (
    FixedPointContinuousOcrSweepRouteTest,
)


class FixedPointContinuousOcrSweepRouteTestV2(
        FixedPointContinuousOcrSweepRouteTest):
    def __init__(self):
        # Parent constructors start the control timer before initialization ends.
        self.v2_ready = False
        self.progress_route_index = -1
        self.translation_progress_pose = None
        self.translation_progress_stamp = None
        self.candidate_waiting = False
        self.candidate_retry_not_before = None
        self.approach_align_cmd = 0.0
        self.approach_align_last = None
        self.route_transition_hold_until = None
        self.scan_safety_mode_active = False
        self.scan_safety_mode_pub = None
        super(FixedPointContinuousOcrSweepRouteTestV2, self).__init__()

        self.track_merge_radius = float(rospy.get_param(
            "~dynamic_track_merge_radius_m", 0.23))
        self.translation_progress_distance = float(rospy.get_param(
            "~route_translation_progress_m", 0.045))
        self.translation_stuck_timeout = float(rospy.get_param(
            "~route_translation_stuck_timeout_s", 4.8))
        self.candidate_retry_delay = float(rospy.get_param(
            "~continuous_scan_candidate_retry_delay_s", 2.5))
        self.approach_distance = float(rospy.get_param(
            "~continuous_scan_approach_distance_m", 0.40))
        self.arrival_blend_distance = float(rospy.get_param(
            "~continuous_scan_arrival_blend_distance_m", 0.40))
        self.arrival_heading_gain = float(rospy.get_param(
            "~continuous_scan_arrival_heading_gain", 0.80))
        self.arrival_heading_max = float(rospy.get_param(
            "~continuous_scan_arrival_heading_max_rps", 0.35))
        self.arrival_large_error = math.radians(float(rospy.get_param(
            "~continuous_scan_arrival_large_error_deg", 38.0)))
        self.arrival_large_error_speed = float(rospy.get_param(
            "~continuous_scan_arrival_large_error_speed_mps", 0.10))
        self.approach_align_max_speed = float(rospy.get_param(
            "~continuous_scan_approach_align_max_rps", 0.48))
        self.approach_align_min_speed = float(rospy.get_param(
            "~continuous_scan_approach_align_min_rps", 0.07))
        self.approach_align_acceleration = float(rospy.get_param(
            "~continuous_scan_approach_align_acceleration_rps2", 0.80))
        self.route_transition_observe_s = float(rospy.get_param(
            "~continuous_scan_route_transition_observe_s", 0.65))
        self.scan_safety_mode_pub = rospy.Publisher(
            rospy.get_param("~continuous_scan_safety_topic",
                            "/continuous_scan/safety_mode"),
            Bool, queue_size=2, latch=True)
        self._set_scan_safety_mode(False, force=True)

        self._reset_translation_progress()
        self.v2_ready = True
        rospy.logwarn(
            "FIXED_CONTINUOUS_SCAN_V2_READY scan=%.2frps ocr_full_speed=true "
            "merge=%.2fm no_progress=%.1fs paired_approach=%.2fm "
            "approach_align=%.2frps",
            self.scan_max_speed,
            self.track_merge_radius, self.translation_stuck_timeout,
            self.approach_distance, self.approach_align_max_speed)

    def _set_scan_safety_mode(self, active, force=False):
        active = bool(active)
        if self.scan_safety_mode_pub is None:
            return
        if not force and active == self.scan_safety_mode_active:
            return
        self.scan_safety_mode_active = active
        self.scan_safety_mode_pub.publish(Bool(data=active))
        rospy.logwarn(
            "CONTINUOUS_SCAN_SAFETY_MODE %s",
            "active" if active else "clear")

    def _begin_scan(self):
        result = super(FixedPointContinuousOcrSweepRouteTestV2,
                       self)._begin_scan()
        self._set_scan_safety_mode(self.scan_active)
        return result

    def _reset_translation_progress(self):
        self.progress_route_index = getattr(self, "route_index", -1)
        if getattr(self, "pose", None) is None:
            self.translation_progress_pose = None
        else:
            self.translation_progress_pose = (self.pose[0], self.pose[1])
        self.translation_progress_stamp = rospy.Time.now()

    def ocr_callback(self, msg):
        # The OCR node publishes label=unknown while accumulating votes. That is
        # not a candidate and must never hold the sweep at a reduced speed.
        super(FixedPointContinuousOcrSweepRouteTestV2, self).ocr_callback(msg)

    def _scan_control_loop(self):
        super(FixedPointContinuousOcrSweepRouteTestV2,
              self)._scan_control_loop()

    def _update_tracks(self, candidates, stamp):
        super(FixedPointContinuousOcrSweepRouteTestV2,
              self)._update_tracks(candidates, stamp)
        if not self.v2_ready or len(self.obstacle_tracks) < 2:
            return
        points = [(track.x, track.y) for track in self.obstacle_tracks]
        groups = merge_groups(points, self.track_merge_radius)
        if len(groups) == len(self.obstacle_tracks):
            return

        merged = []
        merged_count = 0
        confirmed_changed = False
        for group in groups:
            tracks = [self.obstacle_tracks[index] for index in group]
            keep = min(tracks, key=lambda track: track.track_id)
            if len(tracks) > 1:
                weights = [max(1.0, min(float(track.hits), 20.0))
                           for track in tracks]
                total = sum(weights)
                keep.x = sum(track.x * weight
                             for track, weight in zip(tracks, weights)) / total
                keep.y = sum(track.y * weight
                             for track, weight in zip(tracks, weights)) / total
                keep.first_seen = min(
                    (track.first_seen for track in tracks),
                    key=lambda value: value.to_sec())
                keep.last_seen = max(
                    (track.last_seen for track in tracks),
                    key=lambda value: value.to_sec())
                keep.hits = max(track.hits for track in tracks)
                was_confirmed = keep.confirmed
                keep.confirmed = any(track.confirmed for track in tracks)
                confirmed_changed = confirmed_changed or keep.confirmed or was_confirmed
                merged_count += len(tracks) - 1
            merged.append(keep)
        self.obstacle_tracks = merged
        if confirmed_changed:
            self.dynamic_layer_dirty = True
        rospy.logwarn_throttle(
            1.0,
            "TEMP_OBSTACLE_TRACKS_MERGED removed=%d retained=%d radius=%.2f",
            merged_count, len(merged), self.track_merge_radius)

    def compute_cmd(self, target):
        vx, wz = super(FixedPointContinuousOcrSweepRouteTestV2,
                       self).compute_cmd(target)
        if (not self.v2_ready or self.scan_active or
                not self._is_scan_point() or self.pose is None):
            return vx, wz
        point = self.current_route_point()
        distance = math.hypot(point["x"] - self.pose[0],
                              point["y"] - self.pose[1])
        if distance >= self.arrival_blend_distance:
            return vx, wz
        yaw_error = wrap_angle(
            self.scan_specs[point["name"]]["start"] - self.pose[2])
        correction = arrival_heading_correction(
            yaw_error, distance, self.arrival_blend_distance,
            self.arrival_heading_gain, self.arrival_heading_max)
        wz = max(-self.max_turn_cmd,
                 min(self.max_turn_cmd, wz + correction))
        if abs(yaw_error) > self.arrival_large_error:
            vx = min(vx, self.arrival_large_error_speed)
        rospy.logwarn_throttle(
            0.8,
            "SCAN_ARRIVAL_GUIDANCE point=%s dist=%.2f yaw_error=%.1fdeg "
            "correction=%.3f",
            point["name"], distance, math.degrees(yaw_error), correction)
        return vx, wz

    def _candidate_plan_with_approach(self, start, requested, point_name):
        spec = self.scan_specs[point_name]
        prepoint = approach_point(
            requested[0], requested[1], spec["start"],
            self.approach_distance)
        if not self._dynamic_spin_safe(*prepoint):
            return None
        first = self.planner.plan(start, prepoint)
        if not first.get("ok"):
            return None
        second = self.planner.plan(first["active_goal_world"], requested)
        if not second.get("ok"):
            return None
        path = list(first["path_world"])
        second_path = list(second["path_world"])
        if path and second_path and math.hypot(
                path[-1][0] - second_path[0][0],
                path[-1][1] - second_path[0][1]) < 0.03:
            second_path = second_path[1:]
        return {
            "ok": True,
            "active_goal_world": second["active_goal_world"],
            "path_world": path + second_path,
            "approach_world": first["active_goal_world"],
        }

    def _select_alternative_scan_point(self, reason):
        self._set_scan_safety_mode(False)
        if self.pose is None:
            return False
        now = rospy.Time.now()
        if (self.candidate_retry_not_before is not None and
                now < self.candidate_retry_not_before):
            return False
        point = self.current_route_point()
        name = point["name"]
        nominal_x = float(point.get("nominal_x", point["x"]))
        nominal_y = float(point.get("nominal_y", point["y"]))
        point["nominal_x"] = nominal_x
        point["nominal_y"] = nominal_y
        attempts = set()
        self.scan_candidate_attempts[name] = attempts
        attempts.add((round(point["x"], 3), round(point["y"], 3)))
        self.publish_zero("SCAN_POINT_ALTERNATE_SEARCH_V2")
        self._set_ocr(False)
        self.scan_active = False

        start = (self.pose[0], self.pose[1])
        candidates = []
        for offset_x, offset_y in self.scan_candidate_offsets:
            requested = (nominal_x + offset_x, nominal_y + offset_y)
            key = (round(requested[0], 3), round(requested[1], 3))
            if key in attempts:
                continue
            attempts.add(key)
            if (not self._static_spin_safe(*requested) or
                    not self._dynamic_spin_safe(*requested)):
                continue
            candidates.append(requested)

        # Stay local first. A direct path avoids sending a robot that is already
        # near a scan point away to a paired approach point and back again.
        for requested in candidates:
            result = self.planner.plan(start, requested)
            if not result.get("ok"):
                continue
            active = result["active_goal_world"]
            if (not self._static_spin_safe(*active) or
                    not self._dynamic_spin_safe(*active)):
                continue
            self._install_planned_candidate(point, requested, result, reason)
            self.candidate_waiting = False
            self.candidate_retry_not_before = None
            self._reset_translation_progress()
            rospy.logwarn(
                "CONTINUOUS_SCAN_DIRECT_FIRST point=%s "
                "local_motion_preferred=true",
                name)
            return True

        # Use an arrival-yaw approach only when no direct candidate is usable.
        for requested in candidates:
            result = self._candidate_plan_with_approach(start, requested, name)
            if result is None:
                continue
            active = result["active_goal_world"]
            if (not self._static_spin_safe(*active) or
                    not self._dynamic_spin_safe(*active)):
                continue
            self._install_planned_candidate(point, requested, result, reason)
            self.candidate_waiting = False
            self.candidate_retry_not_before = None
            self._reset_translation_progress()
            rospy.logwarn(
                "CONTINUOUS_SCAN_PAIRED_APPROACH point=%s via=(%.3f,%.3f)",
                name, result["approach_world"][0], result["approach_world"][1])
            return True

        self.candidate_waiting = True
        self.candidate_retry_not_before = now + rospy.Duration(
            self.candidate_retry_delay)
        self.path_world = []
        self.replan_fail_count = 0
        self._reset_translation_progress()
        rospy.logerr(
            "CONTINUOUS_SCAN_NO_SAFE_CANDIDATE_RETRY point=%s reason=%s "
            "retry_in=%.1fs mission_continues=true",
            name, reason, self.candidate_retry_delay)
        self.publish_zero("SCAN_POINT_WAITING_SAFE_CANDIDATE")
        return False

    def _skip_blocked_approach(self, reason):
        previous = self.current_route_point()["name"]
        if self.route_index >= len(self.route_points) - 1:
            return False
        self.route_index += 1
        self._activate_route_point(clear_path=True)
        self.scan_route_started = rospy.Time.now()
        self.last_progress_time = rospy.Time.now()
        self.plan_from_current_pose("blocked scan approach fallback", force=True)
        self._reset_translation_progress()
        rospy.logwarn(
            "SCAN_APPROACH_SKIPPED blocked=%s next=%s reason=%s",
            previous, self.current_route_point()["name"], reason)
        return True

    def _monitor_translation_progress(self):
        if self.pose is None or self.scan_active:
            return False
        if self.progress_route_index != self.route_index:
            self._reset_translation_progress()
            return False
        current = (self.pose[0], self.pose[1])
        if self.translation_progress_pose is None:
            self._reset_translation_progress()
            return False
        moved = math.hypot(current[0] - self.translation_progress_pose[0],
                           current[1] - self.translation_progress_pose[1])
        if moved >= self.translation_progress_distance:
            self.translation_progress_pose = current
            self.translation_progress_stamp = rospy.Time.now()
            return False
        if self.translation_progress_stamp is None:
            self.translation_progress_stamp = rospy.Time.now()
            return False
        stalled = (rospy.Time.now() -
                   self.translation_progress_stamp).to_sec()
        if stalled < self.translation_stuck_timeout:
            return False
        point = self.current_route_point()
        if (self.distance_to_active_goal() <=
                point["tolerance"] + self.translation_progress_distance):
            return False
        reason = "no translation progress for %.1fs" % stalled
        if point["name"] in self.scan_specs:
            self._select_alternative_scan_point(reason)
            return True
        if point["name"].endswith("_approach"):
            return self._skip_blocked_approach(reason)
        self.translation_progress_stamp = rospy.Time.now()
        return False

    def _approach_scan_name(self):
        if not self.route_ready:
            return ""
        name = self.current_route_point()["name"]
        if not name.endswith("_approach"):
            return ""
        scan_name = name[:-len("_approach")]
        return scan_name if scan_name in self.scan_specs else ""

    def _reset_approach_alignment(self):
        self.approach_align_cmd = 0.0
        self.approach_align_last = None

    def check_goal(self):
        scan_name = self._approach_scan_name() if self.v2_ready else ""
        point = self.current_route_point() if self.route_ready else None
        if (not scan_name or self.pose is None or
                self.distance_to_active_goal() > point["tolerance"]):
            self._reset_approach_alignment()
            return super(FixedPointContinuousOcrSweepRouteTestV2,
                         self).check_goal()

        now = rospy.Time.now()
        dt = (1.0 / max(self.control_rate_hz, 1.0)
              if self.approach_align_last is None else
              max(0.02, min(0.16,
                            (now - self.approach_align_last).to_sec())))
        self.approach_align_last = now
        yaw_error = wrap_angle(
            self.scan_specs[scan_name]["start"] - self.pose[2])
        remaining = abs(yaw_error)
        desired_magnitude = braking_limited_speed(
            remaining, self.approach_align_max_speed,
            self.approach_align_acceleration,
            self.approach_align_min_speed, self.route_yaw_tolerance)
        desired = (0.0 if desired_magnitude <= 0.0 else
                   math.copysign(desired_magnitude, yaw_error))
        self.approach_align_cmd = slew_rate(
            self.approach_align_cmd, desired,
            self.approach_align_acceleration, dt)

        if (remaining <= self.route_yaw_tolerance and
                abs(self.approach_align_cmd) <= 0.035):
            self._reset_approach_alignment()
            rospy.logwarn(
                "SCAN_APPROACH_ALIGNED point=%s yaw=%.1fdeg error=%.1fdeg; "
                "final_approach_will_be_front_first",
                point["name"], math.degrees(self.pose[2]),
                math.degrees(yaw_error))
            return super(FixedPointContinuousOcrSweepRouteTestV2,
                         self).check_goal()

        self.requested_vy = 0.0
        self.publish_cmd(0.0, self.approach_align_cmd)
        rospy.logwarn_throttle(
            0.45,
            "SCAN_APPROACH_ALIGN point=%s yaw=%.1fdeg error=%.1fdeg "
            "cmd_wz=%.3f",
            point["name"], math.degrees(self.pose[2]),
            math.degrees(yaw_error), self.approach_align_cmd)
        return True

    def _finish_scan(self):
        finished_index = self.route_index
        finished_name = self.current_route_point()["name"]
        self._set_scan_safety_mode(False)
        super(FixedPointContinuousOcrSweepRouteTestV2, self)._finish_scan()

        if finished_index >= len(self.route_points) - 1:
            self.finished = True
            self.route_transition_hold_until = None
            self.publish_zero("CONTINUOUS_SCAN_ROUTE_COMPLETE")
            self.log_status(
                "all continuous scan points completed; node remains stopped")
            rospy.logwarn(
                "CONTINUOUS_SCAN_ROUTE_COMPLETE last=%s points=%d",
                finished_name, len(self.scan_specs))
            return

        self.route_index += 1
        self._activate_route_point(clear_path=True)
        following = self.current_route_point()["name"]
        self.scan_route_started = rospy.Time.now()
        self.last_progress_time = rospy.Time.now()
        self._reset_approach_alignment()
        self.plan_from_current_pose(
            "atomic advance after continuous scan", force=True)
        self.route_transition_hold_until = (
            rospy.Time.now() + rospy.Duration(self.route_transition_observe_s))
        self._reset_translation_progress()
        rospy.logwarn(
            "CONTINUOUS_SCAN_ADVANCE %s -> %s old_goal_reacquire=false "
            "cone_observe_hold=%.2fs",
            finished_name, following, self.route_transition_observe_s)

    def control_loop(self, event):
        if not self.v2_ready:
            super(FixedPointContinuousOcrSweepRouteTestV2,
                  self).control_loop(event)
            return
        if self.candidate_waiting:
            now = rospy.Time.now()
            if (self.candidate_retry_not_before is not None and
                    now < self.candidate_retry_not_before):
                self.publish_zero("SCAN_POINT_CANDIDATE_RETRY_WAIT")
                return
            self.candidate_retry_not_before = None
            self._select_alternative_scan_point(
                "retry after temporary obstacle hold")
            return
        if self.route_transition_hold_until is not None:
            if rospy.Time.now() < self.route_transition_hold_until:
                self._dynamic_maintenance()
                self.publish_zero("SCAN_ROUTE_CONE_OBSERVE_HOLD")
                return
            self.route_transition_hold_until = None
            rospy.logwarn(
                "CONTINUOUS_SCAN_TRANSITION_RELEASE next=%s",
                self.current_route_point()["name"])
        if self._monitor_translation_progress():
            return
        super(FixedPointContinuousOcrSweepRouteTestV2,
              self).control_loop(event)

    def shutdown(self):
        try:
            self._set_scan_safety_mode(False, force=True)
        except Exception:
            pass
        super(FixedPointContinuousOcrSweepRouteTestV2, self).shutdown()


if __name__ == "__main__":
    FixedPointContinuousOcrSweepRouteTestV2().spin()
