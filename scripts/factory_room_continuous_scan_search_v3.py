#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Recoverable continuous search with candidate-path arrival gating."""

import math

import rospy

from factory_room_candidate_gate_core_v9 import candidate_gate_decision
from factory_room_continuous_scan_search_v2 import (
    RecoverableContinuousScanSearchV2,
)
from fixed_point_continuous_ocr_sweep_route_test_v2 import (
    FixedPointContinuousOcrSweepRouteTestV2,
)


class CandidateGatedContinuousScanSearchV3(
        RecoverableContinuousScanSearchV2):
    """Prevent a nearby final goal from skipping its planned approach path."""

    def __init__(self):
        # Parent constructors start the navigation timer before returning.
        self.candidate_gate_phase = ""
        self.candidate_gate_kind = ""
        self.candidate_gate_approach = None
        self.candidate_gate_final = None
        self.candidate_gate_start = None
        self.candidate_gate_last_pose = None
        self.candidate_gate_travelled = 0.0
        self.candidate_gate_required_travel = 0.0
        self.candidate_gate_approach_tolerance = 0.20
        self.candidate_gate_final_tolerance = 0.10
        self.candidate_gate_minimum_travel = 0.03
        super(CandidateGatedContinuousScanSearchV3, self).__init__()

        self.candidate_gate_approach_tolerance = float(rospy.get_param(
            "~candidate_gate_approach_tolerance_m", 0.20))
        self.candidate_gate_final_tolerance = float(rospy.get_param(
            "~candidate_gate_final_tolerance_m", 0.10))
        self.candidate_gate_minimum_travel = float(rospy.get_param(
            "~candidate_gate_minimum_travel_m", 0.03))
        rospy.logwarn(
            "FACTORY_CONTINUOUS_SEARCH_V3_READY candidate_path_gate=true "
            "approach_tol=%.3fm final_tol=%.3fm direct_min_travel=%.3fm",
            self.candidate_gate_approach_tolerance,
            self.candidate_gate_final_tolerance,
            self.candidate_gate_minimum_travel)

    def _clear_candidate_gate(self):
        self.candidate_gate_phase = ""
        self.candidate_gate_kind = ""
        self.candidate_gate_approach = None
        self.candidate_gate_final = None
        self.candidate_gate_start = None
        self.candidate_gate_last_pose = None
        self.candidate_gate_travelled = 0.0
        self.candidate_gate_required_travel = 0.0

    def _install_planned_candidate(self, point, requested, result, reason):
        super(CandidateGatedContinuousScanSearchV3,
              self)._install_planned_candidate(
                  point, requested, result, reason)
        self.candidate_gate_final = tuple(result["active_goal_world"])
        self.candidate_gate_start = (
            None if self.pose is None else (self.pose[0], self.pose[1]))
        self.candidate_gate_last_pose = self.candidate_gate_start
        self.candidate_gate_travelled = 0.0
        approach = result.get("approach_world")
        if approach is None:
            self.candidate_gate_kind = "direct"
            self.candidate_gate_phase = "to_candidate"
            self.candidate_gate_approach = None
            start_to_final = (
                0.0 if self.candidate_gate_start is None else math.hypot(
                    self.candidate_gate_final[0] -
                    self.candidate_gate_start[0],
                    self.candidate_gate_final[1] -
                    self.candidate_gate_start[1]))
            self.candidate_gate_required_travel = min(
                self.candidate_gate_minimum_travel,
                max(0.008, 0.5 * start_to_final))
        else:
            self.candidate_gate_kind = "paired"
            self.candidate_gate_phase = "to_approach"
            self.candidate_gate_approach = tuple(approach)
            self.candidate_gate_required_travel = 0.0
        rospy.logwarn(
            "CONTINUOUS_SCAN_CANDIDATE_GATE_ARMED point=%s kind=%s "
            "phase=%s final=(%.3f,%.3f)",
            point["name"], self.candidate_gate_kind,
            self.candidate_gate_phase, self.candidate_gate_final[0],
            self.candidate_gate_final[1])

    def _candidate_gate_distances(self):
        if self.pose is None:
            return None, None, 0.0
        current = (self.pose[0], self.pose[1])
        approach_distance = None
        if self.candidate_gate_approach is not None:
            approach_distance = math.hypot(
                current[0] - self.candidate_gate_approach[0],
                current[1] - self.candidate_gate_approach[1])
        final_distance = None
        if self.candidate_gate_final is not None:
            final_distance = math.hypot(
                current[0] - self.candidate_gate_final[0],
                current[1] - self.candidate_gate_final[1])
        if self.candidate_gate_start is not None:
            displacement = math.hypot(
                current[0] - self.candidate_gate_start[0],
                current[1] - self.candidate_gate_start[1])
            # Maximum displacement rejects stationary localization jitter and
            # still proves that a direct candidate caused physical motion.
            if displacement <= 0.80:
                self.candidate_gate_travelled = max(
                    self.candidate_gate_travelled, displacement)
        self.candidate_gate_last_pose = current
        return (approach_distance, final_distance,
                self.candidate_gate_travelled)

    def check_goal(self):
        if not self.candidate_gate_phase:
            return super(CandidateGatedContinuousScanSearchV3,
                         self).check_goal()
        if self.pose is None:
            return False

        approach_distance, final_distance, travelled = (
            self._candidate_gate_distances())
        minimum_travel = self.candidate_gate_required_travel
        decision = candidate_gate_decision(
            self.candidate_gate_phase, approach_distance, final_distance,
            travelled, self.candidate_gate_approach_tolerance,
            self.candidate_gate_final_tolerance, minimum_travel)

        if decision == "advance_to_candidate":
            self.candidate_gate_phase = "to_candidate"
            rospy.logwarn(
                "CONTINUOUS_SCAN_CANDIDATE_APPROACH_REACHED point=%s "
                "distance=%.3fm; final_leg_required=true",
                self.current_route_point()["name"], approach_distance)
            return False

        if decision == "candidate_reached":
            point_name = self.current_route_point()["name"]
            kind = self.candidate_gate_kind
            self._clear_candidate_gate()
            rospy.logwarn(
                "CONTINUOUS_SCAN_CANDIDATE_GATE_RELEASED point=%s kind=%s "
                "final_distance=%.3fm; spin_check_allowed=true",
                point_name, kind, final_distance)
            return super(CandidateGatedContinuousScanSearchV3,
                         self).check_goal()

        rospy.logwarn_throttle(
            0.6,
            "CONTINUOUS_SCAN_CANDIDATE_GATE_WAIT point=%s kind=%s phase=%s "
            "approach=%s final=%.3fm travelled=%.3fm",
            self.current_route_point()["name"], self.candidate_gate_kind,
            self.candidate_gate_phase,
            ("n/a" if approach_distance is None else
             "%.3fm" % approach_distance),
            float("nan") if final_distance is None else final_distance,
            travelled)
        return False

    def compute_cmd(self, target):
        if self.candidate_gate_phase == "to_approach":
            # The composite path is currently travelling away from the final
            # scan point. V2's final-yaw blend is based on that final point and
            # can otherwise bend this short approach leg into a circle.
            return super(FixedPointContinuousOcrSweepRouteTestV2,
                         self).compute_cmd(target)
        return super(CandidateGatedContinuousScanSearchV3,
                     self).compute_cmd(target)

    def _start_search(self):
        self._clear_candidate_gate()
        return super(CandidateGatedContinuousScanSearchV3,
                     self)._start_search()

    def _finish_scan(self):
        self._clear_candidate_gate()
        return super(CandidateGatedContinuousScanSearchV3,
                     self)._finish_scan()

    def _resume_after_false_handoff(self, observed_label):
        self._clear_candidate_gate()
        return super(CandidateGatedContinuousScanSearchV3,
                     self)._resume_after_false_handoff(observed_label)

    def shutdown(self):
        self._clear_candidate_gate()
        return super(CandidateGatedContinuousScanSearchV3, self).shutdown()


if __name__ == "__main__":
    CandidateGatedContinuousScanSearchV3().spin()
