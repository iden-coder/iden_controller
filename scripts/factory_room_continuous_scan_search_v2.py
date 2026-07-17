#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Continuous search with edge rejection and recoverable OCR handoff."""

import json

import rospy

from factory_room_continuous_handoff_core_v1 import (
    canonical_workshop,
    ocr_matches_target,
)
from factory_room_continuous_scan_search_v1 import (
    FactoryRoomContinuousScanSearchV1,
)
from factory_room_false_handoff_core_v8 import (
    handoff_bbox_is_usable,
    resume_transition,
)
from fixed_point_continuous_ocr_sweep_route_test_v2 import (
    FixedPointContinuousOcrSweepRouteTestV2,
)


class RecoverableContinuousScanSearchV2(FactoryRoomContinuousScanSearchV1):
    RESUME_COMMAND = "resume_after_false_handoff"

    def __init__(self):
        self.handoff_edge_margin_px = 12.0
        self.handoff_min_bbox_width_px = 55.0
        super(RecoverableContinuousScanSearchV2, self).__init__()
        self.handoff_edge_margin_px = float(rospy.get_param(
            "~handoff_edge_margin_px", 12.0))
        self.handoff_min_bbox_width_px = float(rospy.get_param(
            "~handoff_min_bbox_width_px", 55.0))
        rospy.logwarn(
            "FACTORY_CONTINUOUS_SEARCH_V2_READY edge_margin=%.1fpx "
            "min_bbox=%.1fpx false_handoff_resume=true",
            self.handoff_edge_margin_px, self.handoff_min_bbox_width_px)

    def ocr_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return super(RecoverableContinuousScanSearchV2,
                         self).ocr_callback(msg)
        with self.integration_lock:
            active = (self.integration_phase == self.ACTIVE and
                      self.scan_active)
            target = self.target_warehouse
        if (active and ocr_matches_target(payload, target) and
                not handoff_bbox_is_usable(
                    payload.get("bbox"), payload.get("image_width", 0),
                    self.handoff_edge_margin_px,
                    self.handoff_min_bbox_width_px)):
            # Preserve sweep diagnostics but bypass V1's parking handoff.
            FixedPointContinuousOcrSweepRouteTestV2.ocr_callback(self, msg)
            rospy.logwarn_throttle(
                0.35,
                "OCR_TARGET_EDGE_REJECTED label=%s bbox=%s width=%s; "
                "sweep_continues=true",
                target, payload.get("bbox"), payload.get("image_width"))
            return
        return super(RecoverableContinuousScanSearchV2,
                     self).ocr_callback(msg)

    def _resume_after_false_handoff(self, observed_label):
        with self.integration_lock:
            if self.integration_phase != self.HANDOFF:
                return False
            transition = resume_transition(
                self.route_index, len(self.route_points))
            if transition == "invalid":
                return False
            point = self.current_route_point()["name"]
            observed = canonical_workshop(observed_label)
            if observed:
                self.ocr_confirmed_non_targets.add(observed)
            self.ocr_candidate_hold_until = None
            self.ocr_candidate_label = ""
            self.ocr_ignore_label = observed
            self.ocr_ignore_until = (
                rospy.Time.now() + rospy.Duration(
                    max(0.5, self.ocr_non_target_ignore_s)))
            self.finished = False
            self.search_completion_reported = False

        self._set_scan_safety_mode(False)
        self._set_ocr(False)
        self.publish_zero("FALSE_HANDOFF_RESUME_STOP")

        # Keep the integration phase at HANDOFF while the route transition is
        # mutated so the navigation timer cannot publish concurrently.
        FixedPointContinuousOcrSweepRouteTestV2._finish_scan(self)

        if transition == "complete":
            with self.integration_lock:
                self.integration_phase = self.HOLD
                self.search_completion_reported = True
            self.publish_integration_status(
                "SEARCH_COMPLETE", last_point=point,
                rejected_false_handoff=observed)
            rospy.logwarn(
                "FALSE_HANDOFF_RECOVERY_COMPLETE point=%s observed=%s",
                point, observed)
            return True

        with self.integration_lock:
            self.integration_phase = self.ACTIVE
            following = self.current_route_point()["name"]
        self.publish_integration_status(
            "SEARCH_RESUMED_AFTER_FALSE_HANDOFF",
            rejected_point=point, observed_label=observed,
            next_point=following)
        rospy.logwarn(
            "FALSE_HANDOFF_SEARCH_RESUMED rejected=%s observed=%s next=%s",
            point, observed, following)
        return True

    def control_callback(self, msg):
        raw = str(msg.data).strip()
        try:
            payload = json.loads(raw)
        except Exception:
            payload = None
        if (isinstance(payload, dict) and
                str(payload.get("command", "")).strip().lower() ==
                self.RESUME_COMMAND):
            self._resume_after_false_handoff(
                str(payload.get("observed_label", "")))
            return
        return super(RecoverableContinuousScanSearchV2,
                     self).control_callback(msg)


if __name__ == "__main__":
    RecoverableContinuousScanSearchV2().spin()

