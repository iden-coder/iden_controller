#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Keep universal filtering, but switch clearances after room entry."""

import math

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool

from global_first_safety_monitor_omni_v1 import DirectionalOmniSafetyMonitor
from global_first_safety_monitor_parking_v5 import clamp
from room_lidar_semantics_v1 import classify_scan


class TwoStageProfileSafetyMonitor(DirectionalOmniSafetyMonitor):
    def __init__(self):
        self.room_profile_active = False
        self.final_approach_active = False
        self.slalom_active = False
        self.semantic_cone_front = float("inf")
        self.semantic_cone_left = float("inf")
        self.semantic_cone_right = float("inf")
        self.semantic_cone_turn = float("inf")
        self.semantic_wall_front = float("inf")
        self.semantic_wall_left = float("inf")
        self.semantic_wall_right = float("inf")
        self.room_cone_single_pass = False
        self.cone_front_lateral_escape = 0.040
        self.cone_front_auto_escape_side_clear = 0.36
        self.cone_front_auto_escape_side_bias = 0.08
        self.cone_turn_hard_lateral_escape = 0.055
        self.cone_turn_hard_escape_side_clear = 0.32
        self.cone_turn_hard_auto_escape_side_bias = 0.08
        super(TwoStageProfileSafetyMonitor, self).__init__()
        # Directional filtering is active from startup, exactly as in the
        # accepted universal first-stage test.  room_profile_active controls
        # only the one-way threshold swap below.
        self.indoor_active = True
        self.cone_base_pad = float(rospy.get_param(
            "~room_cone_base_pad_m", 0.10))
        self.room_cone_single_pass = bool(rospy.get_param(
            "~room_cone_single_pass_filter", False))
        self.wall_front_relax = float(rospy.get_param(
            "~room_wall_front_relax_m", 0.09))
        self.wall_side_relax = float(rospy.get_param(
            "~room_wall_side_relax_m", 0.05))
        self.final_wall_front_relax = float(rospy.get_param(
            "~final_wall_front_relax_m", 0.17))
        self.final_wall_side_relax = float(rospy.get_param(
            "~final_wall_side_relax_m", 0.10))
        self.cone_turn_stop = float(rospy.get_param(
            "~cone_turn_stop_m", 0.28))
        self.cone_turn_slow = float(rospy.get_param(
            "~cone_turn_slow_m", 0.42))
        self.cone_turn_hard_stop = float(rospy.get_param(
            "~cone_turn_hard_stop_m", 0.18))
        self.cone_turn_escape_rps = float(rospy.get_param(
            "~cone_turn_escape_rps", 0.22))
        self.cone_turn_hard_lateral_escape = abs(float(rospy.get_param(
            "~cone_turn_hard_lateral_escape_mps", 0.055)))
        self.cone_turn_hard_escape_side_clear = float(rospy.get_param(
            "~cone_turn_hard_escape_side_clear_m", 0.32))
        self.cone_turn_hard_auto_escape_side_bias = abs(float(rospy.get_param(
            "~cone_turn_hard_auto_escape_side_bias_m", 0.08)))
        self.cone_front_stop = float(rospy.get_param(
            "~cone_front_stop_m", 0.24))
        self.cone_front_slow = float(rospy.get_param(
            "~cone_front_slow_m", 0.42))
        self.cone_front_lateral_escape = abs(float(rospy.get_param(
            "~cone_front_lateral_escape_mps", 0.040)))
        self.cone_front_auto_escape_side_clear = float(rospy.get_param(
            "~cone_front_auto_escape_side_clear_m", 0.36))
        self.cone_front_auto_escape_side_bias = float(rospy.get_param(
            "~cone_front_auto_escape_side_bias_m", 0.08))
        self.cone_side_stop = float(rospy.get_param(
            "~cone_side_stop_m", 0.20))
        self.cone_side_slow = float(rospy.get_param(
            "~cone_side_slow_m", 0.32))
        self.final_front_critical = float(rospy.get_param(
            "~final_wall_front_critical_m", 0.09))
        self.final_front_stop = float(rospy.get_param(
            "~final_wall_front_stop_m", 0.12))
        self.final_front_slow = float(rospy.get_param(
            "~final_wall_front_slow_m", 0.24))
        self.final_side_critical = float(rospy.get_param(
            "~final_wall_side_critical_m", 0.10))
        self.final_side_stop = float(rospy.get_param(
            "~final_wall_side_stop_m", 0.12))
        self.final_side_slow = float(rospy.get_param(
            "~final_wall_side_slow_m", 0.20))
        self.slalom_cone_front_stop = float(rospy.get_param(
            "~slalom_cone_front_stop_m", 0.14))
        self.slalom_cone_front_slow = float(rospy.get_param(
            "~slalom_cone_front_slow_m", 0.30))
        self.slalom_cone_turn_stop = float(rospy.get_param(
            "~slalom_cone_turn_stop_m", 0.24))
        self.slalom_cone_turn_slow = float(rospy.get_param(
            "~slalom_cone_turn_slow_m", 0.36))
        self.slalom_cone_side_stop = float(rospy.get_param(
            "~slalom_cone_side_stop_m", 0.17))
        self.slalom_cone_side_slow = float(rospy.get_param(
            "~slalom_cone_side_slow_m", 0.28))
        self.final_sub = rospy.Subscriber(
            rospy.get_param("~final_approach_topic",
                            "/two_stage/final_approach"),
            Bool, self.cb_final_approach, queue_size=1)
        self.slalom_sub = rospy.Subscriber(
            rospy.get_param("~room_slalom_topic", "/two_stage/slalom_active"),
            Bool, self.cb_slalom, queue_size=1)
        rospy.logwarn(
            "TWO_STAGE_SAFETY_ARMED trigger_y=%.3f first_stage_unchanged=true "
            "room_cone_single_pass=%s",
            self.indoor_trigger_y, str(self.room_cone_single_pass).lower())

    def cb_final_approach(self, msg):
        active = bool(msg.data)
        if active == self.final_approach_active:
            return
        self.final_approach_active = active
        if active:
            self.front_critical = self.final_front_critical
            self.front_stop = self.final_front_stop
            self.front_slow = self.final_front_slow
            self.side_critical = self.final_side_critical
            self.side_stop = self.final_side_stop
            self.side_slow = self.final_side_slow
            rospy.logwarn(
                "TWO_STAGE_SAFETY_FINAL_ACTIVE wall_front=(%.2f,%.2f,%.2f) "
                "cone_protection_preserved=true",
                self.front_critical, self.front_stop, self.front_slow)
        elif self.room_profile_active:
            self.front_critical = self.indoor_front_critical
            self.front_stop = self.indoor_front_stop
            self.front_slow = self.indoor_front_slow
            self.side_critical = self.indoor_side_critical
            self.side_stop = self.indoor_side_stop
            self.side_slow = self.indoor_side_slow

    def cb_slalom(self, msg):
        active = bool(msg.data)
        if active == self.slalom_active:
            return
        self.slalom_active = active
        rospy.logwarn("TWO_STAGE_SAFETY_SLALOM_%s",
                      "ACTIVE" if active else "CLEAR")

    def cb_room_pose(self, msg):
        if (getattr(self, "room_profile_active", False) or
                msg.pose.pose.position.y > self.indoor_trigger_y):
            return
        self.room_profile_active = True
        self.indoor_active = True
        self.front_stop = self.indoor_front_stop
        self.front_critical = self.indoor_front_critical
        self.front_slow = self.indoor_front_slow
        self.side_stop = self.indoor_side_stop
        self.side_critical = self.indoor_side_critical
        self.side_avoid = self.indoor_side_avoid
        self.side_slow = self.indoor_side_slow
        rospy.logwarn(
            "TWO_STAGE_SAFETY_ROOM_ACTIVE y=%.3f front=(%.2f,%.2f,%.2f) "
            "side=(%.2f,%.2f,%.2f,%.2f)",
            msg.pose.pose.position.y, self.front_critical, self.front_stop,
            self.front_slow, self.side_critical, self.side_stop,
            self.side_avoid, self.side_slow)

    def cb_scan(self, msg):
        super(TwoStageProfileSafetyMonitor, self).cb_scan(msg)
        if not getattr(self, "room_profile_active", False):
            return
        wall_front_relax = (self.final_wall_front_relax
                            if self.final_approach_active
                            else self.wall_front_relax)
        wall_side_relax = (self.final_wall_side_relax
                           if self.final_approach_active
                           else self.wall_side_relax)
        semantic = classify_scan(
            msg, cone_base_pad_m=self.cone_base_pad,
            front_angle_deg=48.0,
            side_min_deg=self.side_angle_min_deg,
            side_max_deg=self.side_angle_max_deg,
            wall_front_relax_m=wall_front_relax,
            wall_side_relax_m=wall_side_relax)
        if self.final_approach_active:
            cone_raw = semantic["cone_front"] + self.cone_base_pad
            if (cone_raw < float("inf") and
                    semantic["wall_front"] < float("inf") and
                    abs(cone_raw - semantic["wall_front"]) <= 0.09):
                semantic["cone_front"] = float("inf")
                semantic["effective_front"] = (
                    semantic["wall_front"] + self.final_wall_front_relax)
        if (self.slalom_active and
                semantic["cone_front"] < float("inf") and
                semantic["cone_front"] > self.slalom_cone_front_stop and
                (semantic["wall_front"] == float("inf") or
                 semantic["wall_front"] > semantic["cone_front"] + 0.08)):
            semantic["effective_front"] = max(
                semantic["effective_front"], self.front_stop + 0.035)
        self.semantic_cone_front = semantic["cone_front"]
        self.semantic_cone_left = semantic["cone_left"]
        self.semantic_cone_right = semantic["cone_right"]
        self.semantic_cone_turn = semantic["cone_turn"]
        self.semantic_wall_front = semantic["wall_front"] + wall_front_relax
        self.semantic_wall_left = semantic["wall_left"] + wall_side_relax
        self.semantic_wall_right = semantic["wall_right"] + wall_side_relax
        self.min_front = semantic["effective_front"]
        self.min_left = semantic["effective_left"]
        self.min_right = semantic["effective_right"]
        rospy.logwarn_throttle(
            0.8, "TWO_STAGE_SAFETY_SEMANTIC effective=(%.2f,%.2f,%.2f) "
            "cone_front=%.2f wall_front=%.2f",
            self.min_front, self.min_left, self.min_right,
            semantic["cone_front"], semantic["wall_front"])

    def cb_cmd(self, msg):
        if not self.room_profile_active:
            super(TwoStageProfileSafetyMonitor, self).cb_cmd(msg)
            return
        out = Twist()
        out.linear.x = msg.linear.x
        out.linear.y = msg.linear.y
        out.angular.z = msg.angular.z
        action = ""

        if abs(out.angular.z) > 1.0e-4:
            if self.slalom_active:
                turn_stop = self.slalom_cone_turn_stop
                turn_slow = self.slalom_cone_turn_slow
            else:
                turn_stop = 0.20 if self.final_approach_active else self.cone_turn_stop
                turn_slow = self.cone_turn_slow
            if self.semantic_cone_turn <= self.cone_turn_hard_stop:
                out.angular.z = 0.0
                action = "CONE_TURN_HARD_STOP"
                required_clearance = max(
                    self.cone_turn_hard_escape_side_clear,
                    self.cone_side_stop + 0.03)
                if (abs(out.linear.y) > 1.0e-4 and
                        not self.final_approach_active):
                    escape_clearance = (self.min_left
                                        if out.linear.y > 0.0
                                        else self.min_right)
                    if escape_clearance >= required_clearance:
                        out.linear.x = 0.0
                        out.linear.y = math.copysign(
                            max(abs(out.linear.y),
                                self.cone_turn_hard_lateral_escape),
                            out.linear.y)
                        action = "CONE_TURN_HARD_LATERAL_ESCAPE"
                elif not self.final_approach_active:
                    left_clear = self.min_left
                    right_clear = self.min_right
                    bias = self.cone_turn_hard_auto_escape_side_bias
                    escape_y = 0.0
                    if (left_clear >= required_clearance and
                            math.isfinite(right_clear) and
                            left_clear >= right_clear + bias):
                        escape_y = self.cone_turn_hard_lateral_escape
                    elif (right_clear >= required_clearance and
                            math.isfinite(left_clear) and
                            right_clear >= left_clear + bias):
                        escape_y = -self.cone_turn_hard_lateral_escape
                    if abs(escape_y) > 1.0e-4:
                        out.linear.x = 0.0
                        out.linear.y = escape_y
                        action = "CONE_TURN_HARD_AUTO_LATERAL_ESCAPE"
            elif self.semantic_cone_turn <= turn_stop:
                sign = 1.0 if out.angular.z >= 0.0 else -1.0
                out.angular.z = sign * min(abs(out.angular.z),
                                           self.cone_turn_escape_rps)
                action = "CONE_TURN_ESCAPE"
            elif self.semantic_cone_turn < turn_slow:
                ratio = clamp(
                    (self.semantic_cone_turn - turn_stop) /
                    max(turn_slow - turn_stop, 1.0e-3), 0.18, 1.0)
                out.angular.z *= ratio
                action = "CONE_TURN_SLOW"

        if out.linear.x > 0.0:
            front_stop = (self.slalom_cone_front_stop if self.slalom_active
                          else self.cone_front_stop)
            front_slow = (self.slalom_cone_front_slow if self.slalom_active
                          else self.cone_front_slow)
            if self.semantic_cone_front <= front_stop:
                out.linear.x = 0.0
                action = "CONE_FRONT_STOP"
                auto_lateral_escape = False
                if (abs(out.linear.y) <= 1.0e-4 and
                        not self.final_approach_active):
                    left_clear = self.min_left
                    right_clear = self.min_right
                    if (left_clear >=
                            max(self.cone_front_auto_escape_side_clear,
                                right_clear +
                                self.cone_front_auto_escape_side_bias)):
                        out.linear.y = self.cone_front_lateral_escape
                    elif (right_clear >=
                          max(self.cone_front_auto_escape_side_clear,
                              left_clear +
                              self.cone_front_auto_escape_side_bias)):
                        out.linear.y = -self.cone_front_lateral_escape
                    if abs(out.linear.y) > 1.0e-4:
                        out.angular.z = 0.0
                        auto_lateral_escape = True
                side_clear = (self.semantic_cone_left
                              if out.linear.y > 0.0
                              else self.semantic_cone_right)
                if (abs(out.linear.y) > 1.0e-4 and
                        side_clear > self.cone_side_stop + 0.025):
                    out.linear.y = math.copysign(
                        max(abs(out.linear.y),
                            self.cone_front_lateral_escape),
                        out.linear.y)
                    # Translating away is safer than pivoting the rectangular
                    # chassis around a cone that is already at its nose.
                    out.angular.z = 0.0
                    action = ("CONE_FRONT_AUTO_LATERAL_ESCAPE"
                              if auto_lateral_escape else
                              "CONE_FRONT_LATERAL_ESCAPE")
            elif self.semantic_cone_front < front_slow:
                ratio = clamp(
                    (self.semantic_cone_front - front_stop) /
                    max(front_slow - front_stop, 1.0e-3),
                    0.20, 1.0)
                out.linear.x *= ratio
                action = "CONE_FRONT_SLOW"

        cone_side = (self.semantic_cone_left if out.linear.y > 0.0
                     else self.semantic_cone_right)
        if abs(out.linear.y) > 1.0e-4:
            side_stop = (self.slalom_cone_side_stop if self.slalom_active
                         else self.cone_side_stop)
            side_slow = (self.slalom_cone_side_slow if self.slalom_active
                         else self.cone_side_slow)
            if cone_side <= side_stop:
                out.linear.y = 0.0
                action = "CONE_SIDE_STOP"
            elif cone_side < side_slow:
                ratio = clamp(
                    (cone_side - side_stop) /
                    max(side_slow - side_stop, 1.0e-3),
                    0.20, 1.0)
                out.linear.y *= ratio
                action = "CONE_SIDE_SLOW"

        if action:
            rospy.logwarn_throttle(
                0.5, "TWO_STAGE_SWEEP_GUARD %s cone=(front=%.2f turn=%.2f "
                "left=%.2f right=%.2f)", action,
                self.semantic_cone_front, self.semantic_cone_turn,
                self.semantic_cone_left, self.semantic_cone_right)
        if not self.room_cone_single_pass:
            super(TwoStageProfileSafetyMonitor, self).cb_cmd(out)
            return

        # The semantic cone guard above has already applied the cone stop or
        # slowdown once. Let the inherited directional layer continue guarding
        # walls without multiplying the same cone slowdown a second time.
        saved_ranges = (self.min_front, self.min_left, self.min_right)
        self.min_front = self.semantic_wall_front
        self.min_left = self.semantic_wall_left
        self.min_right = self.semantic_wall_right
        try:
            DirectionalOmniSafetyMonitor.cb_cmd(self, out)
        finally:
            self.min_front, self.min_left, self.min_right = saved_ranges


if __name__ == "__main__":
    try:
        TwoStageProfileSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
