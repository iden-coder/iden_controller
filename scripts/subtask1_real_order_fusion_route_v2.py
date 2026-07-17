#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Spark-only order fusion with non-terminal, backoff-based inference retry."""

import threading

import rospy

from subtask1_real_order_fusion_route import (
    STATE_NAV_AND_SCAN,
    STATE_REASONING,
    Subtask1RealOrderFusion,
)


class ResilientSparkOrderFusion(Subtask1RealOrderFusion):
    def __init__(self):
        self.final_decision_in_progress = False
        super(ResilientSparkOrderFusion, self).__init__()
        self.spark_retry_delay_s = float(rospy.get_param(
            "~spark_retry_delay_s", 6.0))
        self.spark_retry_max_delay_s = float(rospy.get_param(
            "~spark_retry_max_delay_s", 24.0))
        rospy.loginfo(
            "SUBTASK1_SPARK_V2_READY terminal_failure=false retry=(%.1f..%.1f)s",
            self.spark_retry_delay_s, self.spark_retry_max_delay_s)

    def try_decide(self):
        with self.lock:
            if self.decision_done or self.final_decision_in_progress:
                return
            voice_text = self.voice_text
            category = self.target_category
            items = list(self.qr_items)
            evidence = list(self.qr_evidence)
            summary = dict(self.qr_summary) if self.qr_summary else {}
            stream_busy = self.stream_decision_in_progress
        if not voice_text or stream_busy:
            return

        physical_count = int(summary.get(
            "detected_count", len(evidence) if evidence else len(items)))
        if physical_count < self.required_item_count:
            rospy.loginfo_throttle(
                5.0, "waiting physical QR codes: %d/%d items=%s",
                physical_count, self.required_item_count, items)
            return
        if not items:
            rospy.logwarn_throttle(
                5.0, "QR physical count complete but no candidate text yet")
            return

        with self.lock:
            if self.decision_done or self.final_decision_in_progress:
                return
            self.final_decision_in_progress = True
        self.publish_state(STATE_REASONING)
        threading.Thread(
            target=self.decide_thread,
            args=(voice_text, category, items, evidence),
            daemon=True).start()

    def decide_thread(self, voice_text, category, items, evidence=None):
        evidence = evidence or []
        delay = max(1.0, self.spark_retry_delay_s)
        attempt = 0
        try:
            while not rospy.is_shutdown():
                attempt += 1
                resolved_category = category or self.extract_target_category(
                    voice_text)
                if not resolved_category:
                    resolved_category = self.infer_category_with_spark_or_local(
                        voice_text, items)
                if not self.spark_ready():
                    rospy.logerr_throttle(
                        5.0, "SPARK_RETRY_WAIT credentials unavailable")
                elif resolved_category:
                    rospy.logwarn(
                        "SPARK_RESILIENT_ATTEMPT attempt=%d category=%s items=%s",
                        attempt, resolved_category, items)
                    selected = self.select_item_with_spark(
                        resolved_category, items, evidence)
                    if selected:
                        with self.lock:
                            if self.decision_done:
                                return
                            self.decision_done = True
                        self.finish_success(
                            voice_text, resolved_category, selected, items)
                        return
                else:
                    rospy.logwarn(
                        "SPARK_RETRY_NO_CATEGORY attempt=%d voice=%s",
                        attempt, voice_text)

                self.publish_state(STATE_REASONING)
                rospy.logwarn(
                    "SPARK_RESILIENT_RETRY delay=%.1fs attempt=%d",
                    delay, attempt)
                rospy.sleep(delay)
                delay = min(self.spark_retry_max_delay_s, delay * 1.6)
        finally:
            with self.lock:
                self.final_decision_in_progress = False
                retry_final = not self.decision_done
            if retry_final and not rospy.is_shutdown():
                self.publish_state(STATE_NAV_AND_SCAN)
                self.try_decide()


if __name__ == "__main__":
    ResilientSparkOrderFusion()
    rospy.spin()
