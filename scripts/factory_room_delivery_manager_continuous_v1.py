#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Centerline parking manager driven by the continuous-scan search node."""

import json
import threading
import time

import rospy
from std_msgs.msg import String

from factory_room_continuous_handoff_core_v1 import not_found_speech
from factory_room_delivery_manager_center_only_v3 import (
    FastClearanceCenterlineManager,
)


class ContinuousSearchParkingManager(FastClearanceCenterlineManager):
    def __init__(self):
        self.search_started = threading.Event()
        self.search_target_found = threading.Event()
        self.search_complete = threading.Event()
        self.search_aborted = threading.Event()
        self.search_status_payload = None
        self.search_control_pub = None
        super(ContinuousSearchParkingManager, self).__init__()
        self.search_status_topic = rospy.get_param(
            "~search_status_topic", "/factory_room/continuous_scan_status")
        self.search_control_topic = rospy.get_param(
            "~search_control_topic", "/factory_room/continuous_scan_control")
        self.search_start_timeout_s = float(rospy.get_param(
            "~search_start_timeout_s", 22.0))
        self.search_control_pub = rospy.Publisher(
            self.search_control_topic, String, queue_size=5, latch=True)
        rospy.Subscriber(
            self.search_status_topic, String, self.search_status_callback,
            queue_size=20)
        rospy.logwarn(
            "CONTINUOUS_SEARCH_PARKING_MANAGER_READY status=%s "
            "action_navigation=false white_frame=false",
            self.search_status_topic)

    def task_callback(self, msg):
        self.search_started.clear()
        self.search_target_found.clear()
        self.search_complete.clear()
        self.search_aborted.clear()
        self.search_status_payload = None
        super(ContinuousSearchParkingManager, self).task_callback(msg)

    def search_status_callback(self, msg):
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        with self.lock:
            running = self.mission_running
            target = self.target_warehouse
        if not running:
            return
        payload_target = str(payload.get("target_warehouse", "")).strip()
        if payload_target and payload_target != target:
            return
        state = str(payload.get("state", ""))
        self.search_status_payload = payload
        if state == "SEARCH_STARTED":
            self.search_started.set()
        elif state == "TARGET_FOUND":
            self.search_target_found.set()
        elif state == "SEARCH_COMPLETE":
            self.search_complete.set()
        elif state == "SEARCH_ABORTED":
            self.search_aborted.set()

    def request_search_stop(self):
        if self.search_control_pub is not None:
            self.search_control_pub.publish(String(data="stop"))

    def announce_search_failure(self, reason):
        self.request_search_stop()
        self.move_client.cancel_all_goals()
        self.ocr_control("disable")
        self.publish_zero(20)
        speech = not_found_speech(self.target_warehouse)
        self.tts_pub.publish(String(data=speech))
        payload = {
            "status": "error",
            "reason": reason,
            "selected_item": self.target_item,
            "target_warehouse": self.target_warehouse,
            "broadcast_text": speech,
            "stopped_in_place": True,
            "stamp": time.time(),
        }
        self.result_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))
        self.publish_state("SEARCH_COMPLETE_NO_TARGET", **payload)
        rospy.logerr("FACTORY_DELIVERY_SEARCH_FAILED %s", reason)

    def publish_success(self):
        self.ocr_control("disable")
        final_text = "已将{}放入{}".format(
            self.target_item, self.target_warehouse)
        self.tts_pub.publish(String(data=final_text))
        payload = {
            "status": "success",
            "selected_item": self.target_item,
            "target_warehouse": self.target_warehouse,
            "broadcast_text": final_text,
            "parked_by_ocr_centerline": True,
            "stamp": time.time(),
        }
        self.result_pub.publish(String(
            data=json.dumps(payload, ensure_ascii=False)))
        self.publish_state("DONE", **payload)
        rospy.logwarn("FACTORY_DELIVERY_COMPLETE %s", final_text)
        self.publish_zero(20)

    def mission_thread(self):
        mission_start = time.time()
        startup_deadline = (
            mission_start + self.start_after_tts_s +
            self.search_start_timeout_s)
        mission_deadline = mission_start + self.mission_timeout_s
        try:
            self.publish_state(
                "WAITING_FOR_CONTINUOUS_SEARCH",
                item=self.target_item, warehouse=self.target_warehouse,
                tts_wait_s=self.start_after_tts_s)
            rate = rospy.Rate(10)
            while not rospy.is_shutdown():
                now = time.time()
                if self.search_target_found.is_set():
                    break
                if self.search_complete.is_set():
                    self.announce_search_failure(
                        "三个扫描点均已检查，仍未识别到目标车间")
                    return
                if self.search_aborted.is_set():
                    self.announce_search_failure(
                        "车间搜索被安全停止，未识别到目标车间")
                    return
                if (not self.search_started.is_set() and
                        now >= startup_deadline):
                    self.announce_search_failure(
                        "连续扫描模块未按时启动")
                    return
                if now >= mission_deadline:
                    self.announce_search_failure(
                        "车间搜索达到安全时限")
                    return
                rate.sleep()

            self.publish_zero(8)
            self.publish_state(
                "TARGET_WORKSHOP_FOUND",
                warehouse=self.target_warehouse,
                scan_status=self.search_status_payload)
            if not self.wait_for_inputs(timeout_s=8.0):
                self.fail("已找到目标车间，但停车传感器数据未就绪")
                return
            if not self.approach_target_wall_with_navigation():
                self.fail("已找到目标车间，但未能安全接管停车")
                return
            # Center-only mode deliberately bypasses white-frame detection.
            if self.acquire_square(timeout_s=0.1) is None:
                self.fail("中心线停车模式未能启动")
                return
            if not self.park_inside_square():
                self.fail("目标车间中心线停车未通过安全与位置校验")
                return
            self.publish_success()
        except Exception as exc:
            rospy.logerr("continuous delivery exception: %s", exc)
            self.request_search_stop()
            self.fail("房间任务发生异常，小车已安全停车：{}".format(exc))
        finally:
            self.publish_zero(10)
            with self.lock:
                self.mission_running = False


if __name__ == "__main__":
    ContinuousSearchParkingManager().run()
