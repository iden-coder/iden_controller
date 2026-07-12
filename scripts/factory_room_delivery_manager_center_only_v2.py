#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Centerline-only delivery manager with no white-frame mission stage."""

import json
import time

import rospy
from std_msgs.msg import String

from factory_room_delivery_manager_center_only_v1 import (
    CenterlineFactoryDeliveryManager,
)


class NoWhiteFrameDeliveryManager(CenterlineFactoryDeliveryManager):
    def __init__(self):
        super(NoWhiteFrameDeliveryManager, self).__init__()
        # The inherited detector is not subscribed and is never used by this
        # mission. Releasing it also makes accidental future calls fail closed.
        self.square_detector = None
        rospy.logwarn("WHITE_FRAME_DETECTION_DISABLED centerline parking only")

    def detect_square_once(self):
        return None

    def mission_thread(self):
        mission_start = time.time()
        try:
            self.publish_state(
                "WAITING_FOR_TTS", item=self.target_item,
                warehouse=self.target_warehouse,
                wait_s=self.start_after_tts_s)
            rospy.sleep(max(0.0, self.start_after_tts_s))
            if not self.wait_for_move_base() or not self.wait_for_inputs():
                self.fail("房间任务所需的导航、激光或相机数据未就绪")
                return

            if self.navigate_start_pose:
                if not self.navigate(
                        "start", self.start_pose[0], self.start_pose[1],
                        self.start_pose[2], allow_offsets=False):
                    self.fail("未能安全到达大房间入口")
                    return
            else:
                rospy.logwarn("ROOM_NAV_START_SKIPPED; first goal will be d1")

            if time.time() - mission_start > self.mission_timeout_s:
                self.fail("房间任务超时，小车已安全停止")
                return
            if not self.find_target_workshop():
                self.fail("巡检观察点后仍未可靠识别到目标车间")
                return

            self.publish_zero(5)
            self.publish_state(
                "TARGET_WORKSHOP_FOUND", warehouse=self.target_warehouse)
            if not self.approach_target_wall_with_navigation():
                self.fail("找到目标车间，但未能安全进入中心线停车阶段")
                return

            self.publish_state(
                "CENTERLINE_PARKING_READY", warehouse=self.target_warehouse)
            if not self.park_inside_square():
                self.fail("厂牌中心线停车未通过安全与位置校验，小车已停止")
                return

            self.ocr_control("disable")
            final_text = "已将{}放入{}".format(
                self.target_item, self.target_warehouse)
            self.tts_pub.publish(String(data=final_text))
            payload = {
                "status": "success",
                "selected_item": self.target_item,
                "target_warehouse": self.target_warehouse,
                "broadcast_text": final_text,
                "parking_mode": "sign_centerline_and_lidar_wall",
                "nose_wall_gap_m": 0.11,
                "stamp": time.time(),
            }
            self.result_pub.publish(String(
                data=json.dumps(payload, ensure_ascii=False)))
            self.publish_state("DONE", **payload)
            rospy.logwarn("FACTORY_DELIVERY_COMPLETE %s", final_text)
            self.publish_zero(20)
        except Exception as exc:
            rospy.logerr("factory room mission exception: %s", exc)
            self.fail("房间任务发生异常，小车已安全停止：{}".format(exc))
        finally:
            self.publish_zero(10)
            with self.lock:
                self.mission_running = False


if __name__ == "__main__":
    NoWhiteFrameDeliveryManager().run()
