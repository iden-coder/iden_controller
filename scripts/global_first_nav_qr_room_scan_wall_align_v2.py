#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Wall-aligned QR scan with active no-progress camera recovery."""

import subprocess

import rospy

from global_first_nav_qr_room_scan_wall_align_v1 import (
    WallAlignedUniqueQRScanner,
)


class RecoveringWallAlignedQRScanner(WallAlignedUniqueQRScanner):
    def __init__(self):
        super(RecoveringWallAlignedQRScanner, self).__init__()
        self.no_new_rounds_before_camera_restart = int(rospy.get_param(
            "~qr_no_new_rounds_before_camera_restart", 2))
        self.camera_node_name = rospy.get_param(
            "~qr_camera_node_name", "/ucar_camera")
        self.camera_restart_settle_s = float(rospy.get_param(
            "~qr_camera_restart_settle_s", 2.0))
        rospy.logwarn(
            "QR_RECOVERY_V2_READY no_new_restart=%d camera=%s",
            self.no_new_rounds_before_camera_restart,
            self.camera_node_name)

    def recover_no_new_round(self, no_new_rounds):
        self.publish_zero()
        rospy.logwarn(
            "QR_ACTIVE_RECOVERY no_new_rounds=%d valid=%d/%d",
            no_new_rounds, len(self.results), self.target_qr_count)
        threshold = max(1, self.no_new_rounds_before_camera_restart)
        if no_new_rounds % threshold != 0:
            return
        rospy.logwarn("QR_CAMERA_RESTART_REQUEST node=%s",
                      self.camera_node_name)
        try:
            subprocess.call(["rosnode", "kill", self.camera_node_name])
        except Exception as exc:
            rospy.logwarn("QR camera restart failed: %s", exc)
        with self.lock:
            self.latest_gray = None
            self.latest_image_time = rospy.Time(0)
        rospy.sleep(max(0.5, self.camera_restart_settle_s))
        if not self.wait_for_camera(timeout=12.0):
            rospy.logerr("QR camera did not recover; continuing safe retry")

    def run(self):
        try:
            if not self.wait_for_nav_done():
                return
            if not self.wait_for_camera():
                rospy.logerr("no camera image received; cannot scan")
                return
            self.stop_navigation_node()
            wall = 0
            round_index = 1
            round_start_count = len(self.results)
            no_new_rounds = 0
            rospy.logwarn(
                "QR_SCAN_ROUND_START round=%d valid=%d/%d",
                round_index, len(self.results), self.target_qr_count)
            while not rospy.is_shutdown():
                if self.should_stop_scan() or len(self.results) >= self.target_qr_count:
                    break
                if wall in self.valid_wall_indices:
                    result = None
                else:
                    result = self.scan_wall(wall)
                if result is not None and self.result_has_item_payload(result):
                    start_seq = self.get_control_seq()
                    self.publish_item(result)
                    if (self.after_item_decision_wait_s > 0.0 and
                            self.wait_for_decision_after_item(start_seq)):
                        break
                if self.should_stop_scan() or len(self.results) >= self.target_qr_count:
                    break

                next_wall = (wall + 1) % self.wall_count
                self.rotate_90()
                wall = next_wall
                if wall == 0:
                    gained = len(self.results) - round_start_count
                    if gained <= 0:
                        no_new_rounds += 1
                        self.recover_no_new_round(no_new_rounds)
                    else:
                        no_new_rounds = 0
                    round_index += 1
                    round_start_count = len(self.results)
                    rospy.logwarn(
                        "QR_SCAN_ROUND_START round=%d valid=%d/%d",
                        round_index, len(self.results), self.target_qr_count)
            self.publish_zero()
            self.publish_summary()
        finally:
            self.publish_zero()
            self.scanner.close()


if __name__ == "__main__":
    node = RecoveringWallAlignedQRScanner()
    node.run()
    rospy.loginfo("recovering wall-aligned QR scan finished; node stays alive")
    rospy.spin()
