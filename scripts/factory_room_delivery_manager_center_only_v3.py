#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Centerline mission with a proven first-nav handoff through the room door."""

import subprocess
import time

import rospy
from std_msgs.msg import String

from factory_room_delivery_manager_center_only_v2 import (
    NoWhiteFrameDeliveryManager,
)


class FirstNavEntryDeliveryManager(NoWhiteFrameDeliveryManager):
    def __init__(self):
        self.entry_process = None
        self.entry_status = ""
        self.entry_done = False
        super(FirstNavEntryDeliveryManager, self).__init__()
        self.entry_launch_pkg = rospy.get_param(
            "~entry_launch_pkg", "iden_controller")
        self.entry_launch_file = rospy.get_param(
            "~entry_launch_file", "global_first_graph_nav_room_entry_v1.launch")
        self.entry_node_name = rospy.get_param(
            "~entry_node_name", "/global_first_graph_nav_room_entry")
        self.entry_status_topic = rospy.get_param(
            "~entry_status_topic", "/global_first_graph_nav_room_entry/status")
        self.entry_timeout = float(rospy.get_param(
            "~entry_timeout_s", 150.0))
        self.entry_map_yaml = rospy.get_param("~entry_map_yaml", "")
        self.entry_status_sub = rospy.Subscriber(
            self.entry_status_topic, String, self.entry_status_callback,
            queue_size=20)
        rospy.logwarn(
            "FIRST_NAV_ROOM_ENTRY_READY launch=%s timeout=%.1fs",
            self.entry_launch_file, self.entry_timeout)

    def entry_status_callback(self, msg):
        self.entry_status = msg.data or ""

    @staticmethod
    def kill_node(node_name):
        try:
            subprocess.call(
                ["rosnode", "kill", node_name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def stop_entry_navigation(self):
        self.kill_node(self.entry_node_name)
        process = self.entry_process
        self.entry_process = None
        if process is not None and process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=3.0)
            except Exception:
                try:
                    process.kill()
                except Exception:
                    pass
        self.publish_zero(8)

    def navigate_room_entry_with_first_nav(self, x, y, yaw):
        self.stop_entry_navigation()
        # The QR navigator intentionally stays alive after scanning. It must no
        # longer publish zero velocity while the entry navigator takes control.
        self.kill_node("/global_first_graph_nav")
        self.entry_status = ""
        command = [
            "roslaunch", self.entry_launch_pkg, self.entry_launch_file,
            "goal_x:={:.3f}".format(x),
            "goal_y:={:.3f}".format(y),
            "goal_yaw:={:.10f}".format(yaw),
        ]
        if self.entry_map_yaml:
            command.append("map_yaml:={}".format(self.entry_map_yaml))
        self.publish_state(
            "FIRST_NAV_ENTERING_ROOM", x=x, y=y, yaw=yaw)
        rospy.logwarn(
            "FIRST_NAV_ROOM_ENTRY_START x=%.3f y=%.3f yaw=%.1fdeg",
            x, y, yaw * 180.0 / 3.141592653589793)
        try:
            self.entry_process = subprocess.Popen(command)
        except Exception as exc:
            rospy.logerr("FIRST_NAV_ROOM_ENTRY_LAUNCH_FAILED: %s", exc)
            return False

        started = time.time()
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            status = self.entry_status.lower()
            if "goal reached" in status:
                self.entry_done = True
                rospy.logwarn("FIRST_NAV_ROOM_ENTRY_REACHED status=%s",
                              self.entry_status)
                self.stop_entry_navigation()
                rospy.sleep(0.6)
                return True
            if self.entry_process is not None and self.entry_process.poll() is not None:
                rospy.logerr(
                    "FIRST_NAV_ROOM_ENTRY_PROCESS_EXIT code=%s status=%s",
                    self.entry_process.returncode, self.entry_status)
                self.stop_entry_navigation()
                return False
            if time.time() - started > self.entry_timeout:
                rospy.logerr(
                    "FIRST_NAV_ROOM_ENTRY_TIMEOUT last_status=%s",
                    self.entry_status)
                self.stop_entry_navigation()
                return False
            rate.sleep()
        self.stop_entry_navigation()
        return False

    def navigate(self, name, x, y, yaw, allow_offsets=True):
        if name == "start" and not self.entry_done:
            return self.navigate_room_entry_with_first_nav(x, y, yaw)
        return super(FirstNavEntryDeliveryManager, self).navigate(
            name, x, y, yaw, allow_offsets=allow_offsets)

    def shutdown(self):
        self.stop_entry_navigation()
        super(FirstNavEntryDeliveryManager, self).shutdown()


if __name__ == "__main__":
    FirstNavEntryDeliveryManager().run()
