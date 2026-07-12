#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Safety monitor that increases cone clearance only inside the large room."""

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped

from global_first_safety_monitor_parking_v9 import (
    ControlledRetreatParkingSafetyMonitor,
)


class IndoorProfileSafetyMonitor(ControlledRetreatParkingSafetyMonitor):
    def __init__(self):
        self.indoor_active = False
        self.parking_passthrough = True
        super(IndoorProfileSafetyMonitor, self).__init__()
        self.parking_passthrough = bool(rospy.get_param(
            "~parking_passthrough", True))
        self.indoor_trigger_y = float(rospy.get_param(
            "~indoor_trigger_y", -1.75))
        self.indoor_front_stop = float(rospy.get_param(
            "~indoor_front_stop_m", 0.31))
        self.indoor_front_critical = float(rospy.get_param(
            "~indoor_front_critical_m", 0.22))
        self.indoor_front_slow = float(rospy.get_param(
            "~indoor_front_slow_m", 0.55))
        self.indoor_side_stop = float(rospy.get_param(
            "~indoor_side_stop_m", 0.20))
        self.indoor_side_critical = float(rospy.get_param(
            "~indoor_side_critical_m", 0.15))
        self.indoor_side_avoid = float(rospy.get_param(
            "~indoor_side_avoid_m", 0.25))
        self.indoor_side_slow = float(rospy.get_param(
            "~indoor_side_slow_m", 0.35))
        self.pose_sub = rospy.Subscriber(
            rospy.get_param("~pose_topic", "/amcl_pose"),
            PoseWithCovarianceStamped, self.cb_room_pose, queue_size=1)
        rospy.logwarn(
            "SAFETY_INDOOR_PROFILE_ARMED trigger_y=%.3f parking_passthrough=%s",
            self.indoor_trigger_y, str(self.parking_passthrough))

    def cb_cmd(self, msg):
        if self.parking_passthrough and self.parking_mode_active():
            # Final parking intentionally approaches a wall. The centerline
            # controller owns wall fitting, swept-side clearance, emergency
            # stop and the final 11 cm nose gap, so normal navigation obstacle
            # rules must not reinterpret the target wall as a collision.
            self.escape_until = rospy.Time(0)
            self.pub_cmd.publish(msg)
            rospy.logwarn_throttle(
                1.0,
                "PARKING_SAFETY_PASSTHROUGH cmd=(%.3f,%.3f,%.3f) "
                "controller_owns_wall_guard=true",
                msg.linear.x, msg.linear.y, msg.angular.z)
            return
        super(IndoorProfileSafetyMonitor, self).cb_cmd(msg)

    def cb_room_pose(self, msg):
        if self.indoor_active or msg.pose.pose.position.y > self.indoor_trigger_y:
            return
        self.indoor_active = True
        self.front_stop = self.indoor_front_stop
        self.front_critical = self.indoor_front_critical
        self.front_slow = self.indoor_front_slow
        self.side_stop = self.indoor_side_stop
        self.side_critical = self.indoor_side_critical
        self.side_avoid = self.indoor_side_avoid
        self.side_slow = self.indoor_side_slow
        rospy.logwarn(
            "SAFETY_INDOOR_PROFILE_ACTIVE y=%.3f front=(%.2f,%.2f,%.2f) "
            "side=(%.2f,%.2f,%.2f,%.2f)",
            msg.pose.pose.position.y, self.front_critical, self.front_stop,
            self.front_slow, self.side_critical, self.side_stop,
            self.side_avoid, self.side_slow)


if __name__ == "__main__":
    try:
        IndoorProfileSafetyMonitor().run()
    except rospy.ROSInterruptException:
        pass
