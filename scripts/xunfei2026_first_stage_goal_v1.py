#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import threading

import actionlib
import rospy
import tf.transformations
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String


def angle_error(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def yaw_from_quaternion(q):
    return tf.transformations.euler_from_quaternion((q.x, q.y, q.z, q.w))[2]


def quaternion_from_yaw(yaw):
    q = tf.transformations.quaternion_from_euler(0.0, 0.0, yaw)
    return Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])


class FirstStageGoal(object):
    def __init__(self):
        self.goal_x = float(rospy.get_param("~goal_x", -1.48))
        self.goal_y = float(rospy.get_param("~goal_y", -0.45))
        self.goal_yaw = float(rospy.get_param("~goal_yaw", math.pi))
        self.initial_x = float(rospy.get_param("~initial_x", 0.0))
        self.initial_y = float(rospy.get_param("~initial_y", 0.0))
        self.initial_yaw = float(rospy.get_param("~initial_yaw", 0.0))
        self.initial_pose_publish_s = float(rospy.get_param("~initial_pose_publish_s", 3.0))
        self.localization_timeout_s = float(rospy.get_param("~localization_timeout_s", 18.0))
        self.server_timeout_s = float(rospy.get_param("~server_timeout_s", 25.0))
        self.goal_timeout_s = float(rospy.get_param("~goal_timeout_s", 120.0))
        self.start_delay_s = float(rospy.get_param("~start_delay_s", 2.0))
        self.initial_position_limit_m = float(rospy.get_param("~initial_position_limit_m", 0.60))
        self.initial_yaw_limit_deg = float(rospy.get_param("~initial_yaw_limit_deg", 55.0))

        self.lock = threading.Lock()
        self.pose = None
        self.pose_stamp = rospy.Time(0)
        self.odom_stamp = rospy.Time(0)
        self.scan_stamp = rospy.Time(0)

        self.initial_pose_pub = rospy.Publisher(
            "/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
        self.zero_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)
        self.status_pub = rospy.Publisher(
            "/xunfei2026_first_stage/status", String, queue_size=5, latch=True)
        rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_cb, queue_size=5)
        rospy.Subscriber("/odom", Odometry, self.odom_cb, queue_size=5)
        rospy.Subscriber("/scan", LaserScan, self.scan_cb, queue_size=5)

        self.client = actionlib.SimpleActionClient("/move_base", MoveBaseAction)
        rospy.on_shutdown(self.stop)

    def publish_status(self, text):
        self.status_pub.publish(String(data=text))
        rospy.logwarn("XUNFEI2026_FIRST_STAGE %s", text)

    def pose_cb(self, msg):
        with self.lock:
            self.pose = (
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                yaw_from_quaternion(msg.pose.pose.orientation),
            )
            self.pose_stamp = rospy.Time.now()

    def odom_cb(self, _msg):
        self.odom_stamp = rospy.Time.now()

    def scan_cb(self, _msg):
        self.scan_stamp = rospy.Time.now()

    def initial_pose_message(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = rospy.Time.now()
        msg.pose.pose.position.x = self.initial_x
        msg.pose.pose.position.y = self.initial_y
        msg.pose.pose.orientation = quaternion_from_yaw(self.initial_yaw)
        msg.pose.covariance[0] = 0.10
        msg.pose.covariance[7] = 0.10
        msg.pose.covariance[35] = 0.04
        return msg

    def publish_initial_pose(self):
        self.publish_status("INITIALIZING_LOCALIZATION")
        deadline = rospy.Time.now() + rospy.Duration(self.initial_pose_publish_s)
        rate = rospy.Rate(4)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            self.initial_pose_pub.publish(self.initial_pose_message())
            rate.sleep()

    def sensors_ready(self):
        now = rospy.Time.now()
        if self.odom_stamp == rospy.Time(0) or (now - self.odom_stamp).to_sec() > 1.0:
            return False
        if self.scan_stamp == rospy.Time(0) or (now - self.scan_stamp).to_sec() > 1.0:
            return False
        with self.lock:
            if self.pose is None or (now - self.pose_stamp).to_sec() > 1.0:
                return False
            x, y, yaw = self.pose
        distance = math.hypot(x - self.initial_x, y - self.initial_y)
        yaw_error = abs(math.degrees(angle_error(yaw, self.initial_yaw)))
        return distance <= self.initial_position_limit_m and yaw_error <= self.initial_yaw_limit_deg

    def wait_for_localization(self):
        deadline = rospy.Time.now() + rospy.Duration(self.localization_timeout_s)
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.sensors_ready():
                return True
            rate.sleep()
        return False

    def stop(self):
        try:
            if self.client.get_state() in (GoalStatus.PENDING, GoalStatus.ACTIVE):
                self.client.cancel_goal()
        except Exception:
            pass
        zero = Twist()
        for _ in range(8):
            self.zero_pub.publish(zero)
            rospy.sleep(0.025)

    def run(self):
        self.publish_status("WAITING_MOVE_BASE")
        if not self.client.wait_for_server(rospy.Duration(self.server_timeout_s)):
            self.publish_status("FAILED_MOVE_BASE_UNAVAILABLE")
            self.stop()
            return

        self.publish_initial_pose()
        if not self.wait_for_localization():
            self.publish_status("FAILED_LOCALIZATION_NOT_READY")
            self.stop()
            return

        with self.lock:
            x, y, yaw = self.pose
        rospy.logwarn(
            "XUNFEI2026_FIRST_STAGE localized pose=(%.3f, %.3f, %.1fdeg); "
            "goal=(%.3f, %.3f, %.1fdeg)",
            x, y, math.degrees(yaw), self.goal_x, self.goal_y,
            math.degrees(self.goal_yaw))

        self.publish_status("STARTING_IN_{:.1f}S".format(self.start_delay_s))
        rospy.sleep(self.start_delay_s)
        if rospy.is_shutdown():
            return

        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = self.goal_x
        goal.target_pose.pose.position.y = self.goal_y
        goal.target_pose.pose.orientation = quaternion_from_yaw(self.goal_yaw)

        self.publish_status("NAVIGATING")
        self.client.send_goal(goal)
        finished = self.client.wait_for_result(rospy.Duration(self.goal_timeout_s))
        if not finished:
            self.client.cancel_goal()
            self.publish_status("FAILED_GOAL_TIMEOUT")
            self.stop()
            return

        state = self.client.get_state()
        self.stop()
        if state == GoalStatus.SUCCEEDED:
            self.publish_status("SUCCEEDED")
        else:
            self.publish_status("FAILED_ACTION_STATE_{}".format(state))


def main():
    rospy.init_node("xunfei2026_first_stage_goal")
    FirstStageGoal().run()


if __name__ == "__main__":
    main()

