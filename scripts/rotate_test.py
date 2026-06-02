#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time


class RotateTest:
    def __init__(self):
        rospy.init_node('rotate_test', anonymous=True)

        self.angular_speed = rospy.get_param('~angular_speed', 0.8)  # rad/s
        self.target_angle = rospy.get_param('~target_angle', math.pi / 2)  # 90 deg
        self.rounds = rospy.get_param('~rounds', 4)
        self.sleep_time = rospy.get_param('~sleep_time', 0.5)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        self.current_yaw = 0.0
        self.yaw_received = False

        rospy.loginfo("RotateTest: waiting for odom...")
        while not self.yaw_received and not rospy.is_shutdown():
            rospy.sleep(0.1)
        rospy.loginfo("RotateTest: odom received, starting rotations.")

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.current_yaw = math.atan2(siny, cosy)
        self.yaw_received = True

    def rotate_by(self, angle):
        """Rotate by the given angle (radians). Positive = CCW.
        Uses fast speed for most of the rotation, then decelerates
        near the target to avoid overshoot."""
        start_yaw = self.current_yaw
        target_yaw = self._normalize(start_yaw + angle)
        direction = 1.0 if angle > 0 else -1.0

        rospy.loginfo("Rotating from %.2f to %.2f (delta=%.2f deg)",
                      start_yaw * 180 / math.pi,
                      target_yaw * 180 / math.pi,
                      angle * 180 / math.pi)

        rate = rospy.Rate(20)
        cmd = Twist()
        slow_zone = 0.3  # rad (~17 deg): start decelerating inside this window
        min_speed = 0.15  # rad/s: match my_planner fine-adjustment speed

        while not rospy.is_shutdown():
            diff = self._angle_diff(target_yaw, self.current_yaw)

            if abs(diff) < 0.03:  # ~1.7 degrees tolerance
                break

            # Speed scaling: full speed when far, proportional when close
            if abs(diff) > slow_zone:
                speed = self.angular_speed
            else:
                # Linear ramp from min_speed at zero to angular_speed at slow_zone
                speed = min_speed + (self.angular_speed - min_speed) * (abs(diff) / slow_zone)

            cmd.angular.z = direction * speed
            self.cmd_pub.publish(cmd)
            rate.sleep()

        # stop
        self.cmd_pub.publish(Twist())
        rospy.loginfo("Reached target within %.2f deg",
                      abs(self._angle_diff(target_yaw, self.current_yaw)) * 180 / math.pi)

    def _normalize(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _angle_diff(self, target, current):
        diff = target - current
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def run(self):
        for i in range(self.rounds):
            rospy.loginfo("=== Round %d/%d ===", i + 1, self.rounds)
            self.rotate_by(self.target_angle)
            if i < self.rounds - 1:
                rospy.loginfo("Sleeping %.1fs...", self.sleep_time)
                rospy.sleep(self.sleep_time)

        rospy.loginfo("Done: %d rotations of 90 deg completed.", self.rounds)


if __name__ == '__main__':
    tester = RotateTest()
    tester.run()
