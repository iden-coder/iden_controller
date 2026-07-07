#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import rospy
from actionlib_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan


class State:
    scan = None
    scan_t = 0.0
    raw = None
    cmd = None
    amcl = None
    amcl_t = 0.0
    odom = None
    status_text = "NO_STATUS"


S = State()


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def scan_min(msg, angle_min_deg, angle_max_deg):
    if msg is None:
        return float("inf")
    lo = math.radians(angle_min_deg)
    hi = math.radians(angle_max_deg)
    best = float("inf")
    for i, value in enumerate(msg.ranges):
        if math.isnan(value) or math.isinf(value):
            continue
        if not (msg.range_min <= value <= msg.range_max):
            continue
        angle = msg.angle_min + i * msg.angle_increment
        if lo <= angle <= hi and value < best:
            best = value
    return best


def fmt(value):
    if value is None or math.isinf(value):
        return "inf"
    return "%.2f" % value


def status_name(status):
    names = {
        GoalStatus.PENDING: "PENDING",
        GoalStatus.ACTIVE: "ACTIVE",
        GoalStatus.PREEMPTED: "PREEMPTED",
        GoalStatus.SUCCEEDED: "SUCCEEDED",
        GoalStatus.ABORTED: "ABORTED",
        GoalStatus.REJECTED: "REJECTED",
        GoalStatus.PREEMPTING: "PREEMPTING",
        GoalStatus.RECALLING: "RECALLING",
        GoalStatus.RECALLED: "RECALLED",
        GoalStatus.LOST: "LOST",
    }
    return names.get(status, str(status))


def cb_scan(msg):
    S.scan = msg
    S.scan_t = time.time()


def cb_raw(msg):
    S.raw = msg


def cb_cmd(msg):
    S.cmd = msg


def cb_amcl(msg):
    S.amcl = msg
    S.amcl_t = time.time()


def cb_odom(msg):
    S.odom = msg


def cb_status(msg):
    if msg.status_list:
        S.status_text = status_name(msg.status_list[-1].status)
    else:
        S.status_text = "EMPTY"


def main():
    rospy.init_node("codex_live_monitor", anonymous=True)
    rospy.Subscriber("/scan", LaserScan, cb_scan, queue_size=1)
    rospy.Subscriber("/cmd_vel_raw", Twist, cb_raw, queue_size=1)
    rospy.Subscriber("/cmd_vel", Twist, cb_cmd, queue_size=1)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, cb_amcl, queue_size=1)
    rospy.Subscriber("/odom", Odometry, cb_odom, queue_size=1)
    rospy.Subscriber("/move_base/status", GoalStatusArray, cb_status, queue_size=1)

    print("CODEX_MONITOR_START read_only=1 hz=1", flush=True)
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        now = time.time()
        front = scan_min(S.scan, -30, 30)
        left = scan_min(S.scan, 35, 75)
        right = scan_min(S.scan, -75, -35)
        rear = min(scan_min(S.scan, 145, 180), scan_min(S.scan, -180, -145))
        raw_v = S.raw.linear.x if S.raw else 0.0
        raw_w = S.raw.angular.z if S.raw else 0.0
        cmd_v = S.cmd.linear.x if S.cmd else 0.0
        cmd_w = S.cmd.angular.z if S.cmd else 0.0

        amcl = "NA"
        if S.amcl:
            pos = S.amcl.pose.pose.position
            yaw = math.degrees(yaw_from_quat(S.amcl.pose.pose.orientation))
            amcl = "x=%.2f y=%.2f yaw=%.0f age=%.1f" % (
                pos.x, pos.y, yaw, now - S.amcl_t)

        alerts = []
        if now - S.scan_t > 1.0:
            alerts.append("SCAN_STALE %.1fs" % (now - S.scan_t))
        if now - S.amcl_t > 2.5:
            alerts.append("AMCL_STALE %.1fs" % (now - S.amcl_t))
        if raw_v < -0.01 or cmd_v < -0.01:
            alerts.append("REVERSING raw=%.3f cmd=%.3f rear=%s" % (raw_v, cmd_v, fmt(rear)))
        if (raw_v < -0.01 or cmd_v < -0.01) and rear < 0.28:
            alerts.append("REAR_COLLISION_RISK")
        if (raw_v > 0.02 or cmd_v > 0.02) and front < 0.18:
            alerts.append("FRONT_COLLISION_RISK")
        if min(left, right) < 0.09:
            alerts.append("SIDE_TOO_CLOSE L=%s R=%s" % (fmt(left), fmt(right)))

        print(
            "MON t=%s status=%s raw=(%.3f,%.3f) cmd=(%.3f,%.3f) scan F/L/R/Rear=%s/%s/%s/%s amcl=%s alerts=%s" %
            (time.strftime("%H:%M:%S"), S.status_text, raw_v, raw_w, cmd_v, cmd_w,
             fmt(front), fmt(left), fmt(right), fmt(rear), amcl,
             ",".join(alerts) if alerts else "OK"),
            flush=True)
        rate.sleep()


if __name__ == "__main__":
    main()
