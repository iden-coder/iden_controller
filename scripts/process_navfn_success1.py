#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import signal
import sys
import math
import time

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from actionlib_msgs.msg import GoalID, GoalStatus
from std_srvs.srv import Empty


class State:
    def __init__(self):
        self.client = None
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_dist = 0.0
        self.odom_prev = (0.0, 0.0)
        self.odom_yaw = 0.0
        self.odom_yaw_prev = None
        self.odom_yaw_delta = 0.0

        self.amcl_received = False
        self.amcl_x = 0.0
        self.amcl_y = 0.0
        self.amcl_yaw = 0.0

        self.scan_received = False
        self.scan_stamp = rospy.Time(0)
        self.front_min = float('inf')
        self.left_min = float('inf')
        self.right_min = float('inf')

        # Fixed patrol points copied from the old process_navfn.py.
        self.nav_points = {
            "s0":   [1.07,   0.0,    0.0, 0.0, 0.0, 0.0,      1.0],
            "s0t":  [1.07,  -0.05,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s1":   [1.08,  -0.395,  0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s1t":  [1.08,  -0.395,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s2":   [1.53,  -0.395,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s2t":  [1.53,  -0.395,  0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s3":   [1.53,  -0.0366, 0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s3t":  [1.53,  -0.0366, 0.0, 0.0, 0.0, 0.0,      1.0],
            "s4":   [2.57,   0.005,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s4t":  [2.59,   0.005,  0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s5":   [2.59,  -0.42,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s6":   [2.61,  -1.05,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s6t":  [2.57,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s7":   [2.13,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s8":   [1.53,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s9":   [0.63,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s9t":  [0.63,  -1.05,   0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s10":  [0.63,  -0.63,   0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s10t": [0.63,  -0.63,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s11":  [0.25,  -0.63,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s12":  [-0.04, -0.83,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s13":  [-0.33, -1.00,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s15":  [-1.5,  -0.4,    0.0, 0.0, 0.0, 1.0,      0.0],
        }

        self.patrol_path = [
            "s0", "s0t", "s1", "s1t", "s2", "s2t", "s3", "s3t",
            "s4", "s4t", "s6", "s6t",
            "s9", "s9t", "s10", "s10t", "s11", "s12", "s13", "s15",
        ]


S = State()


STATUS_NAMES = {
    GoalStatus.PENDING:    "PENDING",
    GoalStatus.ACTIVE:     "ACTIVE",
    GoalStatus.PREEMPTED:  "PREEMPTED",
    GoalStatus.SUCCEEDED:  "SUCCEEDED",
    GoalStatus.ABORTED:    "ABORTED",
    GoalStatus.REJECTED:   "REJECTED",
    GoalStatus.RECALLED:   "RECALLED",
    GoalStatus.LOST:       "LOST",
}


def goal_state_name(state):
    return STATUS_NAMES.get(state, "UNKNOWN(%s)" % state)


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def cb_odom(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    yaw = yaw_from_quat(msg.pose.pose.orientation)

    px, py = S.odom_prev
    S.odom_dist += math.hypot(x - px, y - py)
    if S.odom_yaw_prev is not None:
        S.odom_yaw_delta += abs(angle_diff(yaw, S.odom_yaw_prev))

    S.odom_prev = (x, y)
    S.odom_x = x
    S.odom_y = y
    S.odom_yaw = yaw
    S.odom_yaw_prev = yaw


def cb_amcl(msg):
    S.amcl_received = True
    S.amcl_x = msg.pose.pose.position.x
    S.amcl_y = msg.pose.pose.position.y
    S.amcl_yaw = yaw_from_quat(msg.pose.pose.orientation)


def scan_min_in_range(msg, min_deg, max_deg):
    lo = math.radians(min_deg)
    hi = math.radians(max_deg)
    best = float('inf')
    for i, r in enumerate(msg.ranges):
        if not (msg.range_min <= r <= msg.range_max):
            continue
        a = msg.angle_min + i * msg.angle_increment
        if lo <= a <= hi and r < best:
            best = r
    return best


def cb_scan(msg):
    S.scan_received = True
    S.scan_stamp = rospy.Time.now()
    S.front_min = scan_min_in_range(msg, -25.0, 25.0)
    S.left_min = scan_min_in_range(msg, 35.0, 70.0)
    S.right_min = scan_min_in_range(msg, -70.0, -35.0)


def scan_is_fresh(max_age=0.6):
    return S.scan_received and (rospy.Time.now() - S.scan_stamp).to_sec() < max_age


def stop_robot(duration=0.25):
    pubs = [
        rospy.Publisher('/cmd_vel_raw', Twist, queue_size=1),
        rospy.Publisher('/cmd_vel', Twist, queue_size=1),
    ]
    rospy.sleep(0.05)
    zero = Twist()
    end_t = time.time() + duration
    rate = rospy.Rate(20)
    while time.time() < end_t and not rospy.is_shutdown():
        for pub in pubs:
            pub.publish(zero)
        rate.sleep()


def clear_costmaps():
    try:
        rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
        rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
        rospy.loginfo("  costmaps cleared")
    except Exception as e:
        rospy.logwarn("  clear_costmaps failed: %s", e)


def cancel_all():
    if S.client:
        S.client.cancel_all_goals()
    pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)
    rospy.sleep(0.1)
    pub.publish(GoalID())


def send_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(0.5)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.orientation.w = 1.0
    pub.publish(msg)
    rospy.loginfo("initial pose sent")


def force_move_x(distance, speed=0.35, direction=1, min_front_clear=0.20):
    topic = rospy.get_param('~force_cmd_vel_topic', '/cmd_vel_raw')
    duration = abs(distance) / max(abs(speed), 1e-3)
    pub = rospy.Publisher(topic, Twist, queue_size=10)
    rospy.sleep(0.2)

    twist = Twist()
    twist.linear.x = direction * abs(speed)
    rospy.loginfo("  force move %s %.2fm @ %.2fm/s via %s",
                  "x" if direction > 0 else "-x", distance, abs(speed), topic)

    start = time.time()
    rate = rospy.Rate(20)
    while time.time() - start < duration and not rospy.is_shutdown():
        if direction > 0 and scan_is_fresh() and S.front_min < min_front_clear:
            rospy.logwarn("  force move stopped: front=%.2fm < %.2fm",
                          S.front_min, min_front_clear)
            break
        pub.publish(twist)
        rate.sleep()

    stop_robot(0.25)
    rospy.loginfo("  force move done")


def reverse_escape(distance=0.18, speed=0.08):
    topic = rospy.get_param('~force_cmd_vel_topic', '/cmd_vel_raw')
    pub = rospy.Publisher(topic, Twist, queue_size=10)
    rospy.sleep(0.15)

    start_x = S.odom_x
    start_y = S.odom_y
    max_duration = distance / max(abs(speed), 1e-3) + 0.6
    twist = Twist()
    twist.linear.x = -abs(speed)

    rospy.logwarn("  escape: reverse %.2fm @ %.2fm/s via %s",
                  distance, abs(speed), topic)
    start_t = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        moved = math.hypot(S.odom_x - start_x, S.odom_y - start_y)
        if moved >= distance or time.time() - start_t >= max_duration:
            break
        pub.publish(twist)
        rate.sleep()

    stop_robot(0.25)
    rospy.logwarn("  escape done: moved %.2fm", math.hypot(S.odom_x - start_x, S.odom_y - start_y))


def make_goal(pose7):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(
        Point(pose7[0], pose7[1], pose7[2]),
        Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))
    return goal


def target_yaw(pose7):
    q = Quaternion(pose7[3], pose7[4], pose7[5], pose7[6])
    return yaw_from_quat(q)


def goal_error(pose7):
    if not S.amcl_received:
        return None, None
    dist = math.hypot(S.amcl_x - pose7[0], S.amcl_y - pose7[1])
    yaw_err = abs(angle_diff(S.amcl_yaw, target_yaw(pose7)))
    return dist, yaw_err


def goal_is_verified(pose7):
    dist_tol = rospy.get_param('~goal_pos_tolerance', 0.28)
    yaw_tol = rospy.get_param('~goal_yaw_tolerance', 0.70)
    dist, yaw_err = goal_error(pose7)
    if dist is None:
        return True, "no amcl pose"
    if dist <= dist_tol and yaw_err <= yaw_tol:
        return True, "pos %.2fm yaw %.1fdeg" % (dist, math.degrees(yaw_err))
    return False, "pos %.2fm/%.2fm yaw %.1f/%.1fdeg" % (
        dist, dist_tol, math.degrees(yaw_err), math.degrees(yaw_tol))


def goto_point_once(name, pose7, hard_timeout=12.0,
                    stuck_window=3.0, stuck_min_dist=0.06,
                    stuck_min_yaw_deg=8.0):
    S.odom_dist = 0.0
    S.odom_prev = (S.odom_x, S.odom_y)
    S.odom_yaw_delta = 0.0
    S.odom_yaw_prev = S.odom_yaw

    stuck_dist_at = 0.0
    stuck_yaw_at = 0.0
    stuck_time_at = time.time()
    start_t = time.time()

    rospy.loginfo("  [%s] goal x=%.3f y=%.3f timeout=%.1fs",
                  name, pose7[0], pose7[1], hard_timeout)
    S.client.send_goal(make_goal(pose7))

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        now_t = time.time()
        elapsed = now_t - start_t
        state = S.client.get_state()

        if state == GoalStatus.SUCCEEDED:
            verified, reason = goal_is_verified(pose7)
            if verified:
                rospy.loginfo("  [%s] reached dist=%.2fm time=%.1fs (%s)",
                              name, S.odom_dist, elapsed, reason)
                return True
            rospy.logwarn("  [%s] move_base succeeded but target not verified: %s",
                          name, reason)
            return False
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                     GoalStatus.RECALLED, GoalStatus.PREEMPTED):
            rospy.logwarn("  [%s] move_base ended: %s",
                          name, goal_state_name(state))
            return False
        if elapsed >= hard_timeout:
            verified, reason = goal_is_verified(pose7)
            if verified:
                rospy.loginfo("  [%s] accepted on timeout: %s", name, reason)
                return True
            rospy.logwarn("  [%s] timeout %.1fs dist=%.2fm",
                          name, hard_timeout, S.odom_dist)
            return False

        if now_t - stuck_time_at >= stuck_window:
            moved = S.odom_dist - stuck_dist_at
            yaw = S.odom_yaw_delta - stuck_yaw_at
            turning = yaw > math.radians(stuck_min_yaw_deg)
            verified, reason = goal_is_verified(pose7)
            if verified:
                rospy.loginfo("  [%s] accepted while checking stuck: %s", name, reason)
                return True
            if moved < stuck_min_dist and not turning:
                rospy.logwarn("  [%s] stuck: moved=%.3fm yaw=%.1fdeg",
                              name, moved, math.degrees(yaw))
                return False
            stuck_dist_at = S.odom_dist
            stuck_yaw_at = S.odom_yaw_delta
            stuck_time_at = now_t

        rate.sleep()

    return False


def recover_before_retry(name):
    rospy.logwarn("  [%s] recovery: cancel, stop, clear, retry", name)
    cancel_all()
    stop_robot(0.3)
    if scan_is_fresh() and S.front_min < 0.18:
        reverse_escape(distance=0.18, speed=0.08)
    clear_costmaps()
    rospy.sleep(0.6)


def goto_point(name, pose7, retries=1):
    for attempt in range(retries + 1):
        ok = goto_point_once(name, pose7)
        if ok:
            return True
        if attempt < retries:
            recover_before_retry(name)
    return False


def cruise():
    total = len(S.patrol_path)
    ok_count = 0
    fail_list = []

    for idx, name in enumerate(S.patrol_path):
        if rospy.is_shutdown():
            break
        if name not in S.nav_points:
            rospy.logerr("[%s] missing from nav_points, skip", name)
            fail_list.append(name)
            continue

        rospy.loginfo("--- [%d/%d] %s ---", idx + 1, total, name)
        success = goto_point(name, S.nav_points[name], retries=1)
        if success:
            ok_count += 1
        else:
            fail_list.append(name)
            cancel_all()
            stop_robot(0.25)
            clear_costmaps()
            rospy.logerr("patrol stopped at %s after failed retries", name)
            break

        if success and name == "s13":
            force_move_x(0.60, speed=0.35, direction=1)

        rospy.sleep(0.25)

    stop_robot(0.4)
    rospy.loginfo("=" * 50)
    rospy.loginfo("patrol finished: success %d/%d, failed %d",
                  ok_count, total, len(fail_list))
    if fail_list:
        rospy.logwarn("failed points: %s", ", ".join(fail_list))
    rospy.loginfo("=" * 50)


def on_shutdown(sig=None, frame=None):
    rospy.loginfo("shutting down...")
    try:
        cancel_all()
        stop_robot(0.3)
    except Exception:
        pass
    rospy.signal_shutdown("user_exit")
    sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('cruise_mode_safe_fast')
    signal.signal(signal.SIGINT, on_shutdown)

    rospy.loginfo("connecting move_base action server...")
    S.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not S.client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("move_base unavailable")
        sys.exit(1)
    rospy.loginfo("move_base connected")

    rospy.Subscriber('/odom', Odometry, cb_odom, queue_size=1)
    rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, cb_amcl, queue_size=1)
    rospy.Subscriber('/scan', LaserScan, cb_scan, queue_size=1)
    rospy.sleep(0.8)

    send_initial_pose()
    rospy.sleep(1.0)
    rospy.loginfo("start patrol: safe-fast navfn mode")
    cruise()
    rospy.loginfo("task done, node stays alive (Ctrl-C to exit)")
    rospy.spin()
