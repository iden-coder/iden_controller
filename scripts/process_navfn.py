#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定点巡航脚本 — navfn 简化版
配套 launch: cruise_navfn.launch (navfn + IdenPlanner)

与 process_5.30.1.py 的区别:
  - 配合 navfn/NavfnROS 全局规划器（路径更直）
  - 其他逻辑完全相同
"""

import rospy
import actionlib
import signal
import sys
import math
import time

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from actionlib_msgs.msg import GoalID, GoalStatus
from std_srvs.srv import Empty


class State:
    def __init__(self):
        self.client = None
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_dist = 0.0
        self.odom_prev = (0.0, 0.0)

        self.nav_points = {
            "s0":   [1.07,   0.0,    0.0, 0.0, 0.0, 0.0,      1.0],
            "s0t":  [1.07,  -0.05,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s1":   [1.08,  -0.395,  0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s1t":  [1.08,  -0.395,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s2":   [1.55,  -0.395,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s2t":  [1.55,  -0.395,  0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s3":   [1.55,   -0.04,   0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s3t":  [1.55,   -0.04,   0.0, 0.0, 0.0, 0.0,      1.0],
            "s4":   [2.55,  -0.02,   0.0, 0.0, 0.0, 0.0,      1.0],
            "s4t":  [2.57,  -0.02,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s5":   [2.61,  -0.42,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s6":   [2.61,  -1.05,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s6t":  [2.52,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s7":   [2.13,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s8":   [1.53,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s9":   [0.63,  -1.05,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s9t":  [0.63,  -1.05,   0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s10":  [0.63,  -0.63,   0.0, 0.0, 0.0, 0.7071,  0.7071],
            "s10t": [0.63,  -0.63,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s11":  [0.25, -0.63,  0.0, 0.0, 0.0, 1.0,      0.0],
            "s12":  [-0.04, -0.73, 0.0, 0.0, 0.0, -0.9239,  0.3827],
            "s13":  [-0.304, -1.02, 0.0, 0.0, 0.0, 1.0,   0.0],
            "s14":  [-0.344, -1.03, 0.0, 0.0, 0.0, 1.0,   0.0],
        }

        self.patrol_path = [
            "s0", "s0t", "s1", "s1t", "s2", "s2t", "s3", "s3t",
            "s4", "s4t", "s6", "s6t",
            "s9", "s9t", "s10", "s10t", "s11", "s12","s13","s14",
        ]


S = State()


def cb_odom(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    px, py = S.odom_prev
    d = math.hypot(x - px, y - py)
    S.odom_dist += d
    S.odom_prev = (x, y)
    S.odom_x = x
    S.odom_y = y


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
    return STATUS_NAMES.get(state, f"UNKNOWN({state})")


def clear_costmaps():
    try:
        rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
        srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        srv()
        rospy.loginfo("  代价地图已清除")
    except Exception as e:
        rospy.logwarn(f"  清除代价地图失败: {e}")


def cancel_all():
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
    rospy.loginfo("初始位姿已发送")


def goto_point(name, pose7,
               hard_timeout=10.0,
               stuck_window=3.0,
               stuck_min_dist=0.08):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(
        Point(pose7[0], pose7[1], pose7[2]),
        Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))

    S.odom_dist = 0.0
    S.odom_prev = (S.odom_x, S.odom_y)
    stuck_dist_at = 0.0
    stuck_time_at = time.time()
    start_t = time.time()

    rospy.loginfo(f"  [{name}] 发目标 x={pose7[0]:.3f} y={pose7[1]:.3f} timeout={hard_timeout}s")
    S.client.send_goal(goal)

    rate = rospy.Rate(4)
    while not rospy.is_shutdown():
        now_t = time.time()
        elapsed = now_t - start_t

        state = S.client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo(f"  [{name}] ✓ 到达 (移动 {S.odom_dist:.2f}m, 耗时 {elapsed:.1f}s)")
            return True
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                     GoalStatus.RECALLED, GoalStatus.PREEMPTED):
            rospy.logwarn(f"  [{name}] ✗ move_base 终止 (state={goal_state_name(state)})")
            return False

        if elapsed >= hard_timeout:
            rospy.logwarn(f"  [{name}] ✗ 超时 {hard_timeout}s (移动 {S.odom_dist:.2f}m)")
            return False

        window_elapsed = now_t - stuck_time_at
        if window_elapsed >= stuck_window:
            moved_in_window = S.odom_dist - stuck_dist_at
            if moved_in_window < stuck_min_dist:
                rospy.logwarn(f"  [{name}] ✗ 卡死! {stuck_window}s 内仅移动 {moved_in_window:.3f}m")
                return False
            stuck_dist_at = S.odom_dist
            stuck_time_at = now_t

        rate.sleep()

    return False


def cruise():
    total = len(S.patrol_path)
    ok_count = 0
    fail_list = []

    for idx, name in enumerate(S.patrol_path):
        if rospy.is_shutdown():
            break
        if name not in S.nav_points:
            rospy.logerr(f"[{name}] 不在 nav_points 中，跳过")
            fail_list.append(name)
            continue

        rospy.loginfo(f"--- [{idx+1}/{total}] {name} ---")
        pose = S.nav_points[name]
        success = goto_point(name, pose)

        if success:
            ok_count += 1
        else:
            fail_list.append(name)
            cancel_all()
            rospy.sleep(0.3)
            clear_costmaps()
            rospy.sleep(0.5)

        rospy.sleep(0.3)

    rospy.loginfo("=" * 50)
    rospy.loginfo(f"巡航结束: 成功 {ok_count}/{total}, 失败 {len(fail_list)}")
    if fail_list:
        rospy.loginfo(f"失败点: {', '.join(fail_list)}")
    rospy.loginfo("=" * 50)


def on_shutdown(sig=None, frame=None):
    rospy.loginfo("正在退出...")
    try:
        cancel_all()
    except:
        pass
    rospy.signal_shutdown("user_exit")
    sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('cruise_mode')
    signal.signal(signal.SIGINT, on_shutdown)

    rospy.loginfo("连接 move_base action server ...")
    S.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not S.client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("无法连接 move_base!")
        sys.exit(1)
    rospy.loginfo("move_base 已连接")

    send_initial_pose()

    rospy.Subscriber('/odom', Odometry, cb_odom)
    rospy.sleep(1.0)
    rospy.loginfo("odom 监听就绪")

    rospy.sleep(1.0)
    rospy.loginfo("开始巡航 (navfn 模式)")
    cruise()
    rospy.loginfo("任务结束，节点保持运行 (Ctrl-C 退出)")
    rospy.spin()
