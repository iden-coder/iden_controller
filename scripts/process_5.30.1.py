#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定点巡航脚本 — 防卡死版本

核心策略：
  每个导航点只尝试 1 次，硬超时 20s。超时或卡死 → 直接跳到下一个点，绝不原地死等。
  卡死判定：每 3 秒检查 odom 移动距离，< 0.08m 即认为卡死。
  代价地图只在切换导航点时清一次，避免反复清图破坏规划。
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


# ============================================================
#  全局状态
# ============================================================
class State:
    def __init__(self):
        self.client = None
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_dist = 0.0       # 本轮累计移动距离
        self.odom_prev = (0.0, 0.0) # 上一帧位置

        self.nav_points = {
            "s0":   [1.07,   0.0,    0.0, 0.0, 0.0, 0.0,      1.0],
            "s0t":  [1.07,  -0.05,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s1":   [1.08,  -0.395,  0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s1t":  [1.08,  -0.395,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s2":   [1.55,  -0.395,  0.0, 0.0, 0.0, 0.0,      1.0],
            "s2t":  [1.55,  -0.365,  0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s3":   [1.56,   0.01,   0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s3t":  [1.75,   0.01,   0.0, 0.0, 0.0, 0.0,      1.0],
            "s4":   [2.60,  0.01,   0.0, 0.0, 0.0, 0.0,      1.0],
            "s4t":  [2.60,  -0.06,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s5":   [2.60,  -0.42,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s6":   [2.60,  -0.99,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s6t":  [2.47,  -0.99,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s7":   [2.13,  -0.99,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s8":   [1.53,  -0.99,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s9":   [0.63,  -0.99,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s9t":  [0.63,  -0.94,   0.0, 0.0, 0.0, 0.7071,   0.7071],
            "s10":  [0.63,  -0.55,   0.0, 0.0, 0.0, -0.7071,  0.7071],
            "s10t": [0.48,  -0.55,   0.0, 0.0, 0.0, 1.0,      0.0],
            "s11":  [-0.344, -0.55,  0.0, 0.0, 0.0, 1.0,      0.0],
            "s12":  [-0.344, -0.985, 0.0, 0.0, 0.0, 1.0,      0.0],
        }

        self.patrol_path = [
            "s0", "s0t", "s1", "s1t", "s2", "s2t", "s3", "s3t",
            "s4", "s4t", "s5", "s6", "s6t", "s7", "s8",
            "s9", "s9t", "s10", "s10t", "s11", "s12",
        ]


S = State()


# ============================================================
#  odom 回调
# ============================================================
def cb_odom(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y

    px, py = S.odom_prev
    d = math.hypot(x - px, y - py)
    S.odom_dist += d
    S.odom_prev = (x, y)
    S.odom_x = x
    S.odom_y = y


# ============================================================
#  action 状态 → 可读字符串
# ============================================================
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


# ============================================================
#  清除代价地图
# ============================================================
def clear_costmaps():
    try:
        rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
        srv = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        srv()
        rospy.loginfo("  代价地图已清除")
    except Exception as e:
        rospy.logwarn(f"  清除代价地图失败: {e}")


# ============================================================
#  取消所有导航目标
# ============================================================
def cancel_all():
    S.client.cancel_all_goals()
    # 补一刀 topic 确保送达
    pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)
    rospy.sleep(0.1)
    pub.publish(GoalID())


# ============================================================
#  发送初始位姿
# ============================================================
def send_initial_pose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(0.5)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.orientation.w = 1.0
    pub.publish(msg)
    rospy.loginfo("初始位姿已发送")


# ============================================================
#  核心：去一个点，永不卡死
#  - hard_timeout:     总超时秒数 (到时间直接放弃)
#  - stuck_window:      卡死检测窗口秒数
#  - stuck_min_dist:    窗口内最低移动距离，低于此值 → 卡死
# ============================================================
def goto_point(name, pose7,
               hard_timeout=10.0,
               stuck_window=3.0,
               stuck_min_dist=0.08):
    """
    发送一个导航目标，轮询等待直到:
      1. move_base 报告成功 → return True
      2. move_base 报告失败 → return False
      3. 总超时 → return False
      4. 卡死(移动量不足) → return False

    绝不重试，绝不阻塞超过 hard_timeout 秒。
    """
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(
        Point(pose7[0], pose7[1], pose7[2]),
        Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))

    # ---- 重置本轮里程计 ----
    S.odom_dist = 0.0
    S.odom_prev = (S.odom_x, S.odom_y)
    stuck_dist_at = 0.0       # 卡死窗口起点时的累计距离
    stuck_time_at = time.time()
    start_t = time.time()

    rospy.loginfo(f"  [{name}] 发目标 x={pose7[0]:.3f} y={pose7[1]:.3f} timeout={hard_timeout}s")

    S.client.send_goal(goal)

    rate = rospy.Rate(4)  # 4Hz 轮询
    while not rospy.is_shutdown():
        now_t = time.time()
        elapsed = now_t - start_t

        # ---- 检查 action 状态 ----
        state = S.client.get_state()
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo(f"  [{name}] ✓ 到达 (移动 {S.odom_dist:.2f}m, 耗时 {elapsed:.1f}s)")
            return True
        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                     GoalStatus.RECALLED, GoalStatus.PREEMPTED):
            rospy.logwarn(f"  [{name}] ✗ move_base 终止 (state={goal_state_name(state)})")
            return False

        # ---- 硬超时 ----
        if elapsed >= hard_timeout:
            rospy.logwarn(f"  [{name}] ✗ 超时 {hard_timeout}s (移动 {S.odom_dist:.2f}m)")
            return False

        # ---- 卡死检测 ----
        window_elapsed = now_t - stuck_time_at
        if window_elapsed >= stuck_window:
            moved_in_window = S.odom_dist - stuck_dist_at
            if moved_in_window < stuck_min_dist:
                rospy.logwarn(f"  [{name}] ✗ 卡死! {stuck_window}s 内仅移动 {moved_in_window:.3f}m (总移动 {S.odom_dist:.2f}m)")
                return False
            # 未卡死，重置窗口
            stuck_dist_at = S.odom_dist
            stuck_time_at = now_t

        rate.sleep()

    return False


# ============================================================
#  主巡航逻辑
# ============================================================
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
            # 失败后清理：取消目标 + 清代价地图，为下个点做准备
            cancel_all()
            rospy.sleep(0.3)
            clear_costmaps()
            rospy.sleep(0.5)

        # 点与点之间短暂停一下，让机器人稳定
        rospy.sleep(0.3)

    rospy.loginfo("=" * 50)
    rospy.loginfo(f"巡航结束: 成功 {ok_count}/{total}, 失败 {len(fail_list)}")
    if fail_list:
        rospy.loginfo(f"失败点: {', '.join(fail_list)}")
    rospy.loginfo("=" * 50)


# ============================================================
#  信号处理
# ============================================================
def on_shutdown(sig=None, frame=None):
    rospy.loginfo("正在退出...")
    try:
        cancel_all()
    except:
        pass
    rospy.signal_shutdown("user_exit")
    sys.exit(0)


# ============================================================
#  main
# ============================================================
if __name__ == '__main__':
    rospy.init_node('cruise_mode')
    signal.signal(signal.SIGINT, on_shutdown)

    # --- 连接 move_base ---
    rospy.loginfo("连接 move_base action server ...")
    S.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not S.client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("无法连接 move_base!")
        sys.exit(1)
    rospy.loginfo("move_base 已连接")

    # --- 发初始位姿 ---
    send_initial_pose()

    # --- 订阅 odom ---
    rospy.Subscriber('/odom', Odometry, cb_odom)
    rospy.sleep(1.0)
    rospy.loginfo("odom 监听就绪")

    rospy.sleep(1.0)
    rospy.loginfo("开始巡航")
    cruise()
    rospy.loginfo("任务结束，节点保持运行 (Ctrl-C 退出)")
    rospy.spin()
