#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音触发巡航 — 唤醒词 "小飞小飞" → 语音指令 → 自动执行对应的巡航任务

依赖: speech_command 节点的 /angle (唤醒角度) 和 /factory/voice_raw_text (语音识别文本)

用法:
  roslaunch iden_controller cruise_navfn_v2_wide.launch
  rosrun speech_command speech_command_node &
  rosrun speech_command spark_llm_node.py &
  rosrun iden_controller cruise_voice.py

或者全部塞进一个 launch 里启动。
"""

import rospy
import actionlib
import signal
import sys
import math
import time
import json

from std_msgs.msg import String, Int32
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from actionlib_msgs.msg import GoalID, GoalStatus
from std_srvs.srv import Empty


# ============================================================
#  全局状态
# ============================================================
class VoiceState:
    def __init__(self):
        self.client = None
        self.odom_x = 0.0; self.odom_y = 0.0
        self.odom_dist = 0.0; self.odom_prev = (0.0, 0.0)
        self.map_x = 0.0; self.map_y = 0.0; self.map_yaw = 0.0
        self.tf_listener = None

        # 语音状态
        self.wake_angle = 0          # 唤醒角度
        self.last_wake_time = rospy.Time(0)
        self.voice_text = ""          # 语音识别文本
        self.voice_ready = False      # 有新指令
        self.target_warehouse = None  # LLM解析后的目标仓库

    def reset_voice(self):
        self.wake_angle = 0
        self.voice_text = ""
        self.voice_ready = False
        self.target_warehouse = None

    @property
    def wake_timeout(self):
        return (rospy.Time.now() - self.last_wake_time).to_sec() > 10.0


VS = VoiceState()


# ============================================================
#  语音回调
# ============================================================
def cb_angle(msg):
    """唤醒角度 — 非0表示已唤醒"""
    if msg.data != 0:
        VS.wake_angle = msg.data
        VS.last_wake_time = rospy.Time.now()
        rospy.loginfo(f"🔊 唤醒! 角度={msg.data}°")

def cb_voice_text(msg):
    """语音识别原始文本"""
    text = msg.data.strip()
    if text:
        VS.voice_text = text
        VS.voice_ready = True
        rospy.loginfo(f"🎤 语音: {text}")

def cb_target_warehouse(msg):
    """LLM 解析后的目标仓库"""
    try:
        data = json.loads(msg.data)
        VS.target_warehouse = data
        VS.voice_ready = True
        rospy.loginfo(f"📦 目标仓库: {data}")
    except:
        VS.target_warehouse = {"raw": msg.data}
        VS.voice_ready = True


# ============================================================
#  基础导航工具
# ============================================================
STATUS_NAMES = {
    GoalStatus.PENDING: "PENDING", GoalStatus.ACTIVE: "ACTIVE",
    GoalStatus.PREEMPTED: "PREEMPTED", GoalStatus.SUCCEEDED: "SUCCEEDED",
    GoalStatus.ABORTED: "ABORTED", GoalStatus.REJECTED: "REJECTED",
    GoalStatus.RECALLED: "RECALLED", GoalStatus.LOST: "LOST",
}

def cancel_all(): VS.client.cancel_all_goals()
def clear_costmaps():
    try:
        rospy.wait_for_service('/move_base/clear_costmaps', timeout=2.0)
        rospy.ServiceProxy('/move_base/clear_costmaps', Empty)()
    except: pass

def cb_odom(msg):
    x = msg.pose.pose.position.x; y = msg.pose.pose.position.y
    px, py = VS.odom_prev
    VS.odom_dist += math.hypot(x-px, y-py)
    VS.odom_prev = (x,y); VS.odom_x = x; VS.odom_y = y

def set_planner_mode(mode):
    """动态切换 IdenPlannerV2 行为模式"""
    ns = "/move_base/iden_planner_v2/IdenPlannerV2"
    if mode == "simple":
        rospy.set_param(f"{ns}/enable_traj_sampling", False)
        rospy.set_param(f"{ns}/enable_forward_sim", False)
        rospy.set_param(f"{ns}/enable_graded_speed", False)
        rospy.set_param(f"{ns}/max_linear_vel", 0.8)
        # rospy.loginfo("  规划器模式: simple (纯PID追踪)")
    elif mode == "avoid":
        rospy.set_param(f"{ns}/enable_traj_sampling", True)
        rospy.set_param(f"{ns}/enable_forward_sim", True)
        rospy.set_param(f"{ns}/enable_graded_speed", True)
        rospy.set_param(f"{ns}/weight_obstacle", 15.0)
        rospy.set_param(f"{ns}/max_linear_vel", 0.4)
        # rospy.loginfo("  规划器模式: avoid (全力避障)")
    elif mode == "precise":
        rospy.set_param(f"{ns}/enable_traj_sampling", False)
        rospy.set_param(f"{ns}/enable_graded_speed", True)
        rospy.set_param(f"{ns}/max_linear_vel", 0.25)
        # rospy.loginfo("  规划器模式: precise (精确靠站)")

def stop_robot(duration=0.5):
    pubs = [rospy.Publisher('/cmd_vel', Twist, queue_size=1)]
    rospy.sleep(0.05); msg = Twist()
    end = time.time() + duration
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.time() < end:
        for p in pubs: p.publish(msg)
        rate.sleep()

def backup_distance(dist=0.13, speed=-0.08, timeout=3.0):
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
    rospy.sleep(0.1); dist = abs(dist); speed = -abs(speed)
    VS.odom_dist = 0.0; VS.odom_prev = (VS.odom_x, VS.odom_y)
    start = time.time(); rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.time() - start < timeout:
        if VS.odom_dist >= dist: break
        msg = Twist(); msg.linear.x = speed; pub.publish(msg); rate.sleep()
    stop_robot(0.5)


# ============================================================
#  核心导航
# ============================================================
def goto_point(name, pose7, hard_timeout=15.0, mode="simple"):
    """去一个航点，带卡死检测和恢复"""
    set_planner_mode(mode)

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(Point(pose7[0], pose7[1], pose7[2]),
                                  Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))

    for attempt in range(3):
        VS.odom_dist = 0.0; VS.odom_prev = (VS.odom_x, VS.odom_y)
        stuck_dist_at = 0.0; stuck_time_at = time.time()
        start_t = time.time(); grace_period = 3.0

        rospy.loginfo(f"  [{name}] → ({pose7[0]:.3f}, {pose7[1]:.3f}) "
                      f"timeout={hard_timeout}s mode={mode} {'(重试)' if attempt > 0 else ''}")
        VS.client.send_goal(goal)

        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            elapsed = time.time() - start_t
            state = VS.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo(f"  [{name}] ✓ 到达 ({VS.odom_dist:.2f}m, {elapsed:.1f}s)")
                return True
            if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                         GoalStatus.RECALLED, GoalStatus.PREEMPTED):
                rospy.logwarn(f"  [{name}] ✗ 终止 ({STATUS_NAMES.get(state, '?')})")
                break
            if elapsed >= hard_timeout:
                rospy.logwarn(f"  [{name}] ✗ 超时")
                break
            if elapsed < grace_period:
                if elapsed >= grace_period - 0.5:
                    stuck_dist_at = VS.odom_dist; stuck_time_at = time.time()
                rate.sleep(); continue
            if time.time() - stuck_time_at >= 3.0:
                if VS.odom_dist - stuck_dist_at < 0.05:
                    rospy.logwarn(f"  [{name}] ✗ 卡死!")
                    break
                stuck_dist_at = VS.odom_dist; stuck_time_at = time.time()
            rate.sleep()

        if attempt < 2:
            cancel_all(); rospy.sleep(0.3)
            backup_distance(0.13)
            clear_costmaps(); rospy.sleep(0.5)
            rospy.loginfo(f"  [{name}] 恢复后重试")

    return False


def run_cruise(task_name, waypoints, point_defs):
    """执行一个巡航任务"""
    total = len(waypoints)
    ok = 0
    rospy.loginfo(f"🚀 开始任务: {task_name} ({total} 个航点)")

    ship_to_mode = {"avoid": "avoid", "simple": "simple", "precise": "precise"}

    for idx, wp in enumerate(waypoints):
        if rospy.is_shutdown(): break

        name, mode = wp if isinstance(wp, tuple) else (wp, "simple")
        pose7 = point_defs[name]
        timeout = 20.0 if idx == 0 else 15.0

        rospy.loginfo(f"--- [{idx+1}/{total}] {name} (mode={mode}) ---")
        if goto_point(name, pose7, hard_timeout=timeout, mode=mode):
            ok += 1
        else:
            cancel_all(); clear_costmaps(); rospy.sleep(0.3)

    rospy.loginfo(f"✅ {task_name}: 成功 {ok}/{total}")


# ============================================================
#  巡航任务定义
# ============================================================
def get_task_by_voice():
    """根据语音指令返回对应的巡航任务"""
    text = VS.voice_text.lower().replace(" ", "")

    # 航点库 (所有任务共享)
    point_defs = {
        "start": [-0.85, -1.41, 0, 0, 0, -0.7071, 0.7071],
        "d1":    [-1.63, -2.57, 0, 0, 0,  1,      0     ],
        "d1t":   [-1.63, -2.57, 0, 0, 0,  0,      1     ],
        "d2":    [ 0.41, -1.60, 0, 0, 0,  0.7071, 0.7071],
        "d2t":   [ 0.41, -1.60, 0, 0, 0, -0.7071, 0.7071],
        "d3":    [ 2.54, -2.81, 0, 0, 0, -0.7071, 0.7071],
        "d3t":   [ 2.54, -2.81, 0, 0, 0,  0.9239, 0.3827],
        "d4":    [ 0.173,-3.25, 0, 0, 0, -0.7071, 0.7071],
        # 原版航点 (process_navfn 的任务)
        "s0":   [1.07,   0.0,   0.0, 0.0, 0.0, 0.0,      1.0],
        "s0t":  [1.07,  -0.05,  0.0, 0.0, 0.0, -0.7071,  0.7071],
        "s1":   [1.08,  -0.395, 0.0, 0.0, 0.0, -0.7071,  0.7071],
        "s10":  [0.63,  -0.63,  0.0, 0.0, 0.0, 0.7071,  0.7071],
        "s15":  [-1.5,  -0.4,   0.0, 0.0, 0.0, 1.0,       0.0],
    }

    # 任务1: 锥桶穿行 (d1→d2→d3→d4, 锥桶区全力避障)
    if any(kw in text for kw in ["锥桶", "绕桩", "避障", "demo"]):
        waypoints = [
            ("d1", "simple"), ("d1t", None),   # d1 纯追踪, d1t 原地转(不是导航点)
            ("d2", "avoid"), ("d2t", None),     # 锥桶区: 全力避障
            ("d3", "avoid"), ("d3t", None),
            ("d4", "precise"),                  # 最后精确靠站
        ]
        # 滤掉 None (turn-only 点不在 goto_point 中处理)
        waypoints = [(n, m) for n, m in waypoints if m is not None]
        return "锥桶穿行", waypoints, point_defs

    # 任务2: 原版巡航 (process_navfn 的巡逻路径, 无障碍)
    if any(kw in text for kw in ["巡逻", "巡航", "原版", "简单"]):
        waypoints = ["s0", "s0t", "s1", "s10", "s15"]
        waypoints = [(n, "simple") for n in waypoints]
        return "原版巡逻", waypoints, point_defs

    # LLM 解析的目标仓库
    if VS.target_warehouse:
        wh = VS.target_warehouse
        rospy.loginfo(f"LLM仓库: {wh}")
        # TODO: 根据仓库名映射到具体航点
        return None, [], point_defs

    # 默认: 原版巡逻
    rospy.logwarn(f"未匹配指令 '{text}'，走默认巡逻")
    waypoints = [(n, "simple") for n in ["s0", "s0t", "s1", "s10", "s15"]]
    return "默认巡逻", waypoints, point_defs


# ============================================================
#  语音交互循环
# ============================================================
def voice_loop():
    """等待唤醒词 → 直接启动巡航(可选指令覆写) → 回到等待"""
    tts_pub = rospy.Publisher('/factory/tts_text', String, queue_size=10)

    while not rospy.is_shutdown():
        rospy.loginfo("⏳ 等待唤醒词...")
        VS.reset_voice()

        # 等唤醒
        while not rospy.is_shutdown() and VS.wake_angle == 0:
            rospy.sleep(0.2)

        # 已唤醒 — 先播报，同时偷听3秒有没有指令
        tts_pub.publish(String(data="我在"))
        rospy.sleep(0.5)
        rospy.loginfo("🔊 已唤醒, 听指令3s...")

        # 偷听3秒
        timeout = rospy.Time.now() + rospy.Duration(3.0)
        while not rospy.is_shutdown() and rospy.Time.now() < timeout:
            if VS.voice_ready:
                break
            rospy.sleep(0.1)

        # 匹配任务 (无指令→默认锥桶穿行)
        task_name, waypoints, point_defs = get_task_by_voice()
        if task_name is None or not waypoints:
            # 无指令直接走锥桶路线
            task_name = "锥桶穿行"
            waypoints = [("d1","simple"),("d2","avoid"),("d3","avoid"),("d4","precise")]
            point_defs = {
                "start":[-0.85,-1.41,0,0,0,-0.7071,0.7071],
                "d1":[-1.63,-2.57,0,0,0,1,0],
                "d2":[0.41,-1.60,0,0,0,0.7071,0.7071],
                "d3":[2.54,-2.81,0,0,0,-0.7071,0.7071],
                "d4":[0.173,-3.25,0,0,0,-0.7071,0.7071],
            }

        tts_pub.publish(String(data=f"收到，开始{task_name}"))
        rospy.sleep(0.5)

        # 执行巡航
        run_cruise(task_name, waypoints, point_defs)

        tts_pub.publish(String(data=f"{task_name}完成"))
        rospy.sleep(2.0)


# ============================================================
#  main
# ============================================================
def on_shutdown(sig=None, frame=None):
    rospy.loginfo("退出...")
    try: cancel_all()
    except: pass
    rospy.signal_shutdown("user_exit"); sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('cruise_voice')
    signal.signal(signal.SIGINT, on_shutdown)

    # 连接 move_base
    rospy.loginfo("连接 move_base...")
    VS.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not VS.client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("move_base 未连接!"); sys.exit(1)
    rospy.loginfo("move_base 已连接")

    # 订阅语音
    rospy.Subscriber('/angle', Int32, cb_angle)
    rospy.Subscriber('/factory/voice_raw_text', String, cb_voice_text)
    rospy.Subscriber('/factory/target_warehouses', String, cb_target_warehouse)
    rospy.Subscriber('/odom', Odometry, cb_odom)
    rospy.sleep(1.0)

    rospy.loginfo("=" * 60)
    rospy.loginfo("语音巡航系统就绪")
    rospy.loginfo("唤醒词: 小飞小飞")
    rospy.loginfo("指令:   '锥桶' / 'demo'  → 锥桶穿行")
    rospy.loginfo("        '巡逻' / '巡航' → 原版巡逻")
    rospy.loginfo("=" * 60)

    voice_loop()
