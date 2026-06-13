#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
定点巡航 — start→d1→(锥桶区)→d2

起点: (-0.85, -1.41, yaw≈-90°)
 d1:  (-1.53, -2.57, yaw=180°)
 d2:  ( 0.41, -1.60, yaw=90°)

锥桶区在 d1→d2 之间 (~2.2m 路径)。
可通过配置 midpoints 引导机器人穿行锥桶间隙。

用法:
  roslaunch iden_controller cruise_navfn_v2_wide.launch
  rosrun iden_controller cruise_demo.py
"""

import rospy
import actionlib
import signal
import sys
import math
import time

import tf

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, Path
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

        # map坐标系下的真实位姿：导航目标是map坐标，最终角度也必须用map yaw判断。
        self.map_x = 0.0
        self.map_y = 0.0
        self.map_yaw = 0.0
        self.tf_listener = None

        # 航点: [x, y, z, qx, qy, qz, qw]
        self.nav_points = {
            "start": [-0.85, -1.41, 0,  0, 0, -0.7071, 0.7071],
            "d1":    [-1.63, -2.57, 0,  0, 0,  1,      0     ],
            "d1t":   [-1.63, -2.57, 0,  0, 0,  0,      1     ],
            "d2":    [ 0.41, -1.60, 0,  0, 0,  0.7071, 0.7071],
            "d2t":   [ 0.41, -1.60, 0,  0, 0, -0.7071, 0.7071],
            "d3":    [ 2.54, -2.81, 0,  0, 0, -0.7071, 0.7071],
            "d3t":   [ 2.54, -2.81, 0,  0, 0,  0.9239, 0.3827],
            "d4":    [ 0.273,-3.55, 0,  0, 0, -0.7071, 0.7071],
        }

        # ---- 锥桶区中间航点 (d1→d2 穿行引导) ----
        # 锥桶排列在你指定的 d1→d2 之间，在此添加中间点来引导穿行。
        # 坐标格式: [x, y, z, qx, qy, qz, qw]
        # 如果不放锥桶或不确定位置，留空列表即可。
        self.cone_midpoints = [
            # 示例: 如果你在 d1 和 d2 的正中间 (约 -0.56, -2.08) 留了通道，
            # 取消下面这行的注释并调整坐标:
            # [-0.56, -2.08, 0,  0, 0, 0.7071, 0.7071],
        ]

        # 自动构建巡航路径: d1 → midpoints → d2
        self.patrol_path = self._build_path()

    def _build_path(self):
        # d1t/d2t 与 d1/d2 坐标相同，不作为 move_base 导航点；到达后由本节点原地严格转向。
        path = ["d1"]
        for i, mp in enumerate(self.cone_midpoints):
            name = f"mp{i+1}"
            self.nav_points[name] = mp
            path.append(name)
        path.extend(["d2", "d3", "d4"])
        return path


S = State()


# ===== 精确到点 / 靠墙点 / 严格朝向参数 =====
# 靠墙点不能让 move_base 一直做最终朝向调整，否则会把墙当障碍原地转圈。
# 本节点采用：位置进容差 -> cancel move_base -> 自己严格原地对准目标yaw -> 停车等待。
WALL_GOAL_NAMES = set(rospy.get_param("~wall_goal_names", ["d1", "d2", "d3", "d4"]))
NORMAL_GOAL_XY_TOL = rospy.get_param("~goal_xy_tolerance", 0.08)
WALL_GOAL_XY_TOL = rospy.get_param("~wall_goal_xy_tolerance", 0.08)
STRICT_FINAL_YAW = rospy.get_param("~strict_final_yaw", True)
FINAL_YAW_TOL_DEG = rospy.get_param("~final_yaw_tolerance_deg", 2.0)
FINAL_YAW_TIMEOUT = rospy.get_param("~final_yaw_timeout", 15.0)
ARRIVAL_HOLD_SEC = rospy.get_param("~arrival_hold_sec", 3.0)

# d1 到达后执行 d1t 的朝向，不把 d1t 交给 move_base 导航。
TURN_AFTER_POINTS = {"d1": "d1t", "d2": "d2t", "d3": "d3t"}

# 卡死恢复：先倒退约10cm，清除代价地图，再重新规划/必要时反应式避障。
BACKUP_ON_STUCK = rospy.get_param("~backup_on_stuck", True)
BACKUP_DIST = rospy.get_param("~backup_dist", 0.10)
BACKUP_SPEED = rospy.get_param("~backup_speed", -0.08)
# 倒退安全阈值：后方最近障碍 <= 5cm 时，立即禁止/停止倒退
REAR_OBSTACLE_STOP_DIST = rospy.get_param("~rear_obstacle_stop_dist", 0.05)
HEADING_RECOVERY_FIRST = rospy.get_param("~heading_recovery_first", True)
HEADING_RECOVERY_MIN_DIST = rospy.get_param("~heading_recovery_min_dist", 0.35)
NO_AVOID_NEAR_GOAL = rospy.get_param("~no_avoid_near_goal_dist", 0.30)  # 距目标30cm内不绕障

# ===== 动态锥桶段参数 =====
CONE_ZONE_PARAM = "/iden_controller/cone_zone"
CURRENT_TARGET_PARAM = "/iden_controller/current_cone_target"
CONE_ZONE_NAMES = set(rospy.get_param("~cone_zone_names", ["d2"]))
REACTIVE_RECOVERY_ENABLED = rospy.get_param("~reactive_recovery_enabled", True)
REACTIVE_RECOVERY_ATTEMPTS = int(rospy.get_param("~reactive_recovery_attempts", 2))
REACTIVE_RECOVERY_DURATION = rospy.get_param("~reactive_recovery_duration", 8.0)
CONE_STUCK_WINDOW = rospy.get_param("~cone_stuck_window", 2.0)
CONE_STUCK_MIN_DIST = rospy.get_param("~cone_stuck_min_dist", 0.04)

# ===== 诊断: 当卡死时查看 move_base 内部状态 =====
_last_plan = None
_last_plan_stamp = rospy.Time(0)

def cb_diag_plan(msg):
    global _last_plan, _last_plan_stamp
    _last_plan = msg
    _last_plan_stamp = rospy.Time.now()

_diag_sub1 = None  # GlobalPlanner topic
_diag_sub2 = None  # ThetaStarPlanner topic
_diag_sub3 = None  # move_base/global_plan


def print_diagnostics(name):
    """卡死时打印诊断信息"""
    rospy.logwarn(f"  [诊断] === {name} 卡死诊断 ===")

    # 1. move_base action 状态
    state = S.client.get_state()
    rospy.logwarn(f"  [诊断] move_base state: {goal_state_name(state)}")

    # 2. 全局规划
    if _last_plan and (rospy.Time.now() - _last_plan_stamp).to_sec() < 10:
        plan_len = 0.0
        for i in range(1, len(_last_plan.poses)):
            dx = _last_plan.poses[i].pose.position.x - _last_plan.poses[i-1].pose.position.x
            dy = _last_plan.poses[i].pose.position.y - _last_plan.poses[i-1].pose.position.y
            plan_len += math.hypot(dx, dy)
        rospy.logwarn(f"  [诊断] 全局规划: {len(_last_plan.poses)}点, {plan_len:.2f}m "
                      f"(距今{(rospy.Time.now() - _last_plan_stamp).to_sec():.1f}s)")
    else:
        rospy.logwarn(f"  [诊断] 全局规划: ⚠ 无有效规划! (Theta* 或 GlobalPlanner 均未发布plan)")

    # 3. odom 位置
    rospy.logwarn(f"  [诊断] odom位置: ({S.odom_x:.3f}, {S.odom_y:.3f}) "
                  f"累计移动: {S.odom_dist:.3f}m")

    # 4. cmd_vel
    try:
        cmd = rospy.wait_for_message('/cmd_vel', Twist, timeout=1.0)
        rospy.logwarn(f"  [诊断] /cmd_vel: vx={cmd.linear.x:.4f} wz={cmd.angular.z:.4f}")
    except:
        rospy.logwarn(f"  [诊断] /cmd_vel: 无数据")

    rospy.logwarn(f"  [诊断] ================")


def is_cone_zone_goal(name):
    return name in CONE_ZONE_NAMES or name.startswith("mp")


def publish_cone_context(name):
    try:
        rospy.set_param(CONE_ZONE_PARAM, bool(is_cone_zone_goal(name)))
        rospy.set_param(CURRENT_TARGET_PARAM, name)
    except Exception:
        pass


def clear_cone_context():
    try:
        rospy.set_param(CONE_ZONE_PARAM, False)
        rospy.set_param(CURRENT_TARGET_PARAM, "")
    except Exception:
        pass


def cb_odom(msg):
    global _odom_yaw
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    q = msg.pose.pose.orientation
    _odom_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))
    px, py = S.odom_prev
    d = math.hypot(x - px, y - py)
    S.odom_dist += d
    S.odom_prev = (x, y)
    S.odom_x = x
    S.odom_y = y


STATUS_NAMES = {
    GoalStatus.PENDING:   "PENDING",
    GoalStatus.ACTIVE:    "ACTIVE",
    GoalStatus.PREEMPTED: "PREEMPTED",
    GoalStatus.SUCCEEDED: "SUCCEEDED",
    GoalStatus.ABORTED:   "ABORTED",
    GoalStatus.REJECTED:  "REJECTED",
    GoalStatus.RECALLED:  "RECALLED",
    GoalStatus.LOST:      "LOST",
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



def stop_robot(duration=0.4):
    """同时向 /cmd_vel_raw 和 /cmd_vel 发零速度，确保底盘停止。"""
    pubs = [
        rospy.Publisher('/cmd_vel_raw', Twist, queue_size=1),
        rospy.Publisher('/cmd_vel', Twist, queue_size=1),
    ]
    rospy.sleep(0.05)
    msg = Twist()
    end_t = time.time() + duration
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.time() < end_t:
        for pub in pubs:
            pub.publish(msg)
        rate.sleep()


def norm_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def pose7_to_yaw(pose7):
    q = [pose7[3], pose7[4], pose7[5], pose7[6]]
    _, _, yaw = tf.transformations.euler_from_quaternion(q)
    return yaw


def update_robot_pose():
    """更新机器人在 map 坐标系下的位置和yaw。"""
    if S.tf_listener is None:
        return False
    try:
        S.tf_listener.waitForTransform("map", "base_link", rospy.Time(0), rospy.Duration(0.2))
        (trans, rot) = S.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
        S.map_x = trans[0]
        S.map_y = trans[1]
        _, _, yaw = tf.transformations.euler_from_quaternion(rot)
        S.map_yaw = yaw
        return True
    except Exception as e:
        rospy.logwarn_throttle(2.0, "  [定位] 无法获取 map->base_link: %s", str(e))
        return False


def get_goal_dist(goal_x, goal_y):
    if not update_robot_pose():
        return 999.0
    return math.hypot(goal_x - S.map_x, goal_y - S.map_y)


def goal_position_reached(goal_x, goal_y, name=None):
    tol = WALL_GOAL_XY_TOL if name in WALL_GOAL_NAMES else NORMAL_GOAL_XY_TOL
    dist = get_goal_dist(goal_x, goal_y)
    return dist <= tol, dist, tol


def rotate_to_yaw(target_yaw, label="turn", yaw_tol_deg=2.0, timeout=15.0):
    """由本节点原地转到指定 map yaw；要求连续稳定在容差内，避免提前结束。"""
    pub_raw = rospy.Publisher('/cmd_vel_raw', Twist, queue_size=1)
    rospy.sleep(0.1)

    yaw_tol = math.radians(yaw_tol_deg)
    max_w = rospy.get_param("~turn_max_angular_vel", 0.80)
    min_w = rospy.get_param("~turn_min_angular_vel", 0.06)
    kp = rospy.get_param("~turn_kp", 1.5)
    stable_need = int(rospy.get_param("~turn_stable_frames", 8))
    stable_cnt = 0

    start_t = time.time()
    rate = rospy.Rate(25)
    rospy.loginfo(f"  [{label}] 严格原地转向开始: target_yaw={math.degrees(target_yaw):.1f}° tol={yaw_tol_deg:.1f}°")

    stuck_yaw = None       # 检测旋转卡住
    stuck_count = 0

    while not rospy.is_shutdown() and time.time() - start_t < timeout:
        if not update_robot_pose():
            stop_robot(0.1)
            rate.sleep()
            continue

        err = norm_angle(target_yaw - S.map_yaw)

        # 误差很小直接接受 (卡住时不强求到0误差)
        if abs(err) <= yaw_tol:
            stable_cnt += 1
            stop = Twist()
            pub_raw.publish(stop)
            if stable_cnt >= stable_need:
                stop_robot(0.6)
                update_robot_pose()
                final_err = norm_angle(target_yaw - S.map_yaw)
                rospy.loginfo(f"  [{label}] ✓ 朝向完成 yaw={math.degrees(S.map_yaw):.1f}° err={math.degrees(final_err):+.1f}°")
                return True
            rate.sleep()
            continue
        else:
            stable_cnt = 0

        # 卡住检测: yaw连续多帧不变 → min_w太低, 接受当前朝向
        if stuck_yaw is not None and abs(S.map_yaw - stuck_yaw) < math.radians(0.5):
            stuck_count += 1
            if stuck_count > 25 and abs(err) <= math.radians(5.0):  # 约1秒未转动，且误差≤5°才允许接受
                rospy.logwarn(f"  [{label}] 旋转接近目标但底盘响应不足(yaw={math.degrees(S.map_yaw):.1f}°未变), "
                              f"当前误差{math.degrees(err):.1f}°，允许结束")
                return True
        else:
            stuck_count = 0
        stuck_yaw = S.map_yaw

        # 分段P控制：大误差快转，小误差慢转
        abs_err = abs(err)
        if abs_err < math.radians(8.0):
            wz = 0.75 * err
        else:
            wz = kp * err
        if wz > max_w:
            wz = max_w
        elif wz < -max_w:
            wz = -max_w
        # 小误差时不用min_w强制加速，让机器人自然停下
        if abs(wz) < min_w and abs_err > math.radians(5.0):
            wz = min_w if wz > 0 else -min_w

        msg = Twist()
        msg.angular.z = round(wz, 3)
        pub_raw.publish(msg)
        rospy.loginfo_throttle(0.8, f"  [{label}] turning yaw={math.degrees(S.map_yaw):.1f}° err={math.degrees(err):+.1f}° wz={msg.angular.z:+.2f}")
        rate.sleep()

    stop_robot(0.6)
    update_robot_pose()
    final_err = norm_angle(target_yaw - S.map_yaw)
    rospy.logwarn(f"  [{label}] 朝向调整超时 yaw={math.degrees(S.map_yaw):.1f}° err={math.degrees(final_err):+.1f}°，继续后续路线")
    return False


def get_current_pose7():
    """获取机器人当前的 map 坐标系下 7 坐标 (x,y,z,qx,qy,qz,qw)。"""
    if not update_robot_pose():
        return None
    try:
        (trans, rot) = S.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
        return [trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]]
    except:
        return None


def finish_goal(name, pose7, align_yaw=True, hold_sec=None):
    """到达航点后的统一收尾：停、微调位置、严格对角、等待。"""
    cancel_all()
    stop_robot(0.5)
    if hold_sec is None:
        hold_sec = ARRIVAL_HOLD_SEC

    xy_tol = WALL_GOAL_XY_TOL if name in WALL_GOAL_NAMES else NORMAL_GOAL_XY_TOL

    # 旋转对齐前先微调位置(旋转会引入漂移)
    for attempt in range(2):
        update_robot_pose()
        dx = pose7[0] - S.map_x
        dy = pose7[1] - S.map_y
        dist = math.hypot(dx, dy)
        if dist <= xy_tol:
            break
        # 朝目标方向微移
        rospy.loginfo(f"  [{name}] 位置微调: 偏差{dist:.3f}m → 修正")
        target_yaw = math.atan2(dy, dx)
        rotate_to_yaw(target_yaw, label=f"{name}_pos_adj", yaw_tol_deg=8.0, timeout=5.0)
        nudge_to_target(dx, dy, target_yaw)

    # 严格对准最终朝向
    if align_yaw and STRICT_FINAL_YAW:
        rotate_to_yaw(pose7_to_yaw(pose7), label=f"{name}_final_yaw", yaw_tol_deg=FINAL_YAW_TOL_DEG, timeout=FINAL_YAW_TIMEOUT)

    # 旋转后再检查位置
    update_robot_pose()
    dx2 = pose7[0] - S.map_x
    dy2 = pose7[1] - S.map_y
    dist2 = math.hypot(dx2, dy2)

    # 打印真实到达坐标
    actual = get_current_pose7()
    if actual:
        rospy.loginfo(f"  [{name}] 📍 真实坐标: [{actual[0]:.4f}, {actual[1]:.4f}, {actual[2]:.2f},  "
                      f"{actual[3]:.4f}, {actual[4]:.4f}, {actual[5]:.4f}, {actual[6]:.4f}]")
    rospy.loginfo(f"  [{name}] 🎯 目标坐标: [{pose7[0]:.4f}, {pose7[1]:.4f}, {pose7[2]:.2f},  "
                  f"{pose7[3]:.4f}, {pose7[4]:.4f}, {pose7[5]:.4f}, {pose7[6]:.4f}] "
                  f"(偏差 {dist2*100:.1f}cm)")

    if hold_sec > 0:
        rospy.loginfo(f"  [{name}] 到点停车等待 {hold_sec:.1f}s")
        stop_robot(hold_sec)
    return True


def nudge_to_target(dx, dy, target_yaw, max_dist=0.20, speed=0.06):
    """朝目标方向微移一小段，用于修正旋转引入的位置漂移。"""
    dist = math.hypot(dx, dy)
    if dist < 0.02:
        return
    duration = min(dist, max_dist) / speed
    pub_raw = rospy.Publisher('/cmd_vel_raw', Twist, queue_size=1)
    rospy.sleep(0.1)
    rospy.loginfo(f"    [nudge] 微移{dist:.3f}m @ {speed:.2f}m/s, {duration:.1f}s")
    start = time.time()
    rate = rospy.Rate(20)
    while time.time() - start < duration and not rospy.is_shutdown():
        msg = Twist()
        msg.linear.x = speed
        pub_raw.publish(msg)
        rate.sleep()
    stop_robot(0.5)


def rear_clearance():
    """读取机器人后方扇区最近距离。小于 REAR_OBSTACLE_STOP_DIST 时禁止倒退。

    使用后方两个扇区：110°~180° 和 -180°~-110°。
    如果没有雷达数据，返回 10.0，避免因为暂时无数据导致恢复逻辑完全失效。
    """
    global _scan_data
    if _scan_data is None:
        return 10.0

    rear_left = sector_min(_scan_data, 110.0, 180.0, inflate=0.0)
    rear_right = sector_min(_scan_data, -180.0, -110.0, inflate=0.0)
    return min(rear_left, rear_right)


def backup_distance(distance=0.10, speed=-0.08, timeout=3.0):
    """卡死后安全倒退：后方 5cm 内有障碍则不退/立即停。"""
    pub_raw = rospy.Publisher('/cmd_vel_raw', Twist, queue_size=1)
    rospy.sleep(0.1)
    distance = abs(distance)
    speed = -abs(speed)
    S.odom_dist = 0.0
    S.odom_prev = (S.odom_x, S.odom_y)
    rear = rear_clearance()
    rospy.logwarn(f"  [恢复] 准备倒退 {distance:.2f}m，后方最近障碍={rear:.3f}m，阈值={REAR_OBSTACLE_STOP_DIST:.3f}m")

    if rear <= REAR_OBSTACLE_STOP_DIST:
        rospy.logwarn("  [恢复] 后方障碍 ≤ 5cm，禁止倒退，直接停车并交给后续清图/重规划")
        stop_robot(0.5)
        return False

    start_t = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.time() - start_t < timeout:
        if S.odom_dist >= distance:
            break

        rear = rear_clearance()
        if rear <= REAR_OBSTACLE_STOP_DIST:
            rospy.logwarn(f"  [恢复] 倒退中检测到后方障碍 {rear:.3f}m ≤ 5cm，立即停止")
            break

        msg = Twist()
        msg.linear.x = speed
        pub_raw.publish(msg)
        rate.sleep()

    stop_robot(0.5)
    rospy.logwarn(f"  [恢复] 倒退结束，实际移动 {S.odom_dist:.3f}m，后方最近障碍={rear_clearance():.3f}m")
    return S.odom_dist >= distance * 0.6


def set_initial_pose(x, y, z, qx, qy, qz, qw):
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(1.0)

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = rospy.Time.now()
    msg.pose.pose.position.x = x
    msg.pose.pose.position.y = y
    msg.pose.pose.position.z = z
    msg.pose.pose.orientation.x = qx
    msg.pose.pose.orientation.y = qy
    msg.pose.pose.orientation.z = qz
    msg.pose.pose.orientation.w = qw

    cov = [0.0] * 36
    cov[0]  = 0.0025; cov[7]  = 0.0025; cov[14] = 0.0685
    cov[21] = 0.0685; cov[28] = 0.0685; cov[35] = 0.0076
    msg.pose.covariance = cov

    for _ in range(3):
        pub.publish(msg)
        rospy.sleep(0.3)

    yaw = 2 * math.atan2(qz, qw)
    rospy.loginfo(f"初始位姿: ({x:.3f}, {y:.3f}, yaw≈{math.degrees(yaw):.0f}°)")


# ============================================================
#  宽道恢复
# ============================================================
# ============================================================
#  LiDAR 全局共享 (reactive_avoidance 和诊断共用)
# ============================================================
_scan_data = None
_scan_stamp = rospy.Time(0)
_odom_yaw = 0.0

def cb_scan_shared(msg):
    global _scan_data, _scan_stamp
    _scan_data = msg
    _scan_stamp = rospy.Time.now()

_scan_sub = None


# ============================================================
#  反应式避障 (移植自 avoid.py 的逻辑)
#
#  核心机制:
#    1. 分角度膨胀: 前方膨胀0.22m, 侧面膨胀0.38m
#    2. 方向锁定(tdir): 选定绕行方向后锁定, 防止左右摇摆
#    3. 渐进解锁: 前方开阔 + (朝向目标 或 锁太久) → 解锁
#    4. 目标牵引: 无锁定且开阔时, 主动转向目标
#    5. 速度平滑: EMA 平滑过渡
#    6. 三段状态: DANGER(后退+猛转) / WARN(慢进+偏转) / CLEAR(朝目标)
# ============================================================
def sector_min(scan, angle_min_deg, angle_max_deg, inflate=0.0):
    """
    获取 LiDAR 指定角度扇形内膨胀后的最近距离.
      scan:           LaserScan 消息
      angle_min_deg:  扇形起始角度 (度)
      angle_max_deg:  扇形结束角度 (度)
      inflate:        膨胀距离 (m), 从测量值中减去
    返回: 膨胀后的最近距离 (m), 10.0 表示无有效数据
    """
    if scan is None or len(scan.ranges) == 0:
        return 10.0
    min_rad = math.radians(angle_min_deg)
    max_rad = math.radians(angle_max_deg)
    # 确保 min_rad <= max_rad
    if min_rad > max_rad:
        min_rad, max_rad = max_rad, min_rad
    best = 10.0
    for i, d in enumerate(scan.ranges):
        ang = scan.angle_min + i * scan.angle_increment
        if min_rad <= ang <= max_rad:
            if scan.range_min < d < scan.range_max:
                inflated = max(0.03, d - inflate)
                if inflated < best:
                    best = inflated
    return best


def reactive_avoidance(goal_x, goal_y, max_duration=15.0):
    """
    反应式避障主循环.
    参数:
      goal_x, goal_y:  目标地图坐标
      max_duration:    最长运行时间 (秒)
    返回:
      True  如果到达目标附近
      False 如果超时
    """
    global _scan_data

    # ---- 参数 (同 avoid.py) ----
    FRONT_INF = 0.22   # 前方膨胀 (机器人本体)
    SIDE_INF  = 0.38   # 侧面膨胀 (本体+底座)
    MAX_V     = rospy.get_param("~avoid_max_v", 0.40)   # 最大线速度, 途中不慢
    WARN_F    = rospy.get_param("~avoid_warn_front", 0.40)   # 膨胀后前方预警阈值
    DANG_F    = rospy.get_param("~avoid_danger_front", 0.22)   # 膨胀后前方危险阈值
    WARN_S    = rospy.get_param("~avoid_warn_side", 0.30)   # 膨胀后侧面预警阈值

    tdir = 0           # -1=左绕, 0=无锁定, +1=右绕
    lock_cnt = 0       # 锁定持续帧数
    svx = 0.0          # 平滑线速度
    svz = 0.0          # 平滑角速度

    pub_cmd = rospy.Publisher(rospy.get_param('~avoid_cmd_vel_topic', '/cmd_vel_raw'), Twist, queue_size=1)
    # 等 publisher 注册
    rospy.sleep(0.3)

    rospy.logwarn(f"  [反应式] 启动! 目标=({goal_x:.2f},{goal_y:.2f}) 最长{max_duration:.0f}s")

    start_t = time.time()
    rate = rospy.Rate(15)  # 同 avoid.py

    while not rospy.is_shutdown():
        elapsed = time.time() - start_t
        if elapsed > max_duration:
            rospy.logwarn(f"  [反应式] 超时 {max_duration:.0f}s, 退出")
            pub_cmd.publish(Twist())
            return False

        # ---- 获取 LiDAR ----
        if _scan_data is None:
            rospy.sleep(0.1)
            continue

        # LIDAR 数据传感器检查
        if (rospy.Time.now() - _scan_stamp).to_sec() > 1.0:
            rospy.logwarn_throttle(1.0, "  [反应式] LiDAR超时")
            pub_cmd.publish(Twist())
            rate.sleep()
            continue

        scan = _scan_data
        raw_f = sector_min(scan, -10.0, 10.0, inflate=0.0)
        f30   = sector_min(scan, -30.0, 30.0, inflate=FRONT_INF)
        L     = sector_min(scan,  30.0, 90.0, inflate=SIDE_INF)
        R     = sector_min(scan, -90.0, -30.0, inflate=SIDE_INF)
        rospy.loginfo_throttle(8.0,
            "[反应式] scan角度范围: %.1f°~%.1f° | front=%.2f L=%.2f R=%.2f",
            math.degrees(scan.angle_min), math.degrees(scan.angle_max), raw_f, L, R)

        # ---- 目标朝向：必须使用 map 位姿，而不是 odom ----
        if not update_robot_pose():
            pub_cmd.publish(Twist())
            rate.sleep()
            continue
        dx = goal_x - S.map_x
        dy = goal_y - S.map_y
        dist = math.hypot(dx, dy)
        gdeg = math.degrees(math.atan2(dy, dx) - S.map_yaw)
        gdeg = (gdeg + 180) % 360 - 180

        # ---- 距目标越近，避障越弱、速度越低 ----
        # 只在最后0.5m减速靠站，途中保持正常速度
        if dist < NORMAL_GOAL_XY_TOL:
            rospy.logwarn(f"  [反应式] 距目标仅{dist:.2f}m，退出避障")
            pub_cmd.publish(Twist())
            stop_robot(0.2)
            return True

        # 最后0.5m开始渐变减速靠站
        approach_zone = 0.50
        approach_factor = max(0.0, min(1.0, (approach_zone - dist) / (approach_zone - NORMAL_GOAL_XY_TOL)))
        speed_scale = 1.0 - approach_factor * 0.80        # 途中1.0 → 靠站0.20
        turn_scale  = 1.0 - approach_factor * 0.70        # 途中1.0 → 靠站0.30
        goal_pull   = approach_factor * 3.0               # 靠站时强力牵引朝向

        tvx = 0.0
        tvz = 0.0

        # ===== 状态机 (同 avoid.py) + 距离渐变 =====

        if f30 < DANG_F:
            # 危险: 远目标时后退+硬转, 近目标(<0.3m)时只微速朝目标挪
            if dist < 0.30:
                tvx = 0.03
                tvz = math.radians(gdeg) * 1.5
                tdir = 0; lock_cnt = 0
            else:
                tvx = -0.10 * speed_scale
                if tdir == 0:
                    tdir = 1 if L > R else -1
                tvz = 0.8 * tdir * turn_scale
                lock_cnt += 1

        elif f30 < WARN_F or (tdir != 0 and min(L, R) < WARN_S):
            # 预警: 慢进 + 偏转，近目标加goal牵引
            tvx = 0.06 * speed_scale
            if tdir == 0:
                tdir = 1 if L > R else -1
            tvz = 0.8 * tdir * turn_scale + math.radians(gdeg) * goal_pull
            lock_cnt += 1

        elif tdir != 0:
            # 开阔但仍锁定: 近目标更容易解锁
            near_unlock = lock_cnt > 30 or dist < 0.40
            if (raw_f > 0.6 and abs(gdeg) < 70) or near_unlock:
                tdir = 0
                lock_cnt = 0
            else:
                tvx = 0.12 * speed_scale
                tvz = 0.8 * tdir * turn_scale + math.radians(gdeg) * goal_pull * 0.5
                lock_cnt += 1
        else:
            # 无锁定 + 开阔: 朝目标走，越近越慢
            tvx = min(MAX_V, raw_f * 0.5) * speed_scale
            tvx = max(0.03, tvx)
            tvz = math.radians(gdeg) * (3.0 + goal_pull)
            tvz = max(-0.9, min(0.9, tvz))

        # ---- 速度平滑 ----
        svx = 0.45 * tvx + 0.55 * svx
        svz = 0.45 * tvz + 0.55 * svz

        t = Twist()
        t.linear.x = round(svx, 3)
        t.angular.z = round(svz, 3)
        pub_cmd.publish(t)

        # 状态符
        m = "⚡" if f30 < DANG_F else ("~" if tdir != 0 else "→")
        rospy.loginfo_throttle(2.0,
            "%s f=%.2f raw=%.2f L=%.2f R=%.2f g=%+d° d=%.2f dir=%+d lock=%d vx=%.2f wz=%+.2f",
            m, f30, raw_f, L, R, int(gdeg), dist, tdir, lock_cnt, t.linear.x, t.angular.z)

        rate.sleep()

    return False


# ============================================================
#  核心导航
# ============================================================
def goto_point(name, pose7, hard_timeout=15.0):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(
        Point(pose7[0], pose7[1], pose7[2]),
        Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))

    attempt = 0
    while True:  # 不允许跳点，无限重试直到到达
        cone_zone = is_cone_zone_goal(name)
        S.odom_dist = 0.0
        S.odom_prev = (S.odom_x, S.odom_y)
        stuck_dist_at = 0.0
        stuck_time_at = time.time()
        start_t = time.time()
        grace_period = 3.0

        rospy.loginfo(f"  [{name}] → ({pose7[0]:.3f}, {pose7[1]:.3f}) "
                      f"timeout={hard_timeout}s {'(重试' + str(attempt) + ')' if attempt > 0 else ''}")
        S.client.send_goal(goal)

        rate = rospy.Rate(4)
        while not rospy.is_shutdown():
            now_t = time.time()
            elapsed = now_t - start_t

            reached_by_pos, pos_dist, tol_used = goal_position_reached(pose7[0], pose7[1], name=name)
            if reached_by_pos:
                rospy.loginfo(f"  [{name}] ✓ 位置到达 dist={pos_dist:.2f}m ≤ tol={tol_used:.2f}m")
                # d1 后会执行 d1t 朝向，因此 d1 本身不先转到180°，避免重复旋转。
                return finish_goal(name, pose7, align_yaw=True, hold_sec=ARRIVAL_HOLD_SEC)

            state = S.client.get_state()
            if state == GoalStatus.SUCCEEDED:
                rospy.loginfo(f"  [{name}] ✓ move_base到达 ({S.odom_dist:.2f}m, {elapsed:.1f}s)")
                return finish_goal(name, pose7, align_yaw=True, hold_sec=ARRIVAL_HOLD_SEC)
            if state in (GoalStatus.ABORTED, GoalStatus.REJECTED,
                         GoalStatus.RECALLED, GoalStatus.PREEMPTED):
                rospy.logwarn(f"  [{name}] ✗ move_base 终止 ({goal_state_name(state)})")
                break

            if elapsed >= hard_timeout:
                rospy.logwarn(f"  [{name}] ✗ 超时 {hard_timeout}s (移动 {S.odom_dist:.2f}m)")
                break

            if elapsed < grace_period:
                if elapsed >= grace_period - 0.5:
                    stuck_dist_at = S.odom_dist
                    stuck_time_at = now_t
                rate.sleep()
                continue

            stuck_window = CONE_STUCK_WINDOW if cone_zone else 3.0
            stuck_min_dist = CONE_STUCK_MIN_DIST if cone_zone else 0.05
            window_elapsed = now_t - stuck_time_at
            if window_elapsed >= stuck_window:
                moved = S.odom_dist - stuck_dist_at
                if moved < stuck_min_dist:
                    rospy.logwarn(f"  [{name}] ✗ 卡死! {stuck_window:.1f}s 仅移动 {moved:.3f}m "
                                  f"(总移动{S.odom_dist:.2f}m 耗时{elapsed:.1f}s)")
                    print_diagnostics(name)
                    break
                stuck_dist_at = S.odom_dist
                stuck_time_at = now_t

            rate.sleep()

        cancel_all()
        rospy.sleep(0.3)
        print_diagnostics(name)

        goal_pose = pose7
        # 距目标30cm内不再绕障，直接对准朝向停车
        near_dist = get_goal_dist(goal_pose[0], goal_pose[1])
        if near_dist <= NO_AVOID_NEAR_GOAL:
            rospy.logwarn(f"  [{name}] 距目标仅{near_dist:.2f}m ≤ {NO_AVOID_NEAR_GOAL:.2f}m，不绕障，直接朝向停车")
            return finish_goal(name, pose7, align_yaw=True, hold_sec=ARRIVAL_HOLD_SEC)

        if (REACTIVE_RECOVERY_ENABLED and cone_zone and
                attempt < REACTIVE_RECOVERY_ATTEMPTS):
            rospy.logwarn(f"  [{name}] 锥桶段恢复: 先启动反应式绕障 "
                          f"({attempt + 1}/{REACTIVE_RECOVERY_ATTEMPTS})")
            clear_costmaps()
            rospy.sleep(0.2)
            if reactive_avoidance(goal_pose[0], goal_pose[1],
                                  max_duration=REACTIVE_RECOVERY_DURATION):
                rospy.logwarn(f"  [{name}] 反应式绕障已到达目标附近")
                return finish_goal(name, pose7, align_yaw=True, hold_sec=ARRIVAL_HOLD_SEC)
            reached_now, pos_dist_now, tol_now = goal_position_reached(
                goal_pose[0], goal_pose[1], name=name)
            if reached_now:
                rospy.logwarn(f"  [{name}] 反应式结束后位置已到达 "
                              f"dist={pos_dist_now:.2f}m ≤ tol={tol_now:.2f}m")
                return finish_goal(name, pose7, align_yaw=True, hold_sec=ARRIVAL_HOLD_SEC)
            stop_robot(0.3)
            clear_costmaps()
            rospy.sleep(0.4)
            attempt += 1
            continue

        # 卡死恢复：倒退 → 旋转一定角度(优先右转3次, 之后左右交替) → 清图 → 重规划
        if BACKUP_ON_STUCK:
            backup_distance(BACKUP_DIST, BACKUP_SPEED,
                            timeout=max(2.0, BACKUP_DIST / max(abs(BACKUP_SPEED), 0.02) + 1.0))
            rospy.sleep(0.3)

        if update_robot_pose():
            face_yaw = math.atan2(goal_pose[1] - S.map_y, goal_pose[0] - S.map_x)
            offset_angles = [-30, -60, -90, 30, -30, 60, -60, 90, -90]
            offset = math.radians(offset_angles[attempt % len(offset_angles)])
            target_yaw = face_yaw + offset
            rospy.logwarn(f"  [{name}] 倒退后旋转{offset_angles[attempt % len(offset_angles)]}°(偏移), 重新规划")
            rotate_to_yaw(target_yaw, label=f"{name}_recover", yaw_tol_deg=8.0, timeout=5.0)
        clear_costmaps()
        rospy.sleep(0.5)
        attempt += 1
        continue
    # unreachable


def cruise():
    total = len(S.patrol_path)

    for idx, name in enumerate(S.patrol_path):
        if rospy.is_shutdown():
            break
        rospy.loginfo(f"--- [{idx + 1}/{total}] {name} ---")
        pose = S.nav_points[name]
        is_first = (idx == 0)
        is_midpoint = name.startswith("mp")
        cone_zone = is_cone_zone_goal(name)
        timeout = 20.0 if is_first else (10.0 if cone_zone else 15.0)

        publish_cone_context(name)
        goto_point(name, pose, hard_timeout=timeout)

        if name in TURN_AFTER_POINTS:
            turn_name = TURN_AFTER_POINTS[name]
            turn_pose = S.nav_points[turn_name]
            rospy.loginfo(f"  [{name}] 停车完成，开始执行 {turn_name} 严格原地转向")
            rotate_to_yaw(pose7_to_yaw(turn_pose), label=turn_name,
                          yaw_tol_deg=FINAL_YAW_TOL_DEG, timeout=FINAL_YAW_TIMEOUT)
            stop_robot(0.5)
            clear_costmaps()
            rospy.sleep(0.5)
        rospy.sleep(0.3)

    clear_cone_context()
    rospy.loginfo("=" * 60)
    rospy.loginfo(f"巡航结束: 全部 {total} 个航点完成")
    rospy.loginfo("=" * 60)


def on_shutdown(sig=None, frame=None):
    rospy.loginfo("正在退出...")
    try:
        clear_cone_context()
        cancel_all()
    except:
        pass
    rospy.signal_shutdown("user_exit")
    sys.exit(0)


if __name__ == '__main__':
    rospy.init_node('cruise_cone_v1')
    signal.signal(signal.SIGINT, on_shutdown)

    rospy.loginfo("连接 move_base action server ...")
    S.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not S.client.wait_for_server(rospy.Duration(5)):
        rospy.logerr("无法连接 move_base!")
        sys.exit(1)
    rospy.loginfo("move_base 已连接")

    S.tf_listener = tf.TransformListener()
    rospy.sleep(0.5)

    # 初始位姿
    start = S.nav_points["start"]
    set_initial_pose(*start)
    rospy.sleep(2.0)

    # 诊断订阅: 监听所有可能的全局规划话题
    _diag_sub1 = rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, cb_diag_plan)
    _diag_sub2 = rospy.Subscriber('/move_base/ThetaStarPlanner/plan', Path, cb_diag_plan)
    _diag_sub3 = rospy.Subscriber('/move_base/NavfnROS/plan', Path, cb_diag_plan)

    # LiDAR 共享订阅 (反应式避障用)
    _scan_sub = rospy.Subscriber('/scan', LaserScan, cb_scan_shared)

    # odom (同时更新全局 _odom_yaw 给反应式避障用)
    rospy.Subscriber('/odom', Odometry, cb_odom)
    rospy.sleep(1.0)
    clear_cone_context()
    update_robot_pose()
    rospy.loginfo(f"当前map位姿: ({S.map_x:.3f}, {S.map_y:.3f}, yaw={math.degrees(S.map_yaw):.1f}°)")
    rospy.loginfo(f"严格朝向: {STRICT_FINAL_YAW}, yaw容差={FINAL_YAW_TOL_DEG:.1f}°, 到点等待={ARRIVAL_HOLD_SEC:.1f}s, 卡死倒退={BACKUP_DIST:.2f}m, 近目标禁绕={NO_AVOID_NEAR_GOAL:.2f}m")

    rospy.loginfo("=" * 60)
    rospy.loginfo("宽道巡航 (锥桶场景)")
    rospy.loginfo(f"  start: ({start[0]:.2f}, {start[1]:.2f})")
    for name in S.patrol_path:
        p = S.nav_points[name]
        tag = " [锥桶区]" if name.startswith("mp") else ""
        rospy.loginfo(f"  {name}:   ({p[0]:.2f}, {p[1]:.2f}){tag}")
    rospy.loginfo(f"  锥桶区中间点: {len(S.cone_midpoints)} 个")
    rospy.loginfo("=" * 60)

    cruise()
    rospy.loginfo("任务结束, 节点保持运行 (Ctrl-C 退出)")
    rospy.spin()
