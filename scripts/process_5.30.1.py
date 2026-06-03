#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import actionlib
import signal
import sys
import tf2_ros

from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, Twist
from actionlib_msgs.msg import GoalID
from std_srvs.srv import Empty


# ---------------------------- 全局共享变量 ----------------------------
class SharedVariables:
    def __init__(self):
        self.move_base = None  # 等 init_node 后再创建
        self.tf_buffer = None
        self.tf_listener = None

        self.nav_point = {
            "s0":  [1.54,  0.0,   0.0, 0.0, 0.0, 0.0, 1.0],
            "s0t":  [1.54,  0.0,   0.0, 0.0, 0.0, -0.7071, 0.7071],
            
            "s1":  [1.68, -0.42,  0.0, 0.0, 0.0, -0.7071, 0.7071],
            "s1t":  [1.68, -0.42,  0.0, 0.0, 0.0, 0.0, 1.0],

            "s2":  [2.07, -0.42,  0.0, 0.0, 0.0, 0.0, 1.0],
            "s2t":  [2.07, -0.42,  0.0, 0.0, 0.0, 0.7071, 0.7071],

            "s3":  [2.12, 0.01,  0.0, 0.0, 0.0, 0.7071, 0.7071],
            "s3t": [2.12, 0.01,  0.0, 0.0, 0.0, 0.0, 1.0],

            "s4":  [3.10, -0.01,  0.0, 0.0, 0.0, 0.0, 1.0],
            "s4t": [3.10, -0.01,  0.0, 0.0, 0.0, -0.7071, 0.7071],

            "s5":  [3.15, -0.42,  0.0, 0.0, 0.0, -0.7071, 0.7071],

            "s6":  [3.13, -0.89,  0.0, 0.0, 0.0, -0.7071, 0.7071],
            "s6t": [3.13, -0.89, 0.0, 0.0, 0.0, 1.0, 0.0],

            "s7":  [3.13, -0.91,  0.0, 0.0, 0.0, 1.0, 0.0],

            "s8":  [2.03, -0.95,  0.0, 0.0, 0.0, 1.0, 0.0],

            "s9":  [1.18, -0.95,  0.0, 0.0, 0.0, 1.0, 0.0],
            "s9t":  [1.18, -0.95,  0.0, 0.0, 0.0, 0.7071, 0.7071],

            "s10": [1.07, -0.54,  0.0, 0.0, 0.0, -0.7071, 0.707],
            "s10t": [1.07, -0.54,  0.0, 0.0, 0.0, 1.0, 0.0],

            "s11": [0.228, -0.537, 0.0, 0.0, 0.0, 1.0, 0.0],

            "s12": [-0.143, -0.13, 0.0, 0.0, 0.0, 1.0, 0.0],
        }

SV = SharedVariables()


# ---------------------------- 初始化 ----------------------------
def init_move_base():
    SV.move_base = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not SV.move_base.wait_for_server(rospy.Duration(5)):
        rospy.logerr("无法连接 move_base action server")
        sys.exit(1)
    rospy.loginfo("move_base action server 已连接")
    send_initialpose()

def send_initialpose():
    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    initial_pose = PoseWithCovarianceStamped()
    initial_pose.header.frame_id = "map"
    initial_pose.header.stamp = rospy.Time.now()
    initial_pose.pose.pose.position.x = 0.0
    initial_pose.pose.pose.position.y = 0.0
    initial_pose.pose.pose.position.z = 0.0
    initial_pose.pose.pose.orientation.x = 0.0
    initial_pose.pose.pose.orientation.y = 0.0
    initial_pose.pose.pose.orientation.z = 0.0
    initial_pose.pose.pose.orientation.w = 1.0
    rospy.sleep(1)
    pub.publish(initial_pose)
    rospy.loginfo("已发送初始位姿")

def init_tf_listener():
    SV.tf_buffer = tf2_ros.Buffer()
    SV.tf_listener = tf2_ros.TransformListener(SV.tf_buffer)
    rospy.sleep(1.0)
    rospy.loginfo("TF 监听器已初始化")


# ---------------------------- 工具函数 ----------------------------
def reset_navigation():
    """取消当前导航目标并清除代价地图"""
    # 先通过 action client 取消
    SV.move_base.cancel_all_goals()
    # 再发 cancel topic 确保送达
    cancel_pub = rospy.Publisher('/move_base/cancel', GoalID, queue_size=10)
    rospy.sleep(0.2)  # 等 publisher 建立连接
    cancel_msg = GoalID()
    cancel_pub.publish(cancel_msg)
    rospy.loginfo("Current navigation goal canceled.")

    # 重置代价地图
    rospy.wait_for_service('/move_base/clear_costmaps')
    try:
        clear_costmaps = rospy.ServiceProxy('/move_base/clear_costmaps', Empty)
        clear_costmaps()
        rospy.loginfo("Costmaps cleared.")
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def recovery_back_up():
    """轻微前后晃动 + 小角度转向，适合 0.5m 窄赛道"""
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    rospy.sleep(0.2)

    rospy.loginfo("Recovery: 轻微后退...")
    cmd = Twist()
    cmd.linear.x = -0.1
    pub.publish(cmd)
    rospy.sleep(0.8)  # 退 ~8cm

    rospy.loginfo("Recovery: 小角度转向...")
    cmd.linear.x = 0.0
    cmd.angular.z = 0.3
    pub.publish(cmd)
    rospy.sleep(1.0)  # 转 ~17°

    rospy.loginfo("Recovery: 轻微前进...")
    cmd.angular.z = 0.0
    cmd.linear.x = 0.1
    pub.publish(cmd)
    rospy.sleep(0.8)  # 进 ~8cm

    pub.publish(Twist())
    rospy.loginfo("Recovery done.")

def send_nav_point_and_wait(target_pose_list, waypoint_timeout=12, max_retries=3):
    """发送导航点，超时或失败后自动恢复重试，超过次数跳过"""
    if len(target_pose_list) != 7:
        rospy.logerr(f"导航点列表长度错误: {len(target_pose_list)}")
        return False

    name = "unknown"
    for k, v in SV.nav_point.items():
        if v == target_pose_list:
            name = k
            break

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(Point(*target_pose_list[:3]),
                                 Quaternion(*target_pose_list[3:]))

    for attempt in range(1, max_retries + 1):
        if rospy.is_shutdown():
            return False

        rospy.loginfo(f"前往 {name} (第 {attempt}/{max_retries} 次尝试)")
        SV.move_base.send_goal(goal)

        # 带超时的等待
        finished = SV.move_base.wait_for_result(rospy.Duration(waypoint_timeout))

        if finished:
            result = SV.move_base.get_result()
            if result:
                rospy.loginfo(f"成功到达 {name}")
                return True

        # 超时或失败 → 取消 + 恢复
        rospy.logwarn(f"{name} 超时/失败，取消目标并恢复...")
        SV.move_base.cancel_all_goals()
        rospy.sleep(0.3)

        if attempt < max_retries:
            reset_navigation()
            recovery_back_up()
            rospy.sleep(1)

    rospy.logerr(f"{name} 尝试 {max_retries} 次后仍失败，跳过此点")
    return False

def signal_handler(sig, frame):
    rospy.loginfo("收到中断信号，正在退出...")
    SV.move_base.cancel_all_goals()
    reset_navigation()          # 彻底清理
    rospy.signal_shutdown("用户终止")
    sys.exit(0)


# ---------------------------- 定点巡航任务 ----------------------------
def cruise_points(points):
    rospy.loginfo(f"开始定点巡航，共 {len(points)} 个点")
    skipped = 0
    for name in points:
        if rospy.is_shutdown():
            break
        if name not in SV.nav_point:
            rospy.logerr(f"导航点 '{name}' 不存在，跳过")
            continue
        ok = send_nav_point_and_wait(SV.nav_point[name])
        if not ok:
            skipped += 1
        rospy.sleep(0.5)
    rospy.loginfo(f"定点巡航完成 (跳过 {skipped} 个点)")


# ---------------------------- 主程序 ----------------------------
if __name__ == '__main__':
    rospy.init_node('cruise_mode')
    signal.signal(signal.SIGINT, signal_handler)

    rospy.loginfo("初始化 move_base ...")
    init_move_base()

    rospy.loginfo("初始化 TF 监听器 ...")
    init_tf_listener()

    rospy.sleep(2)
    rospy.loginfo("开始巡航")

    patrol_path = [
        "s0", "s0t",          # 到位 → 转朝 -y
        "s1", "s1t",          # 到位 → 转朝 +x
        "s2", "s2t",          # 到位 → 转朝 +y
        "s3", "s3t",          # 到位 → 转朝 +x
        "s4", "s4t",          # 到位 → 转朝 -y
        "s5",
        "s6", "s6t",          # 到位 → 转朝 -x
        "s7", "s8",
        "s9", "s9t",          # 到位 → 转朝 +y
        "s10", "s10t",        # 到位 → 转朝 -x
        "s11",
        "s12",
    ]

    cruise_points(patrol_path)
    rospy.loginfo("任务结束，节点保持运行...")
    rospy.spin()