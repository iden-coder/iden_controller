#!/usr/bin/env python3
"""
实时导航监控 — 记录所有关键数据供分析

用法 (另开终端):
  rosrun iden_controller monitor_nav.py

会在 /tmp/nav_monitor.log 记录完整数据。
Ctrl-C 停止后自动打印摘要。
"""

import rospy
import math
import time
import json
import os
from collections import deque
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry, Path
from actionlib_msgs.msg import GoalStatusArray


class NavMonitor:
    def __init__(self):
        rospy.init_node('nav_monitor')

        self.log_path = '/tmp/nav_monitor.log'
        self.log_fd = open(self.log_path, 'w')

        # 统计
        self.t0 = time.time()
        self.cmd_vel_history = deque(maxlen=1000)
        self.scan_history = deque(maxlen=100)
        self.odom_history = deque(maxlen=1000)
        self.path_history = deque(maxlen=10)
        self.goal_history = []
        self.stuck_count = 0
        self.stop_count = 0
        self.recovery_count = 0
        self.last_odom = None
        self.odom_total = 0.0
        self.goal_changes = 0
        self.last_goal_id = None

        # 订阅
        rospy.Subscriber('/cmd_vel', Twist, self.cb_cmd, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.cb_scan, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.cb_odom, queue_size=1)
        rospy.Subscriber('/move_base/status', GoalStatusArray, self.cb_status, queue_size=1)
        rospy.Subscriber('/move_base/current_goal', PoseStamped, self.cb_goal, queue_size=1)
        rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.cb_global_plan, queue_size=1)
        rospy.Subscriber('/move_base/GlobalPlanner/plan', Path, self.cb_global_plan, queue_size=1)

        # 每秒打印
        rospy.Timer(rospy.Duration(1.0), self.cb_timer)

        self.log("=== 导航监控启动 ===")
        self.log(f"日志: {self.log_path}")

    def log(self, msg):
        t = time.time() - self.t0
        line = f"[{t:7.1f}s] {msg}"
        print(line)
        self.log_fd.write(line + '\n')
        self.log_fd.flush()

    def cb_cmd(self, msg):
        self.cmd_vel_history.append({
            't': time.time() - self.t0,
            'vx': msg.linear.x, 'wz': msg.angular.z
        })

    def cb_scan(self, msg):
        if len(msg.ranges) == 0:
            return
        n = len(msg.ranges)
        front = msg.ranges[n//2-20 : n//2+20]
        front_min = min([r for r in front if msg.range_min < r < msg.range_max], default=10.0)
        # 侧面
        left_min = min([r for r in msg.ranges[n*3//4 : n*9//10]
                        if msg.range_min < r < msg.range_max], default=10.0)
        right_min = min([r for r in msg.ranges[n//10 : n//4]
                         if msg.range_min < r < msg.range_max], default=10.0)

        self.scan_history.append({
            't': time.time() - self.t0,
            'front': front_min, 'left': left_min, 'right': right_min
        })

    def cb_odom(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.last_odom:
            dx = x - self.last_odom[0]
            dy = y - self.last_odom[1]
            self.odom_total += math.hypot(dx, dy)
        self.last_odom = (x, y)
        self.odom_history.append({
            't': time.time() - self.t0,
            'x': x, 'y': y, 'dist': self.odom_total
        })

    def cb_status(self, msg):
        for s in msg.status_list:
            if s.status == 4:  # ABORTED
                self.log(f"⚠ move_base ABORTED: goal={s.goal_id.id[-6:]} text={s.text}")

    def cb_goal(self, msg):
        gid = f"({msg.pose.position.x:.2f},{msg.pose.position.y:.2f})"
        if gid != self.last_goal_id:
            self.goal_changes += 1
            self.last_goal_id = gid
            self.log(f"🎯 新目标: {gid}")

    def cb_global_plan(self, msg):
        length = 0.0
        for i in range(1, len(msg.poses)):
            dx = msg.poses[i].pose.position.x - msg.poses[i-1].pose.position.x
            dy = msg.poses[i].pose.position.y - msg.poses[i-1].pose.position.y
            length += math.hypot(dx, dy)
        self.path_history.append({
            't': time.time() - self.t0,
            'len': length, 'pts': len(msg.poses)
        })
        self.log(f"🗺 全局规划: {len(msg.poses)}点, {length:.2f}m")

    def cb_timer(self, event):
        """每秒统计分析"""
        elapsed = time.time() - self.t0
        if elapsed < 2:
            return

        # 最近1秒的 cmd_vel
        recent = [c for c in self.cmd_vel_history if elapsed - c['t'] < 2.0]
        if recent:
            avg_vx = sum(abs(c['vx']) for c in recent) / len(recent)
            avg_wz = sum(abs(c['wz']) for c in recent) / len(recent)
        else:
            avg_vx = avg_wz = 0.0

        # 最近 scan
        if self.scan_history:
            s = self.scan_history[-1]
            f, l, r = s['front'], s['left'], s['right']
        else:
            f = l = r = 10.0

        # 简单判断
        if avg_vx < 0.01 and avg_wz < 0.02 and elapsed > 5:
            self.stuck_count += 1
            if self.stuck_count == 3:
                self.log("⚠ 疑似卡死: 连续3秒无有效移动")
        else:
            if self.stuck_count >= 3:
                self.log(f"✓ 恢复移动 (卡了{self.stuck_count}秒)")
            self.stuck_count = 0

        # 每5秒详细状态
        if int(elapsed) % 5 == 0 and int(elapsed) != int(elapsed - 1):
            self.log(f"📊 t={elapsed:.0f}s | odom={self.odom_total:.2f}m | "
                     f"vx={avg_vx:.3f} wz={avg_wz:.3f} | "
                     f"front={f:.2f}m left={l:.2f}m right={r:.2f}m | "
                     f"goals={self.goal_changes}")

    def summary(self):
        elapsed = time.time() - self.t0
        print("\n" + "=" * 60)
        print("              导航监控摘要")
        print("=" * 60)

        # 速度分析
        if self.cmd_vel_history:
            vxs = [abs(c['vx']) for c in self.cmd_vel_history]
            wzs = [abs(c['wz']) for c in self.cmd_vel_history]
            zero_vx = sum(1 for v in vxs if v < 0.005)
            print(f"运行时间:       {elapsed:.0f}s")
            print(f"总移动距离:     {self.odom_total:.2f}m")
            print(f"平均速度:       {sum(vxs)/len(vxs):.3f} m/s")
            print(f"最大速度:       {max(vxs):.3f} m/s")
            print(f"零速时间占比:   {zero_vx/len(vxs)*100:.0f}%")
            print(f"目标切换次数:   {self.goal_changes}")

        # 障碍物分布
        if self.scan_history:
            fronts = [s['front'] for s in self.scan_history]
            print(f"前方最近障碍:   {min(fronts):.2f}m")
            print(f"前方平均障碍:   {sum(fronts)/len(fronts):.2f}m")
            near_frames = sum(1 for f in fronts if f < 0.5)
            print(f"前方<0.5m占比:  {near_frames/len(fronts)*100:.0f}%")

        # 路径分析
        if self.path_history:
            lengths = [p['len'] for p in self.path_history]
            print(f"全局路径数:     {len(self.path_history)}")
            print(f"平均路径长度:   {sum(lengths)/len(lengths):.2f}m")
            print(f"最短路径:       {min(lengths):.2f}m")

        # 诊断建议
        print("\n--- 诊断 ---")
        issues = []

        if self.cmd_vel_history:
            vxs = [abs(c['vx']) for c in self.cmd_vel_history]
            zero_ratio = sum(1 for v in vxs if v < 0.005) / len(vxs)
            if zero_ratio > 0.6:
                issues.append("🔴 机器人超过60%时间速度为0, 可能频繁卡死或move_base规划慢")
            elif zero_ratio > 0.3:
                issues.append("🟡 机器人30%时间速度为0, 可能有间歇性卡顿")

            avg_vx = sum(vxs) / len(vxs)
            if avg_vx < 0.03:
                issues.append("🔴 平均速度极低 (<0.03m/s), 可能在原地反复微调")
            elif avg_vx < 0.08:
                issues.append("🟡 平均速度偏低 (<0.08m/s), 检查是否被safety_monitor或分级策略持续限速")

        if self.scan_history:
            fronts = [s['front'] for s in self.scan_history]
            near_pct = sum(1 for f in fronts if f < 0.5) / len(fronts)
            if near_pct > 0.5:
                issues.append("🔴 前方<0.5m占比>50%, 锥桶密度很高, 机器人几乎一直在避障")
            elif near_pct > 0.2:
                issues.append("🟡 前方<0.5m占比>20%, 有一定锥桶密度")

            min_f = min(fronts)
            if min_f < 0.15:
                issues.append("🔴 机器人曾到达离障碍物<0.15m的极限距离, safety_monitor应该触发了")

        if self.path_history and len(self.path_history) > 3:
            lengths = [p['len'] for p in self.path_history[-5:]]
            if max(lengths) < 0.5:
                issues.append("🔴 全局路径极短 (<0.5m), GlobalPlanner可能找不到穿越锥桶的路径")

        if not issues:
            issues.append("✅ 未发现明显问题")

        for i in issues:
            print(i)

        print(f"\n完整日志: {self.log_path}")
        print("=" * 60)

    def run(self):
        rospy.on_shutdown(self.summary)
        rospy.spin()


if __name__ == '__main__':
    try:
        m = NavMonitor()
        m.run()
    except rospy.ROSInterruptException:
        pass
