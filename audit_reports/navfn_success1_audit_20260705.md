# cruise_navfn_success1 流程审核报告

检查对象:
- roslaunch iden_controller cruise_navfn_success1.launch
- rosrun iden_controller process_navfn_success1.py

检查日期: 2026-07-05
原则: 本次只读取旧文件；本文件为新增审核报告，未修改任何已有 launch、script、config 或源码。

## 当前流程

launch 解析后会启动 /my_base_driver、/ydlidar_node、/base_link_to_laser、/map_server、/amcl、/safety_monitor、/move_base。
process_navfn_success1.py 按固定序列发送 s0 到 s15 的目标点；每点 hard_timeout=12s，失败后恢复并重试 1 次，再失败会停止整条巡航但节点保持 spin。

## 已验证事实

- roslaunch --nodes 解析成功。
- process_navfn_success1.py 与 safety_monitor_success1.py 通过 python3 -m py_compile。
- 当前 ROS master 被动检查仅看到 /rosout 和一个 RViz，未发现导航流程正在运行。
- 地图 6.3.1.yaml: 640x640, resolution=0.03, origin=[-9.99,-9.99,0]。
- success1 目标点都在地图范围内且目标点本身为 FREE。

## P0 高风险

1. 地图与目标点硬编码。launch 固定加载 /home/ucar/instant_ws/src/ucar_nav/maps/6.3.1.yaml，process 内固定写死目标点。换相似地图后，origin、resolution、墙体边界或起点略变都可能导致贴墙、不可达或假成功。

2. 安全余量偏小。footprint 约 0.342m x 0.256m，但 inflation_radius=0.10，stop_zone=0.11，side_stop_zone=0.055。地图投影估算 s3/s3t 中心离障碍约 0.150m，s1t->s2 直连段约 0.120m，s13->s15 直连段约 0.060m。对半车宽 0.128m 来说，部分点几乎贴墙。

3. safety_monitor_success1 左右扇区疑似反向。success1 中 left 使用负角、right 使用正角；按 ROS 常规激光坐标正角为左。success2 已将左右写法对调，并增加 max_stop_angular。若实机因 ydlidar reversion 导致角度语义不同，必须用左右墙实测确认。

4. IdenPlanner V1 碰撞检查只看路径中心点，不检查完整 footprint。iden_planner.cpp 的碰撞预检只读取 global_plan 点所在 costmap cell，不能保证车体四角和边线不碰墙。当前包内 IdenPlannerV2 已有 footprint 检查，但 success1 launch 没用 V2。

5. 存在假到达链路。IdenPlanner 位姿调整超过 pose_adjust_timeout 会强制 goal_reached=true；process 的 goal_is_verified 在没有 AMCL pose 时返回 True。AMCL 丢失或定位不稳时可能提前放行。

6. 任一点失败会提前停止巡航。每点只重试一次，再失败就 break，无法保证跑到最终目标。它能避免无限卡死，但不满足终点优先。

## P1 中高风险

7. process 等 move_base 只有 5 秒。roslaunch 后立刻 rosrun 时，map/amcl/move_base 插件加载稍慢就可能直接退出。

8. 关键节点未 respawn。move_base、ydlidar、safety_monitor、base driver 掉线后没有自动恢复；安全监控掉线时 /cmd_vel_raw 不会继续转发到 /cmd_vel，任务会停住或失败。

9. GlobalPlanner 参数混用。当前 base_global_planner 是 navfn/NavfnROS，但仍加载 /move_base/GlobalPlanner 参数，这组参数对 Navfn 基本无效；真正 Navfn 只显式得到 allow_unknown 与 default_tolerance。

10. s13 后 force_move_x 0.60m 绕过 move_base。按当前 s13 yaw=pi 估算，这段在 6.3.1 地图上最小静态 clearance 约 0.24m，尚可；但它对定位、yaw 和换图非常敏感，且没有地图 footprint 分段检查。

## 建议的新文件方向

1. 新增 launch/cruise_navfn_success1_guarded.launch: map_yaml 作为 arg；使用更保守 safety 参数；明确当前 planner 真正读取的参数；按需要给关键节点 respawn 或 watchdog。

2. 新增 scripts/preflight_navfn_success1_map.py: 启动前读取地图 yaml/pgm 和目标点，检查目标是否在图内、是否 FREE、clearance 是否大于阈值；失败则不发任何 goal。

3. 新增 scripts/process_navfn_success1_guarded.py: 等待 /scan、/map、/tf、/amcl_pose、/move_base/status；AMCL 未就绪绝不判定成功；使用进展式 timeout 和分级恢复，不因一次失败就停止全任务。

4. 新增 scripts/safety_monitor_success1_guarded.py: 基于 success2 的左右修正，增加角速度上限、scan 超时零速保持和左右方向诊断日志。

## 推荐优先级

P0: 修正/验证左右方向；禁止 no amcl pose 判成功；增加地图目标预检；增大安全余量或降低窄段速度。
P1: 使用完整 footprint 检查或切到 IdenPlannerV2；把地图和目标点参数化；加入分级恢复与全局任务预算。
P2: 增加事件日志、心跳监控，清理 GlobalPlanner/Navfn 参数混用。
