# IdenPlannerV2 集成说明 — 窄道版

## 赛道约束

```
赛道宽度:  0.50m
机器人:    0.342m × 0.256m (footprint: ±0.171m × ±0.128m)
对角半宽:  0.214m
单侧间隙:  0.122m (居中时)

⚠️ 原地旋转 15° → 对角点位移 ~5.5cm, 超过半侧间隙的一半
⚠️ 原地旋转 22° → 对角点位移 ~8cm, 接近极限
⚠️ 后退 12cm → 碰到后方障碍(假定后方紧贴中线的背面)
```

## 新建文件列表

```
iden_controller/
├── include/iden_controller/
│   └── iden_planner_v2.h           # V2 规划器头文件 (270行)
├── src/
│   └── iden_planner_v2.cpp         # V2 规划器实现 (1430行)
├── scripts/
│   ├── safety_monitor.py            # 窄道安全监测节点 (230行)
│   └── cruise_with_recovery.py      # 窄道安全巡航脚本 (360行)
├── launch/
│   └── cruise_navfn_v2.launch       # V2 系统 launch 文件 (170行)
└── iden_planner_v2_plugin.xml       # V2 规划器插件注册
```

## 手动添加到现有文件 (纯增量)

### 1. CMakeLists.txt 末尾添加

```cmake
## ===== IdenPlannerV2: 增强版局部规划器 =====
add_library(iden_planner_v2
  src/iden_planner_v2.cpp
)
add_dependencies(iden_planner_v2 ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(iden_planner_v2
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

install(FILES
  iden_planner_v2_plugin.xml
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(TARGETS iden_planner_v2
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)
```

### 2. package.xml `<export>` 块内添加一行

```xml
<export>
  <nav_core plugin="${prefix}/iden_planner_plugin.xml" />
  <nav_core plugin="${prefix}/iden_planner_v2_plugin.xml" />   <!-- 新增 -->
</export>
```

## 编译运行

```bash
cd /home/iden/car_ws
catkin_make --pkg iden_controller
source devel/setup.bash

# 启动
roslaunch iden_controller cruise_navfn_v2.launch
rosrun iden_controller cruise_with_recovery.py
```

## 窄道关键参数速查

### IdenPlannerV2 窄道约束

| 参数 | 值 | 原因 |
|------|-----|------|
| `traj_samples_wz` | 7 | 角速度候选数减少, 避免探索危险的大转角 |
| `traj_delta_wz` | 0.15 | 角速度步长极小 (总范围 ±0.45 rad/s) |
| `traj_delta_vx` | 0.075 | 线速度步长极小 |
| `weight_path` | 3.0 | 偏离路径代价提高, 保持中线 |
| `max_linear_vel` | 0.6 | 窄道不允许高速 |
| `max_linear_accel` | 1.5 | 低速加速, 防打滑 |
| `recovery_rotation_angle` | 0.175 (~10°) | 微转, 角点位移仅 ~2cm |
| `recovery_rotation_speed` | 0.3 | 慢转, 防过冲 |
| `narrow_track_mode` | true | 启用侧向间隙检查 |
| `min_side_clearance` | 0.06 (6cm) | 最小侧向间隙阈值 |
| `track_width` | 0.50 | 赛道宽度 |
| `lateral_check_distance` | 0.26 | 侧向扫描距离 = 机器人宽度 |
| `forward_sim_time` | 0.8 | 前向仿真时间缩短(窄道不需要看太远) |

### SafetyMonitor 窄道约束

| 参数 | 值 | 原因 |
|------|-----|------|
| `stop_zone` | 0.12m | 极紧—侧隙仅12cm |
| `side_stop_zone` | 0.08m | 转弯时侧面监测 |
| `slowdown_zone` | 0.30m | 窄道减速距离较短 |
| `slowdown_ratio` | 0.50 | 50%减速(避免频繁启停) |
| `monitor_front_angle_deg` | 30° | 只关注正前方 |
| `monitor_side_angle_*` | 45°~70° | 侧面单独监测 |
| `min_angular_ratio` | 0.15 | 停车时仅保留15%旋转 |

### CruiseWithRecovery 窄道约束

| 参数 | 值 | 原因 |
|------|-----|------|
| `recovery_wiggle_angle` | 0.175 (~10°) | 微转, 角点位移 ~2cm |
| `recovery_nudge_dist` | 0.05m | 微移5cm, 远小于侧隙 |
| `recovery_max_rounds` | 2 | 最多2轮(避免无限恢复) |

## 与 IdenPlanner 原版的切换

```bash
# V2 增强版 (推荐用于窄道)
roslaunch iden_controller cruise_navfn_v2.launch

# 原版 (兼容回退)
roslaunch iden_controller cruise_navfn.launch
```

两套系统完全独立, 可以随时切换。

## 运行时热调参数

```bash
# 如果感觉太保守, 放宽约束
rosparam set /move_base/iden_planner_v2/IdenPlannerV2/min_side_clearance 0.04
rosparam set /safety_monitor/stop_zone 0.10

# 如果感觉太激进, 收紧约束
rosparam set /move_base/iden_planner_v2/IdenPlannerV2/max_linear_vel 0.4
rosparam set /safety_monitor/stop_zone 0.15

# 关闭轨迹采样 (回退到纯 PID)
rosparam set /move_base/iden_planner_v2/IdenPlannerV2/enable_traj_sampling false

# 关闭窄道模式 (测试用)
rosparam set /move_base/iden_planner_v2/IdenPlannerV2/narrow_track_mode false

# 动态开关安全监测
rosservice call /safety_monitor/toggle "data: false"
rosservice call /safety_monitor/toggle "data: true"
```
