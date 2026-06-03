# UCAR 工程代码优化报告

> 基于对 iflytek_car.odt（第二十一届全国大学生智能车竞赛讯飞组"智慧工厂"赛项）和全部工程代码的深度分析

## 一、总体评估

当前工程代码覆盖了竞赛要求的**大部分基础能力**（巡线、导航、语音、二维码检测），但存在**三个核心问题**：

1. **系统碎片化** — 各子系统独立开发，缺少端到端任务编排
2. **关键技术栈不完整** — QR→大模型→语音→导航的链路未打通
3. **两套并行的巡线方案** — flow_end (C++) 和 ucar_followline.py (Python) 功能重叠、各自为政

---

## 二、按优先级分级的优化建议

### 🔴 P0 — 必须优先解决（阻塞比赛完成）

#### 2.1 搭建端到端任务状态机（对应全部 5 个子任务）

**现状：** 当前没有任何一个节点串联起 5 个子任务。`process_5.5.1.py` 和 `process_5.30.1.py` 是简单的 waypoint 巡逻脚本，不涉及语音交互、QR 识别、大模型推理、仿真协同。

**建议：** 新建 `src/task_orchestrator/` 包，用 Python 实现有限状态机：

```
IDLE → WAKEUP_RECEIVED
  → NAV_TO_QR_AREA   → QR_SCANNING（子任务1）
  → CALL_SPARK_X2    → BROADCAST_RESULT
  → NAV_TO_PRODUCTION_AREA → DYNAMIC_AVOID（子任务2）
  → WAREHOUSE_PARKING → BROADCAST_PLACEMENT
  → SIM_TRIGGER → WAIT_SIM → BROADCAST_SIM（子任务3）
  → NAV_TO_TRAFFIC → TRAFFIC_RECOGNITION（子任务4）
  → PATH_DECISION → BROADCAST_DECISION
  → LANE_FOLLOW → FORK_DECISION（子任务5）
  → FINISH_PARKING → TASK_COMPLETE
```

代码位置：`/home/ucar/instant_ws/src/task_orchestrator/scripts/task_state_machine.py`

---

#### 2.2 打通 QR → 大模型 → 语音播报链路（子任务 1）

**现状：**
- `test/src/qr_node.cpp` 使用 OpenCV + ZBar 做本地 QR 解码 — **但赛题要求 QR 码返回的是 URL 链接**，不是直接内容，需要访问网页获取 JSON
- `speech_command/src/spark_llm_node.py` 已实现 Spark X2 HTTP 调用，但订阅的 `/factory/llm_trigger` 话题**没有任何节点发布**
- 语音播报的 TTS 功能在 `aiuiMain.cpp` 中通过 `/factory/tts_text` 接收文本，链路理论可用

**需要实现：**

```python
# 子任务1核心逻辑伪代码
def subtask_1():
    # 1. 语音唤醒（已有）
    wakeup_result = wait_for_wakeup()  # 等待 speech_command_node 的 /question topic

    # 2. 导航到二维码区（已有）
    navigate_to(qr_area_point)

    # 3. 依次识别3个二维码 → 需要新增: 从QR URL获取JSON
    items = []
    for i in range(3):
        qr_url = detect_qr_code()  # qr_node 改造后直接输出URL
        json_data = requests.get(qr_url, timeout=5).json()  # 获取JSON
        items.append(json_data["result"])  # 提取货品名称

    # 4. 调用星火大模型（已有spark_llm_node.py，需要触发）
    trigger_msg = {
        "real_category": wakeup_result.real_category,
        "sim_category": wakeup_result.sim_category,
        "items": items
    }
    pub_to_llm_trigger.publish(json.dumps(trigger_msg))

    # 5. 等待LLM结果并播报（已有spark_llm_node.py的回调链）
    # spark_llm_node.py → /factory/tts_text → aiuiMain.cpp TTS → 语音输出
```

**具体改动：**

| 文件 | 改动 |
|------|------|
| `test/src/qr_node.cpp` | 改为返回解码后的 URL 字符串而非直接内容 |
| `test/src/qr_url_parser.py` | 新增：接收 URL，发起 HTTP 请求，解析 JSON |
| `speech_command/src/spark_llm_node.py` | 订阅方改为状态机节点触发（替换 `/factory/llm_trigger`） |
| **新建** `task_orchestrator/scripts/subtask1_handler.py` | 协调 QR→HTTP→LLM→TTS 全流程 |

---

#### 2.3 统一巡线方案（子任务 4/5）

**现状：** 存在两套并行的视觉巡线系统：

| | flow_end (C++) | ucar_followline.py |
|---|---|---|
| 代码量 | ~4000 行 C++ | 2753 行 Python |
| 状态机 | IDLE→ALIGN→FOLLOW→PARK | NORMAL→PARKING→COMPLETED |
| 路径选择 | Left/Middle/Right/Y-branch | 无（仅巡线） |
| 停车检测 | 双 L 角点确认 | 单圆弧角点检测 |
| 交通灯决策 | 无（依赖外部触发） | HSV 色域检测 |
| 维护状态 | 活跃开发中 | 最近更新 |

**建议：** **以 `flow_end` (C++) 为主方案**，理由：
- 性能更好（C++ vs Python）
- 功能更完整（路径选择、Y 岔路、双 L 角点停车）
- 已经是活跃开发分支

需要从 `ucar_followline.py` 中提取并增强到 `flow_end` 的功能：

| 功能 | 来源 | 目标 |
|------|------|------|
| 精确 stop-line 停车（车头≤10cm） | ucar_followline.py 的 `LaneDetectionNode` | `flow_end/src/follow_line_test.cpp` |
| 多进程仲裁架构（巡线/避障优先级） | ucar_followline.py 的 `ChassisController` | `flow_end/src/Callback_test.cpp` |
| 交通灯色域检测 HSV | ucar_nav/src/task_class.h 的 `greenCallback` | `flow_end/src/` 新增 `traffic_light_detector.cpp` |

---

### 🟡 P1 — 应尽快解决（影响比赛得分）

#### 2.4 实现动态避障模块（子任务 2）

**现状：**
- `ucar_nav/scripts/escape_obstacle.py` 有基础的旋转→平移→回避功能
- `my_planner/src/my_planner.cpp` 自定义局部规划器结合了代价地图碰撞检测
- 但没有**锥桶检测**专用模块

**建议：**

1. **激光雷达锥桶检测模块** — 新建 `perception/cone_detector.py`

```python
class ConeDetector:
    def __init__(self):
        self.cone_radius = 0.15  # 锥桶半径约15cm
        self.min_cluster_size = 3
        self.max_cluster_size = 15
        self.detection_range = 3.0  # 检测范围3米

    def detect(self, laser_scan):
        # 1. 欧几里得聚类分割
        # 2. 过滤尺寸: 3~15个点 = 锥桶大小
        # 3. 发布 /cones (PoseArray) 用于代价地图动态层
```

2. **AMCL 优化** — 改用 `likelihood_field` 激光模型（当前 beam 模型计算量大）：

在 `/home/ucar/instant_ws/src/ucar_nav/launch/config/amcl/amcl_omni.launch` 中：
```xml
<param name="laser_model_type" value="likelihood_field"/>
<param name="laser_likelihood_max_dist" value="2.0"/>
```

3. **局部规划器增强** — `my_planner` 当前只有比例控制+PID 转向，建议集成 TEB 的 `teb_local_planner_params.yaml` 配置，TEB 更适合动态避障场景。

---

#### 2.5 增强交通灯识别（子任务 4）

**现状：** `ucar_nav/src/task_class.h` 的 `greenCallback` 使用简单的 HSV 色域分割

**赛题要求：**
- 交通灯是 WS2812 LED 点阵，可以有直行/左转/右转（绿色）和红灯
- 选手自制，外观不统一
- 需要**语音控制**红绿灯状态（聆思 CSK5062）

**建议：**

1. **从 HSV 改为深度学习分类**（训练一个轻量的 4 类分类器：红灯/直行/左转/右转）

```
input: 抠出的红绿灯 ROI 区域 (64x64)
model: MobileNetV2 或更小的 3层CNN
output: [red, straight, left, right] 四分类置信度
dataset: 自己采集的 WS2812 红绿灯样本
```

2. **同步实现红绿灯控制器**（聆思 CSK5062）

这是赛题明确要求的自制硬件，需要：
- 参考文档 `https://docs2.listenai.com/x/AYLuBCyWV`
- 使用 WS2812/WS2812B LED 点阵
- 通过语音控制显示直行/左转/右转/红灯状态

---

#### 2.6 修复已知 Bug

| # | 文件 | 问题 | 影响 |
|---|------|------|------|
| 1 | `speech_command/launch/speech_command.launch:4` | 重复的 `</launch>` 标签，XML 解析失败 | 语音节点无法通过 launch 文件启动 |
| 2 | `speech_command/config/offline_QA.txt` | 4 条指令全部指向同一个音频文件 `offline_left.mp3` | 向前/向右/向后都播报"左走" |
| 3 | `speech_command/config/AIUI/cfg/aiui.cfg` | 所有路径硬编码为 `/home/ucar/ucar_ws/` | 在 `/home/ucar/instant_ws/` 下运行失败 |
| 4 | `speech_command/src/aiuiMain.cpp:172` | `system("aplay /home/iflytek/ucar_ws/...")` 硬编码路径 | 音频播放失败 |
| 5 | `speech_command/src/AudioPlayer.cpp` | `sleep(10000)` (10秒) 在写循环中 | 音频播放严重卡顿 |
| 6 | `ucar_nav/launch/config/amcl/amcl_omni.launch` | `transform_tolerance = 10` (10秒!)，应该是 0.1~0.5 | 掩盖 TF 丢帧问题 |
| 7 | `iden_controller` 和 `ucar_controller` | 两份几乎相同的底盘驱动代码 | 维护困难，容易不同步 |

---

### 🟢 P2 — 应逐步优化（提升稳定性和得分）

#### 2.7 仿真协同集成（子任务 3）

**现状：** `ucar_nav/src/nav_node.cpp` 有 TCP Socket `SimTriggerClient` (192.168.0.240:1145)，但代码不完整。

**建议：** 实现完整的 Gazebo 协同协议：

```python
class GazeboSimClient:
    def send_ready_signal(self, warehouse_name):
        """发送就绪信号，指定目标仓库"""
        msg = {"action": "ready", "warehouse": warehouse_name}
        self.socket.send(json.dumps(msg).encode())

    def wait_completion(self, timeout=60):
        """等待仿真完成反馈"""
        data = self.socket.recv(1024)
        result = json.loads(data)
        return result["status"] == "completed"
```

#### 2.8 参数集中管理和运行时调参

**现状：** 参数散落各处：
- `flow_end/launch/follow_test.launch` — ROS param 50+ 个
- `ucar_nav/launch/config/move_base/` — 7 个独立 YAML
- `iden_controller/config/driver_params_ucarV2.yaml` — 底盘参数
- `line_follower/config/` — 另一套 PID 参数

**建议：**
1. 用 `dynamic_reconfigure` 替代 launch 文件中的硬编码参数
2. 统一到 `config/competition_params.yaml`
3. 增加运行时参数调整的 RQT 面板

#### 2.9 日志和调试能力增强

**现状：** 各个节点只有 `ROS_INFO` 级别的日志，出问题时难以定位。

**建议：**
1. 在状态机节点中记录每次状态转换和时间戳
2. 增加 `/diagnostics` 系统级监控
3. 比赛结束后自动导出完整 log

---

## 三、架构优化建议

### 3.1 当前架构 vs 推荐架构

**当前（碎片化）：**
```
[process_5.5.1.py] → [move_base]
[speech_command]    → [AIUI SDK]（独立）
[flow_end]          → [巡线]（独立）
[ucar_followline.py] → [巡线2]（独立、冗余）
[qr_node]           → [QR解码]（独立）
```

**推荐（统一编排）：**
```
[task_state_machine.py] —— 中心状态机
  ├── 订阅 /question (语音识别结果)
  ├── 调用 qr_detection (QR→URL→JSON)
  ├── 触发 spark_llm_node (大模型推理)
  ├── 发送 waypoint 给 move_base (导航)
  ├── 发送 /follow_begin 给 flow_end (巡线)
  ├── 订阅 /follow_end (巡线完成)
  ├── 调用 simulation_client (仿真协同)
  ├── 发布 /factory/tts_text (语音播报)
  └── 记录状态日志 (debug)
```

### 3.2 代码清理

| 清理项 | 说明 |
|--------|------|
| 删除 `ucar_nav/scripts/ucar_followline.py` | 功能合并到 `flow_end` 后删除 |
| 删除 `line_follower/` | 已有更好的 `flow_end` |
| 合并 `iden_controller` 和 `ucar_controller` | 保留 PID 版本，删除无 PID 版本 |
| 清除所有 `/home/ucar/ucar_ws/` 硬编码路径 | 改为 ROS 包路径动态解析 |
| 删除 `flow_end/src/follow.cpp` / `Callback.cpp` / `follow_line.cpp` | 旧的 `follow_end`，被 `follow_test` 替代 |

---

## 四、子任务差距分析总结

| 子任务 | 现有能力 | 缺失环节 | 优先级 |
|--------|----------|----------|--------|
| 子任务1：智能接单与货品筛选 | 语音唤醒 ✓ / QR检测框架 ✓ / Spark X2 API ✓ | QR→URL→HTTP→JSON链路 ✗ / LLM触发源 ✗ / 端到端编排 ✗ | 🔴 P0 |
| 子任务2：仓库匹配与动态避障 | 导航栈 ✓ / 自定义规划器 ✓ | 锥桶检测 ✗ / 仓库识别导航 ✗ / 避障-巡线切换 ✗ | 🟡 P1 |
| 子任务3：仿真系统协同 | TCP Socket骨架 | Gazebo协议 ✗ / 协同状态管理 ✗ | 🟢 P2 |
| 子任务4：交通决策与路径选择 | 路径选择 ✓ / 交通灯检测框架 | 精准stop-line停车(≤10cm) ✗ / 分类器不鲁棒 ✗ | 🟡 P1 |
| 子任务5：视觉巡线及终点抵达 | 巡线 ✓ / 岔路决策 ✓ / 停车检测 ✓ | 终点播报触发 ✗ / finish区域判定 ✗ | 🟢 P2 |

---

## 五、建议开发顺序（共约 8-10 周）

| 周次 | 任务 | 产出 |
|------|------|------|
| 第1-2周 | P0: 搭建任务状态机骨架 + 修复 speech_command Bug + 统一巡线 | 可运行的空状态机 + fix 6个known bugs |
| 第3-4周 | P0: QR→HTTP→LLM→TTS 链路打通 | 子任务1 端到端可演示 |
| 第5-6周 | P1: 锥桶检测 + 动态避障 + 交通灯DL分类器 | 子任务2/4 可演示 |
| 第7-8周 | P1: 红绿灯硬件(CSK5062) + stop-line精准停车 | 子任务4/5 可演示 |
| 第9-10周 | P2: Gazebo仿真协同 + 全流程联调 | 完整比赛流程可演示 |

---

## 六、关键文件清单（建议重点修改的文件）

### 需要修改的文件（按优先级）
1. `src/speech_command/launch/speech_command.launch` — 修复 XML Bug
2. `src/speech_command/config/AIUI/cfg/aiui.cfg` — 修复路径
3. `src/speech_command/src/aiuiMain.cpp` — 修复硬编码路径
4. `src/flow_end/src/follow_line_test.cpp` — 增加 stop-line 停车 + 交通灯决策
5. `src/ucar_nav/launch/config/amcl/amcl_omni.launch` — AMCL 参数优化
6. `src/my_planner/src/my_planner.cpp` — 增强避障能力

### 需要新建的文件
1. `src/task_orchestrator/` — 全新任务编排包
2. `src/perception/cone_detector.py` — 锥桶检测
3. `src/perception/traffic_light_classifier.py` — 交通灯分类
4. `src/simulation/gazebo_client.py` — 仿真协同客户端
