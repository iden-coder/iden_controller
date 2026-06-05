#ifndef IDEN_PLANNER_H_
#define IDEN_PLANNER_H_

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <tf/transform_listener.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>

// gtest 友元声明 — 允许测试访问 private 方法
#include <gtest/gtest_prod.h>

namespace iden_planner
{

// 前向声明测试夹具（定义在 test/test_iden_planner.cpp 中）
class IdenPlannerTest;

class IdenPlanner : public nav_core::BaseLocalPlanner
{
    // 允许测试夹具访问所有私有成员（SetUp + 所有 TEST_F）
    friend class IdenPlannerTest;

    // gtest 友元 — 允许单元测试访问私有方法
    FRIEND_TEST(IdenPlannerTest, AngleSpeedScale_Boundary);
    FRIEND_TEST(IdenPlannerTest, AngleSpeedScale_MidRange);
    FRIEND_TEST(IdenPlannerTest, AngularPID_IntegralAccumulation);
    FRIEND_TEST(IdenPlannerTest, AngularPID_AntiWindup);
    FRIEND_TEST(IdenPlannerTest, AngularPID_ResetOnWaypointSwitch);
    FRIEND_TEST(IdenPlannerTest, ReverseMode_StateMachine);
    FRIEND_TEST(IdenPlannerTest, ReverseMode_ReconfirmOnDirectionChange);
    FRIEND_TEST(IdenPlannerTest, ReverseMode_InterruptResetsCounter);

public:
    IdenPlanner();
    ~IdenPlanner();

    void initialize(std::string name, tf2_ros::Buffer* tf,
                    costmap_2d::Costmap2DROS* costmap_ros) override;
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override;
    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;
    bool isGoalReached() override;

private:
    // === 路径追踪：选点 + PID ===
    geometry_msgs::PoseStamped selectLookaheadTarget();
    double  computeAngleSpeedScale(double angle_error_rad);
    double  computeAngularPID(double lateral_error);

    // === 位姿最终调整 ===
    bool    computePoseAdjust(geometry_msgs::Twist& cmd_vel,
                              const geometry_msgs::PoseStamped& pose_final);

    // === 双向导航（前进/倒车自动切换）===
    bool    analyzePathDirection();          // 返回 true=应倒车
    void    updateReverseState(bool want_reverse);

    // === 代价地图可视化 ===
    void    renderCostmap();
    void    publishCostmap();          // 通过 cv_bridge 发布到 ROS topic

    // === 参数热重载（每秒一次）===
    void    reloadParams();

    // --- 核心成员 ---
    tf::TransformListener*           tf_listener_;
    costmap_2d::Costmap2DROS*        costmap_ros_;
    std::vector<geometry_msgs::PoseStamped> global_plan_;
    ros::Publisher                   costmap_pub_;     // 代价地图图像发布

    // 状态
    int    target_index_;
    int    prev_target_index_;        // 检测航点切换，重置积分
    bool   pose_adjusting_;
    bool   goal_reached_;
    bool   reverse_mode_;            // false=前进, true=倒车
    int    reverse_confirm_cnt_;      // 方向切换防抖计数

    // PID 内部状态（需跨帧保持）
    double error_sum_;
    double last_error_;

    // 参数重载计数器
    int    param_reload_cnt_;
    std::string param_ns_;           // 存储参数命名空间用于热重载

    // === 全部可调参数（ROS param，运行时热重载）===

    // -- PID --
    double kp_;                       // 角速度 P 增益 (default 3.0)
    double ki_;                       // 角速度 I 增益 (default 0.02)
    double kd_;                       // 角速度 D 增益 (default 0.02)
    double integral_limit_;           // 积分抗饱和上限 (default 0.5)

    // -- 预瞄 --
    double lookahead_dist_;           // 路径点预瞄距离 m (default 0.25)

    // -- 角度-速度耦合 --
    double angle_power_;              // 降速曲线陡峭度 N (default 1.5)
    double angle_full_speed_deg_;     // 满速角度阈值 °(default 0)
    double angle_zero_speed_deg_;     // 零速角度阈值 °(default 90)

    // -- 双向导航 --
    bool   enable_bidirectional_;     // 是否启用自动倒车 (default true)
    double long_lookahead_;           // 长预瞄距离 m (default 0.5)
    double reverse_vote_ratio_;       // 后方路径点占比阈值 (default 0.7)
    int    reverse_confirm_frames_;   // 方向切换确认帧数 (default 3)

    // -- 速度限制 --
    double max_linear_vel_;           // 最大线速度 m/s (default 0.5)
    double min_linear_vel_;           // 最小线速度（倒车时 -max）(default -0.5)
    double max_angular_vel_;          // 最大角速度 rad/s (default 1.0)
    double linear_gain_;              // 线速度 = dx * gain (default 3.0)

    // -- 可视化 --
    bool   enable_visualization_;      // 是否弹出 OpenCV 调试窗口 (default false)
    bool   enable_costmap_pub_;        // 是否通过 topic 发布代价地图 (default false)

    // -- 碰撞检测 --
    int    collision_check_count_;     // 预瞄路径碰撞检测点数量 (default 10)
    int    collision_cooldown_;        // 碰撞冷却帧数，防止频繁触发重规划
    int    collision_cooldown_max_;    // 碰撞冷却最大帧数 (default 20 ≈ 2s@10Hz)
    int    collision_replan_count_;    // 连续重规划次数 (≥2 次后不再触发重规划，仅减速)
    int    collision_replan_max_;      // 最大重规划次数 (default 2)

    // -- 位姿调整 --
    double pose_dist_threshold_;      // 进入位姿调整的距离阈值 m (default 0.1)
    double pos_deadband_;             // 位置死区 m (default 0.015)
    double yaw_deadband_;             // 角度死区 rad (default 0.04)
    double angle_adjust_gain_;        // 角度→角速度增益 (default 1.0)
    double slow_zone_;                // 降速区 rad (default 0.2)
    double min_adjust_speed_;         // 最低调整角速度 rad/s (default 0.08)
    double pose_tolerance_;           // 最终到位容差 rad (default 0.015)
    double pose_adjust_timeout_;      // 位姿调整超时秒数 (default 15.0)
    double pose_adjust_max_linear_;   // 位姿调整最大线速度 m/s (default 0.10)
    ros::Time pose_adjust_start_;     // 进入位姿调整的时刻
};

}  // namespace iden_planner

#endif  // IDEN_PLANNER_H_
