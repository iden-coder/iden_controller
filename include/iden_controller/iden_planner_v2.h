#ifndef IDEN_PLANNER_V2_H_
#define IDEN_PLANNER_V2_H_

#include <ros/ros.h>
#include <nav_core/base_local_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/footprint.h>
#include <tf/transform_listener.h>
#include <opencv2/core/core.hpp>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <visualization_msgs/Marker.h>

namespace iden_planner_v2
{

/**
 * @brief 轨迹候选: 一对速度 + 评分的组合
 */
struct TrajectoryCandidate
{
    double vx;           // 线速度 (m/s)
    double wz;           // 角速度 (rad/s)
    double score;        // 总分 (越小越好)
    double obstacle_cost;   // 障碍物代价分量
    double path_cost;       // 路径偏差代价分量
    double goal_cost;       // 目标进度代价分量
    bool   is_valid;        // 是否合法 (无碰撞)

    TrajectoryCandidate() : vx(0), wz(0), score(1e9),
        obstacle_cost(0), path_cost(0), goal_cost(0), is_valid(true) {}
};

/**
 * @brief IdenPlannerV2 — 增强版局部规划器
 *
 * 在 IdenPlanner 基础上新增:
 *   1. 前向时序碰撞检测 + 足印检查
 *   2. 轨迹采样 + 多维度评分 (采样-评分范式)
 *   3. 速度平滑 (加速度/减速度约束)
 *   4. 智能恢复 (原地旋转换路)
 *
 * 用法: 在 move_base launch 中设置
 *   <param name="base_local_planner" value="iden_planner_v2/IdenPlannerV2"/>
 */
class IdenPlannerV2 : public nav_core::BaseLocalPlanner
{
public:
    IdenPlannerV2();
    ~IdenPlannerV2();

    void initialize(std::string name, tf2_ros::Buffer* tf,
                    costmap_2d::Costmap2DROS* costmap_ros) override;
    bool setPlan(const std::vector<geometry_msgs::PoseStamped>& plan) override;
    bool computeVelocityCommands(geometry_msgs::Twist& cmd_vel) override;
    bool isGoalReached() override;

private:
    // =========== 阶段入口 ===========
    bool computePoseAdjustPhase(geometry_msgs::Twist& cmd_vel);
    bool computeTrackingPhase(geometry_msgs::Twist& cmd_vel);
    bool computeRecoveryPhase(geometry_msgs::Twist& cmd_vel);

    // =========== 路径追踪: 选点 + PID (继承自 IdenPlanner) ===========
    geometry_msgs::PoseStamped selectLookaheadTarget();
    double  computeAngleSpeedScale(double angle_error_rad);
    double  computeAngularPID(double lateral_error);

    // =========== 位姿最终调整 ===========
    bool    computePoseAdjust(geometry_msgs::Twist& cmd_vel,
                              const geometry_msgs::PoseStamped& pose_final);

    // =========== 双向导航 ===========
    bool    analyzePathDirection();
    void    updateReverseState(bool want_reverse);

    // =========== 【新增】前向时序碰撞检测 ===========
    /**
     * @brief 前向仿真当前速度指令,检查是否会碰撞
     * @param vx       线速度
     * @param wz       角速度
     * @param sim_time 仿真时长 (s)
     * @param[out] collision_dist 碰撞距离 (若发生碰撞)
     * @return true 表示会发生碰撞
     */
    bool forwardSimulateCollision(double vx, double wz, double sim_time,
                                  double* collision_dist = nullptr);

    /**
     * @brief 在(x,y,theta)处用足印检查是否碰撞
     */
    bool isFootprintInCollision(double x, double y, double theta);

    // =========== 【新增】轨迹采样 + 评分 ===========
    /**
     * @brief 在当前速度附近采样候选轨迹
     */
    std::vector<TrajectoryCandidate> generateTrajectoryCandidates(
        double current_vx, double current_wz);

    /**
     * @brief 对单个候选轨迹评分
     */
    void scoreTrajectory(TrajectoryCandidate& candidate,
                         double current_vx, double current_wz);

    /**
     * @brief 在候选集中选最优
     */
    TrajectoryCandidate selectBestCandidate(
        const std::vector<TrajectoryCandidate>& candidates);

    // =========== 【新增】速度平滑 ===========
    geometry_msgs::Twist applyVelocitySmoothing(
        const geometry_msgs::Twist& desired,
        const geometry_msgs::Twist& current);

    // =========== 【新增】智能恢复 ===========
    /**
     * @brief 寻找最安全的旋转方向 (带窄道安全检查)
     * @return 推荐角速度 (正=左转, 负=右转), 0=无法安全旋转
     */
    double findSafeRotationDirection();

    // =========== 【新增】侧向间隙监测 ===========
    /**
     * @brief 检查当前位置左右两侧的可用间隙
     * @param x, y, theta  机器人位姿 (全局坐标系)
     * @param[out] left_clearance   左侧可用间隙 (m)
     * @param[out] right_clearance  右侧可用间隙 (m)
     * @return true 表示两侧都有足够的间隙
     */
    bool checkSideClearance(double x, double y, double theta,
                            double* left_clearance = nullptr,
                            double* right_clearance = nullptr);

    // =========== 代价地图可视化 ===========
    void renderCostmap();
    void publishCostmap();

    // =========== 参数热重载 ===========
    void reloadParams();

    // =========== 调试可视化 ===========
    void publishTrajectoryCandidates(
        const std::vector<TrajectoryCandidate>& candidates,
        const TrajectoryCandidate& best);

    // --- 核心成员 ---
    tf::TransformListener*           tf_listener_;
    costmap_2d::Costmap2DROS*        costmap_ros_;
    std::vector<geometry_msgs::PoseStamped> global_plan_;
    ros::Publisher                   costmap_pub_;
    ros::Publisher                   traj_viz_pub_;       // 轨迹候选可视化

    // 状态
    int    target_index_;
    int    prev_target_index_;
    bool   pose_adjusting_;
    bool   goal_reached_;
    bool   reverse_mode_;
    int    reverse_confirm_cnt_;

    // PID 内部状态
    double error_sum_;
    double last_error_;

    // 速度平滑状态
    geometry_msgs::Twist last_cmd_vel_;   // 上一帧实际发出的速度指令

    // 恢复状态
    bool   in_recovery_;                  // 是否处于恢复模式
    bool   recovery_backup_phase_;        // 恢复模式第一阶段: 是否正在后退
    ros::Time recovery_backup_start_;     // 后退开始时间
    double recovery_backup_distance_;     // 恢复时后退距离, 默认0.10m
    double recovery_backup_speed_;        // 恢复时后退速度, 默认0.08m/s
    double recovery_yaw_target_;          // 恢复旋转的目标角速度(正=左,负=右)
    double recovery_yaw_accum_;           // 已旋转的累计角度
    double recovery_prev_yaw_;            // 上一帧yaw (用于累计旋转角度)
    bool   recovery_yaw_initialized_;

    // 参数重载计数器
    int    param_reload_cnt_;
    std::string param_ns_;

    // =========== 全部可调参数 ===========

    // -- PID --
    double kp_;
    double ki_;
    double kd_;
    double integral_limit_;

    // -- 预瞄 --
    double lookahead_dist_;

    // -- 角度-速度耦合 --
    double angle_power_;
    double angle_full_speed_deg_;
    double angle_zero_speed_deg_;

    // -- 双向导航 --
    bool   enable_bidirectional_;
    double long_lookahead_;
    double reverse_vote_ratio_;
    int    reverse_confirm_frames_;

    // -- 速度限制 --
    double max_linear_vel_;
    double min_linear_vel_;
    double max_angular_vel_;
    double linear_gain_;

    // -- 可视化 --
    bool   enable_visualization_;
    bool   enable_costmap_pub_;

    // -- 碰撞检测 (原有,保留兼容) --
    int    collision_check_count_;
    int    collision_cooldown_;
    int    collision_cooldown_max_;
    int    collision_replan_count_;
    int    collision_replan_max_;

    // -- 位姿调整 --
    double pose_dist_threshold_;
    double pos_deadband_;
    double yaw_deadband_;
    double angle_adjust_gain_;
    double slow_zone_;
    double min_adjust_speed_;
    double pose_tolerance_;
    double pose_adjust_timeout_;
    double pose_adjust_max_linear_;
    ros::Time pose_adjust_start_;

    // ====== 【新增参数】 ======

    // -- 前向碰撞检测 --
    bool   enable_forward_sim_;         // 是否启用前向仿真碰撞检测 (default true)
    double forward_sim_time_;            // 前向仿真时长 s (default 1.0)
    double forward_sim_step_;            // 仿真步长 s (default 0.05)

    // -- 轨迹采样 --
    bool   enable_traj_sampling_;        // 是否启用轨迹采样评分 (default true)
    int    traj_samples_vx_;             // 线速度采样数 (default 7, 奇数)
    int    traj_samples_wz_;             // 角速度采样数 (default 11, 奇数)
    double traj_delta_vx_;               // 线速度采样步长 m/s (default 0.1)
    double traj_delta_wz_;               // 角速度采样步长 rad/s (default 0.3)

    // -- 代价权重 --
    double weight_obstacle_;             // 障碍物代价权重 (default 10.0)
    double weight_path_;                 // 路径偏差权重 (default 1.0)
    double weight_goal_;                 // 目标进度权重 (default 0.5)

    // -- 速度平滑 --
    bool   enable_vel_smooth_;           // 是否启用速度平滑 (default true)
    double max_linear_accel_;            // 最大线加速度 m/s² (default 2.0)
    double max_linear_decel_;            // 最大线减速度 m/s² (default 2.5)
    double max_angular_accel_;           // 最大角加速度 rad/s² (default 3.2)
    double max_angular_decel_;           // 最大角减速度 rad/s² (default 3.2)

    // -- 智能恢复 --
    bool   enable_smart_recovery_;       // 是否启用智能恢复 (default true)
    double recovery_rotation_speed_;      // 恢复旋转速度 rad/s (窄道 default 0.3)
    double recovery_rotation_angle_;      // 每次恢复旋转角度 rad (窄道 default 0.175≈10°)

    // -- 窄道模式 --
    bool   narrow_track_mode_;
    double track_width_;
    double min_side_clearance_;
    double lateral_check_distance_;

    // -- 分级速度策略 (锥桶避障核心) --
    bool   enable_graded_speed_;          // 启用分级速度响应 (default true)
    double danger_dist_far_;              // "安全"距离阈值 m (default 0.80, 此距离外正常速度)
    double danger_dist_mid_;              // "谨慎"距离阈值 m (default 0.40, 此距离内大幅减速)
    double danger_dist_near_;             // "危险"距离阈值 m (default 0.18, 此距离内停车)
    double speed_ratio_mid_;              // "谨慎"区速度比例 (default 0.50)
    double speed_ratio_near_;             // "危险"区速度比例 (default 0.20)
};

}  // namespace iden_planner_v2

#endif  // IDEN_PLANNER_V2_H_