#include <iden_controller/iden_planner_v2.h>
#include <pluginlib/class_list_macros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <cmath>
#include <algorithm>

PLUGINLIB_EXPORT_CLASS(iden_planner_v2::IdenPlannerV2, nav_core::BaseLocalPlanner)

namespace iden_planner_v2
{

// ============================================================
//  构造 / 析构
// ============================================================

IdenPlannerV2::IdenPlannerV2()
{
    setlocale(LC_ALL, "");
}

IdenPlannerV2::~IdenPlannerV2()
{
    if (tf_listener_)
        delete tf_listener_;
}

// ============================================================
//  initialize — 加载全部参数
// ============================================================

void IdenPlannerV2::initialize(std::string name, tf2_ros::Buffer* tf,
                               costmap_2d::Costmap2DROS* costmap_ros)
{
    tf_listener_ = new tf::TransformListener();
    costmap_ros_ = costmap_ros;

    param_ns_ = "~/" + name;
    ros::NodeHandle nh(param_ns_);

    // -- PID --
    nh.param("kp",                  kp_,                  3.0);
    nh.param("ki",                  ki_,                  0.02);
    nh.param("kd",                  kd_,                  0.02);
    nh.param("integral_limit",      integral_limit_,      0.5);

    // -- 预瞄 --
    nh.param("lookahead_dist",      lookahead_dist_,      0.25);

    // -- 角度-速度耦合 --
    nh.param("angle_power",         angle_power_,         1.5);
    nh.param("angle_full_speed_deg", angle_full_speed_deg_, 0.0);
    nh.param("angle_zero_speed_deg", angle_zero_speed_deg_, 90.0);

    // -- 双向导航 --
    nh.param("enable_bidirectional", enable_bidirectional_, false);
    nh.param("long_lookahead",      long_lookahead_,      0.5);
    nh.param("reverse_vote_ratio",  reverse_vote_ratio_,  0.7);
    nh.param("reverse_confirm_frames", reverse_confirm_frames_, 3);

    // -- 速度限制 --
    nh.param("max_linear_vel",      max_linear_vel_,      0.5);
    nh.param("min_linear_vel",      min_linear_vel_,     -0.5);
    nh.param("max_angular_vel",     max_angular_vel_,     1.0);
    nh.param("linear_gain",         linear_gain_,         3.0);

    // -- 碰撞检测 (原有参数, 保留兼容) --
    nh.param("collision_check_count",   collision_check_count_,   10);
    nh.param("collision_cooldown_max",  collision_cooldown_max_,  20);
    nh.param("collision_replan_max",    collision_replan_max_,    2);

    // -- 可视化 --
    nh.param("enable_visualization",  enable_visualization_, false);
    nh.param("enable_costmap_pub",    enable_costmap_pub_,   false);

    // -- 位姿调整 --
    nh.param("pose_dist_threshold", pose_dist_threshold_, 0.1);
    nh.param("pos_deadband",        pos_deadband_,        0.015);
    nh.param("yaw_deadband",        yaw_deadband_,        0.02);
    nh.param("angle_adjust_gain",   angle_adjust_gain_,   1.0);
    nh.param("slow_zone",           slow_zone_,           0.2);
    nh.param("min_adjust_speed",    min_adjust_speed_,    0.08);
    nh.param("pose_tolerance",      pose_tolerance_,      0.015);
    nh.param("pose_adjust_timeout", pose_adjust_timeout_, 15.0);
    nh.param("pose_adjust_max_linear", pose_adjust_max_linear_, 0.10);

    // ====== 【新增参数】 ======

    // -- 前向碰撞检测 --
    nh.param("enable_forward_sim",   enable_forward_sim_,  true);
    nh.param("forward_sim_time",     forward_sim_time_,    1.0);
    nh.param("forward_sim_step",     forward_sim_step_,    0.05);

    // -- 轨迹采样 --
    nh.param("enable_traj_sampling", enable_traj_sampling_, true);
    nh.param("traj_samples_vx",      traj_samples_vx_,     7);
    nh.param("traj_samples_wz",      traj_samples_wz_,     11);
    nh.param("traj_delta_vx",        traj_delta_vx_,       0.10);
    nh.param("traj_delta_wz",        traj_delta_wz_,       0.30);

    // -- 代价权重 --
    nh.param("weight_obstacle",      weight_obstacle_,     10.0);
    nh.param("weight_path",          weight_path_,         1.0);
    nh.param("weight_goal",          weight_goal_,         0.5);

    // -- 速度平滑 --
    nh.param("enable_vel_smooth",    enable_vel_smooth_,   true);
    nh.param("max_linear_accel",     max_linear_accel_,    2.0);
    nh.param("max_linear_decel",     max_linear_decel_,    2.5);
    nh.param("max_angular_accel",    max_angular_accel_,   3.2);
    nh.param("max_angular_decel",    max_angular_decel_,   3.2);

    // -- 智能恢复 (窄道参数: 只能微转~10°) --
    nh.param("enable_smart_recovery", enable_smart_recovery_, true);
    nh.param("recovery_rotation_speed", recovery_rotation_speed_, 0.3);
    nh.param("recovery_rotation_angle", recovery_rotation_angle_, 0.175);  // ~10°
    nh.param("recovery_backup_distance", recovery_backup_distance_, 0.10); // 卡死恢复: 先倒退10cm
    nh.param("recovery_backup_speed", recovery_backup_speed_, 0.08);       // 卡死恢复: 倒退速度

    // -- 窄道模式 --
    nh.param("narrow_track_mode",     narrow_track_mode_,     true);
    nh.param("track_width",           track_width_,           0.50);
    nh.param("min_side_clearance",    min_side_clearance_,    0.06);
    nh.param("lateral_check_distance", lateral_check_distance_, 0.26);

    // -- 分级速度策略 --
    nh.param("enable_graded_speed",   enable_graded_speed_,  true);
    nh.param("danger_dist_far",       danger_dist_far_,      0.80);
    nh.param("danger_dist_mid",       danger_dist_mid_,      0.40);
    nh.param("danger_dist_near",      danger_dist_near_,     0.18);
    nh.param("speed_ratio_mid",       speed_ratio_mid_,      0.50);
    nh.param("speed_ratio_near",      speed_ratio_near_,     0.20);

    // -- 状态初始化 --
    target_index_      = 0;
    prev_target_index_ = -1;
    pose_adjusting_    = false;
    goal_reached_      = false;
    reverse_mode_      = false;
    reverse_confirm_cnt_ = 0;
    error_sum_         = 0.0;
    last_error_        = 0.0;
    param_reload_cnt_  = 0;

    // 碰撞检测冷却初始化
    collision_cooldown_     = 0;
    collision_replan_count_ = 0;

    // 速度平滑初始化
    last_cmd_vel_.linear.x  = 0.0;
    last_cmd_vel_.linear.y  = 0.0;
    last_cmd_vel_.linear.z  = 0.0;
    last_cmd_vel_.angular.x = 0.0;
    last_cmd_vel_.angular.y = 0.0;
    last_cmd_vel_.angular.z = 0.0;

    // 恢复状态初始化
    in_recovery_               = false;
    recovery_backup_phase_     = false;
    recovery_yaw_target_       = 0.0;
    recovery_yaw_accum_        = 0.0;
    recovery_prev_yaw_         = 0.0;
    recovery_yaw_initialized_  = false;

    // 代价地图可视化发布
    if (enable_costmap_pub_)
    {
        ros::NodeHandle nh_pub;
        costmap_pub_ = nh_pub.advertise<sensor_msgs::Image>("/iden_planner_v2/costmap", 1);
    }

    // 轨迹候选可视化
    {
        ros::NodeHandle nh_pub;
        traj_viz_pub_ = nh_pub.advertise<visualization_msgs::Marker>(
            "/iden_planner_v2/trajectory_candidates", 1);
    }

    ROS_WARN("IdenPlannerV2 启动! 预瞄=%.2fm Kp=%.1f 前向仿真=%s 轨迹采样=%s 速度平滑=%s 智能恢复=%s",
             lookahead_dist_, kp_,
             enable_forward_sim_ ? "ON" : "OFF",
             enable_traj_sampling_ ? "ON" : "OFF",
             enable_vel_smooth_ ? "ON" : "OFF",
             enable_smart_recovery_ ? "ON" : "OFF",
             narrow_track_mode_ ? "窄道模式" : "宽道模式");
}

// ============================================================
//  setPlan — 接收新全局路径
// ============================================================

bool IdenPlannerV2::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan)
{
    target_index_      = 0;
    prev_target_index_ = -1;
    global_plan_       = plan;
    pose_adjusting_    = false;
    goal_reached_      = false;
    reverse_mode_      = false;
    reverse_confirm_cnt_ = 0;
    error_sum_         = 0.0;
    last_error_        = 0.0;
    collision_replan_count_ = 0;
    in_recovery_       = false;
    recovery_backup_phase_ = false;
    recovery_yaw_accum_ = 0.0;
    recovery_yaw_initialized_ = false;
    return true;
}

// ============================================================
//  isGoalReached
// ============================================================

bool IdenPlannerV2::isGoalReached()
{
    return goal_reached_;
}

// ============================================================
//  参数热重载（每秒一次）
// ============================================================

void IdenPlannerV2::reloadParams()
{
    ros::NodeHandle nh(param_ns_);

    nh.getParam("kp",                  kp_);
    nh.getParam("ki",                  ki_);
    nh.getParam("kd",                  kd_);
    nh.getParam("integral_limit",      integral_limit_);
    nh.getParam("lookahead_dist",      lookahead_dist_);
    nh.getParam("angle_power",         angle_power_);
    nh.getParam("angle_full_speed_deg", angle_full_speed_deg_);
    nh.getParam("angle_zero_speed_deg", angle_zero_speed_deg_);
    nh.getParam("enable_bidirectional", enable_bidirectional_);
    nh.getParam("long_lookahead",      long_lookahead_);
    nh.getParam("reverse_vote_ratio",  reverse_vote_ratio_);
    nh.getParam("reverse_confirm_frames", reverse_confirm_frames_);
    nh.getParam("max_linear_vel",      max_linear_vel_);
    nh.getParam("min_linear_vel",      min_linear_vel_);
    nh.getParam("max_angular_vel",     max_angular_vel_);
    nh.getParam("linear_gain",         linear_gain_);
    nh.getParam("collision_check_count",   collision_check_count_);
    nh.getParam("collision_cooldown_max",  collision_cooldown_max_);
    nh.getParam("collision_replan_max",    collision_replan_max_);
    nh.getParam("enable_visualization",  enable_visualization_);
    nh.getParam("enable_costmap_pub",    enable_costmap_pub_);
    nh.getParam("pose_dist_threshold", pose_dist_threshold_);
    nh.getParam("pos_deadband",        pos_deadband_);
    nh.getParam("yaw_deadband",        yaw_deadband_);
    nh.getParam("angle_adjust_gain",   angle_adjust_gain_);
    nh.getParam("slow_zone",           slow_zone_);
    nh.getParam("min_adjust_speed",    min_adjust_speed_);
    nh.getParam("pose_tolerance",      pose_tolerance_);
    nh.getParam("pose_adjust_timeout", pose_adjust_timeout_);
    nh.getParam("pose_adjust_max_linear", pose_adjust_max_linear_);

    // 新增参数热重载
    nh.getParam("enable_forward_sim",  enable_forward_sim_);
    nh.getParam("forward_sim_time",    forward_sim_time_);
    nh.getParam("forward_sim_step",    forward_sim_step_);
    nh.getParam("enable_traj_sampling", enable_traj_sampling_);
    nh.getParam("traj_samples_vx",     traj_samples_vx_);
    nh.getParam("traj_samples_wz",     traj_samples_wz_);
    nh.getParam("traj_delta_vx",       traj_delta_vx_);
    nh.getParam("traj_delta_wz",       traj_delta_wz_);
    nh.getParam("weight_obstacle",     weight_obstacle_);
    nh.getParam("weight_path",         weight_path_);
    nh.getParam("weight_goal",         weight_goal_);
    nh.getParam("enable_vel_smooth",   enable_vel_smooth_);
    nh.getParam("max_linear_accel",    max_linear_accel_);
    nh.getParam("max_linear_decel",    max_linear_decel_);
    nh.getParam("max_angular_accel",   max_angular_accel_);
    nh.getParam("max_angular_decel",   max_angular_decel_);
    nh.getParam("enable_smart_recovery", enable_smart_recovery_);
    nh.getParam("recovery_rotation_speed", recovery_rotation_speed_);
    nh.getParam("recovery_rotation_angle", recovery_rotation_angle_);
    nh.getParam("recovery_backup_distance", recovery_backup_distance_);
    nh.getParam("recovery_backup_speed", recovery_backup_speed_);
    nh.getParam("narrow_track_mode",     narrow_track_mode_);
    nh.getParam("track_width",           track_width_);
    nh.getParam("min_side_clearance",    min_side_clearance_);
    nh.getParam("lateral_check_distance", lateral_check_distance_);
    nh.getParam("enable_graded_speed",   enable_graded_speed_);
    nh.getParam("danger_dist_far",       danger_dist_far_);
    nh.getParam("danger_dist_mid",       danger_dist_mid_);
    nh.getParam("danger_dist_near",      danger_dist_near_);
    nh.getParam("speed_ratio_mid",       speed_ratio_mid_);
    nh.getParam("speed_ratio_near",      speed_ratio_near_);
}

// ============================================================
//  【原有】角度-速度耦合
// ============================================================

double IdenPlannerV2::computeAngleSpeedScale(double angle_error_rad)
{
    double full  = angle_full_speed_deg_ * M_PI / 180.0;
    double zero  = angle_zero_speed_deg_ * M_PI / 180.0;
    double abs_a = std::fabs(angle_error_rad);

    if (abs_a <= full)  return 1.0;
    if (abs_a >= zero)  return 0.0;

    double t = (abs_a - full) / (zero - full);
    double scale = 1.0 - std::pow(t, angle_power_);

    if (scale < 0.0) scale = 0.0;
    if (scale > 1.0) scale = 1.0;
    return scale;
}

// ============================================================
//  【原有】角速度 PID（带积分抗饱和 + 航点切换重置）
// ============================================================

double IdenPlannerV2::computeAngularPID(double lateral_error)
{
    error_sum_ += lateral_error;

    if (error_sum_ >  integral_limit_) error_sum_ =  integral_limit_;
    if (error_sum_ < -integral_limit_) error_sum_ = -integral_limit_;

    double error_diff = lateral_error - last_error_;
    double output = kp_ * lateral_error
                  + ki_ * error_sum_
                  + kd_ * error_diff;

    last_error_ = lateral_error;
    return output;
}

// ============================================================
//  【原有】选择预瞄路径点
// ============================================================

geometry_msgs::PoseStamped IdenPlannerV2::selectLookaheadTarget()
{
    geometry_msgs::PoseStamped target;
    target.pose.orientation.w = 1.0;

    for (int i = target_index_; i < (int)global_plan_.size(); i++)
    {
        geometry_msgs::PoseStamped pose_base;
        global_plan_[i].header.stamp = ros::Time(0);
        try
        {
            tf_listener_->transformPose("base_link", global_plan_[i], pose_base);
        }
        catch (tf::TransformException& ex)
        {
            ROS_ERROR_THROTTLE(1.0, "TF 变换失败: %s", ex.what());
            continue;
        }

        double dx   = pose_base.pose.position.x;
        double dy   = pose_base.pose.position.y;
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist > lookahead_dist_)
        {
            target       = pose_base;
            target_index_ = i;
            return target;
        }

        if (i == (int)global_plan_.size() - 1) {
            target       = pose_base;
            target_index_ = i;
        }
    }
    return target;
}

// ============================================================
//  【原有】双向导航：分析路径方向
// ============================================================

bool IdenPlannerV2::analyzePathDirection()
{
    if (!enable_bidirectional_)
        return false;

    int behind = 0, ahead = 0, total = 0;

    for (int i = target_index_; i < (int)global_plan_.size(); i++)
    {
        geometry_msgs::PoseStamped pose_base;
        global_plan_[i].header.stamp = ros::Time(0);
        try
        {
            tf_listener_->transformPose("base_link", global_plan_[i], pose_base);
        }
        catch (...) { continue; }

        double lx = pose_base.pose.position.x;
        double ly = pose_base.pose.position.y;
        double ld = std::hypot(lx, ly);

        if (ld > 0.05 && ld < long_lookahead_)
        {
            total++;
            if (lx < -0.05) behind++;
            if (lx >  0.05) ahead++;
        }
        if (ld >= long_lookahead_) break;
    }

    if (total < 3) return reverse_mode_;

    double behind_ratio = (double)behind / total;
    double ahead_ratio  = (double)ahead  / total;

    if (behind_ratio > reverse_vote_ratio_) return true;
    if (ahead_ratio  > reverse_vote_ratio_) return false;
    return reverse_mode_;
}

void IdenPlannerV2::updateReverseState(bool want_reverse)
{
    if (want_reverse != reverse_mode_)
    {
        reverse_confirm_cnt_++;
        if (reverse_confirm_cnt_ >= reverse_confirm_frames_)
        {
            reverse_mode_ = want_reverse;
            reverse_confirm_cnt_ = 0;
            error_sum_  = 0.0;
            last_error_ = 0.0;
            ROS_WARN("IdenPlannerV2: 切换为 %s 模式", reverse_mode_ ? "倒车" : "前进");
        }
    }
    else
    {
        reverse_confirm_cnt_ = 0;
    }
}

// ============================================================
//  【原有】位姿最终调整
// ============================================================

bool IdenPlannerV2::computePoseAdjust(geometry_msgs::Twist& cmd_vel,
                                      const geometry_msgs::PoseStamped& pose_final)
{
    double dx       = pose_final.pose.position.x;
    double dy       = pose_final.pose.position.y;
    double final_yaw = tf::getYaw(pose_final.pose.orientation);

    double desired_linear  = 0.0;
    double desired_angular = 0.0;

    if (std::fabs(dx) > pos_deadband_)
        desired_linear = dx * 0.8;

    desired_linear = std::max(-pose_adjust_max_linear_,
                      std::min(pose_adjust_max_linear_, desired_linear));

    if (std::fabs(final_yaw) > yaw_deadband_)
    {
        double abs_yaw  = std::fabs(final_yaw);
        double raw_speed = abs_yaw * angle_adjust_gain_;

        double speed;
        if (abs_yaw > slow_zone_)
        {
            speed = raw_speed;
            if (speed > max_angular_vel_) speed = max_angular_vel_;
        }
        else
        {
            speed = min_adjust_speed_
                  + (raw_speed - min_adjust_speed_) * (abs_yaw / slow_zone_);
        }
        desired_angular = (final_yaw > 0) ? speed : -speed;
    }

    if (std::fabs(final_yaw) < pose_tolerance_ &&
        std::fabs(dx) < pos_deadband_)
    {
        goal_reached_ = true;
        ROS_WARN("IdenPlannerV2: 精确到达目标点!");
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        return true;
    }

    cmd_vel.linear.x  = desired_linear;
    cmd_vel.angular.z = desired_angular;
    return true;
}

// ============================================================
//  【新增1】足印碰撞检测
//   在(x,y,theta)姿态下用完整机器人footprint检查是否碰撞
// ============================================================

bool IdenPlannerV2::isFootprintInCollision(double x, double y, double theta)
{
    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (!costmap) return false;

    std::vector<geometry_msgs::Point> footprint = costmap_ros_->getRobotFootprint();
    if (footprint.empty()) {
        // 没有 footprint → 退化为中心点检查
        unsigned int mx, my;
        if (!costmap->worldToMap(x, y, mx, my)) return true;
        unsigned char cost = costmap->getCost(mx, my);
        return cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE;
    }

    // 旋转 footprint 到目标朝向
    double cos_th = cos(theta);
    double sin_th = sin(theta);

    std::vector<geometry_msgs::Point> oriented_footprint = footprint;
    for (auto& pt : oriented_footprint)
    {
        double new_x = x + pt.x * cos_th - pt.y * sin_th;
        double new_y = y + pt.x * sin_th + pt.y * cos_th;
        pt.x = new_x;
        pt.y = new_y;
    }

    // 检查所有足印顶点
    unsigned int size_x = costmap->getSizeInCellsX();
    unsigned int size_y = costmap->getSizeInCellsY();

    for (const auto& pt : oriented_footprint)
    {
        unsigned int mx, my;
        if (!costmap->worldToMap(pt.x, pt.y, mx, my)) return true;

        unsigned char cost = costmap->getCost(mx, my);
        if (cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
            return true;
    }

    // 在足印多边形连线上插值检查 (防止大足印顶点间漏检)
    for (size_t i = 0; i < oriented_footprint.size(); i++)
    {
        size_t j = (i + 1) % oriented_footprint.size();
        double dx = oriented_footprint[j].x - oriented_footprint[i].x;
        double dy = oriented_footprint[j].y - oriented_footprint[i].y;
        double len = std::hypot(dx, dy);
        double res = costmap->getResolution();
        int steps = std::max(1, (int)(len / (res * 0.5)));

        for (int k = 1; k < steps; k++)
        {
            double t = (double)k / steps;
            double px = oriented_footprint[i].x + dx * t;
            double py = oriented_footprint[i].y + dy * t;
            unsigned int mx, my;
            if (!costmap->worldToMap(px, py, mx, my)) return true;
            unsigned char cost = costmap->getCost(mx, my);
            if (cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
                return true;
        }
    }

    return false;
}

// ============================================================
//  【新增1b】侧向间隙监测
//   在指定位置检查左右两侧的可用间隙
//   窄道核心安全机制: 确保轨迹不会让机器人擦墙
// ============================================================

bool IdenPlannerV2::checkSideClearance(double x, double y, double theta,
                                        double* left_clearance,
                                        double* right_clearance)
{
    if (!narrow_track_mode_) return true;  // 非窄道模式, 跳过

    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (!costmap) return true;  // 无代价地图时无法检查, 放行

    double check_dist = lateral_check_distance_;  // 检查距离 = 机器人宽度
    double cos_th = cos(theta);
    double sin_th = sin(theta);
    double resolution = costmap->getResolution();
    unsigned int size_x = costmap->getSizeInCellsX();
    unsigned int size_y = costmap->getSizeInCellsY();

    // 向左扫描 (theta + 90°)
    double left_min = 999.0;
    {
        double lx = cos(theta + M_PI_2);
        double ly = sin(theta + M_PI_2);
        for (double d = resolution; d <= check_dist; d += resolution)
        {
            double px = x + lx * d;
            double py = y + ly * d;
            unsigned int mx, my;
            if (!costmap->worldToMap(px, py, mx, my)) break;
            unsigned char cost = costmap->getCost(mx, my);
            if (cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
            {
                left_min = d;
                break;
            }
        }
    }

    // 向右扫描 (theta - 90°)
    double right_min = 999.0;
    {
        double rx = cos(theta - M_PI_2);
        double ry = sin(theta - M_PI_2);
        for (double d = resolution; d <= check_dist; d += resolution)
        {
            double px = x + rx * d;
            double py = y + ry * d;
            unsigned int mx, my;
            if (!costmap->worldToMap(px, py, mx, my)) break;
            unsigned char cost = costmap->getCost(mx, my);
            if (cost >= costmap_2d::INSCRIBED_INFLATED_OBSTACLE)
            {
                right_min = d;
                break;
            }
        }
    }

    if (left_clearance)  *left_clearance  = left_min;
    if (right_clearance) *right_clearance = right_min;

    return (left_min >= min_side_clearance_) &&
           (right_min >= min_side_clearance_);
}

// ============================================================
//  【新增2】前向时序碰撞检测
//   将当前速度指令投影到未来,逐时间步检查足印碰撞
//   窄道模式: 每步也检查侧向间隙
// ============================================================

bool IdenPlannerV2::forwardSimulateCollision(double vx, double wz,
                                             double sim_time,
                                             double* collision_dist)
{
    if (!enable_forward_sim_) return false;

    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (!costmap) return false;

    // 获取当前位姿
    tf::StampedTransform transform;
    try
    {
        tf_listener_->lookupTransform(costmap_ros_->getGlobalFrameID(),
                                      costmap_ros_->getBaseFrameID(),
                                      ros::Time(0), transform);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR_THROTTLE(1.0, "forwardSim: TF error: %s", ex.what());
        return false;
    }

    double cx = transform.getOrigin().x();
    double cy = transform.getOrigin().y();
    double cyaw = tf::getYaw(transform.getRotation());

    double step = forward_sim_step_;
    double accum_dist = 0.0;

    for (double t = step; t <= sim_time; t += step)
    {
        // 差速模型积分
        if (std::fabs(wz) > 1e-6)
        {
            double radius = vx / wz;
            double dtheta = wz * step;
            cx += radius * (sin(cyaw + dtheta) - sin(cyaw));
            cy -= radius * (cos(cyaw + dtheta) - cos(cyaw));
            cyaw += dtheta;
        }
        else
        {
            cx += vx * step * cos(cyaw);
            cy += vx * step * sin(cyaw);
        }
        accum_dist += std::fabs(vx) * step;

        // 检查足印碰撞
        if (isFootprintInCollision(cx, cy, cyaw))
        {
            if (collision_dist) *collision_dist = accum_dist;
            return true;
        }

        // 窄道模式: 额外检查侧向间隙
        if (narrow_track_mode_)
        {
            double left_clr, right_clr;
            bool side_ok = checkSideClearance(cx, cy, cyaw, &left_clr, &right_clr);
            if (!side_ok)
            {
                ROS_WARN_THROTTLE(0.5,
                    "IdenPlannerV2: 侧向间隙不足! L=%.2fm R=%.2fm (需要≥%.2fm)",
                    left_clr, right_clr, min_side_clearance_);
                // 侧向碰撞也触发碰撞检测, 但给予更宽松的阈值
                if (std::min(left_clr, right_clr) < min_side_clearance_ * 0.5)
                {
                    if (collision_dist) *collision_dist = accum_dist;
                    return true;
                }
            }
        }
    }

    return false;
}

// ============================================================
//  【新增3】轨迹候选生成
//   在当前速度附近按网格采样候选速度
// ============================================================

std::vector<TrajectoryCandidate> IdenPlannerV2::generateTrajectoryCandidates(
    double current_vx, double current_wz)
{
    std::vector<TrajectoryCandidate> candidates;

    // 确保采样数是奇数,保证包含当前速度
    int half_vx = traj_samples_vx_ / 2;
    int half_wz = traj_samples_wz_ / 2;

    for (int iv = -half_vx; iv <= half_vx; iv++)
    {
        for (int iw = -half_wz; iw <= half_wz; iw++)
        {
            TrajectoryCandidate c;
            c.vx = current_vx + iv * traj_delta_vx_;

            // 线速度限幅
            if (c.vx >  max_linear_vel_) c.vx =  max_linear_vel_;
            if (c.vx <  min_linear_vel_) c.vx =  min_linear_vel_;

            c.wz = current_wz + iw * traj_delta_wz_;

            // 角速度限幅
            if (c.wz >  max_angular_vel_) c.wz =  max_angular_vel_;
            if (c.wz < -max_angular_vel_) c.wz = -max_angular_vel_;

            candidates.push_back(c);
        }
    }

    return candidates;
}

// ============================================================
//  【新增4】轨迹评分
//   对单个候选轨迹进行多维度评分
// ============================================================

void IdenPlannerV2::scoreTrajectory(TrajectoryCandidate& candidate,
                                    double current_vx, double current_wz)
{
    // ---- 维度1: 障碍物代价 (前向仿真) ----
    double collision_dist = 0.0;
    bool will_collide = forwardSimulateCollision(candidate.vx, candidate.wz,
                                                  forward_sim_time_, &collision_dist);

    if (will_collide)
    {
        if (collision_dist < 0.05)
        {
            candidate.is_valid = false;
            candidate.obstacle_cost = 1e6;
            candidate.score = 1e9;
            return;
        }
        else
        {
            candidate.obstacle_cost = weight_obstacle_ * (forward_sim_time_ / collision_dist);
        }
    }
    else
    {
        candidate.obstacle_cost = 0.0;
    }

    // 窄道模式: 额外评估侧向安全性
    // 在 0.3s 投影后检查侧向间隙, 间隙越小惩罚越大
    if (narrow_track_mode_ && candidate.is_valid)
    {
        tf::StampedTransform transform;
        try
        {
            tf_listener_->lookupTransform(costmap_ros_->getGlobalFrameID(),
                                          costmap_ros_->getBaseFrameID(),
                                          ros::Time(0), transform);
        }
        catch (...) { return; }

        double cx = transform.getOrigin().x();
        double cy = transform.getOrigin().y();
        double cyaw = tf::getYaw(transform.getRotation());

        // 投影 0.3s
        double sim_t = 0.3;
        double steps = sim_t / forward_sim_step_;
        double fx = cx, fy = cy, fyaw = cyaw;
        for (int s = 0; s < (int)steps; s++)
        {
            if (std::fabs(candidate.wz) > 1e-6)
            {
                double r = candidate.vx / candidate.wz;
                double dt = forward_sim_step_;
                fx += r * (sin(fyaw + candidate.wz * dt) - sin(fyaw));
                fy -= r * (cos(fyaw + candidate.wz * dt) - cos(fyaw));
                fyaw += candidate.wz * dt;
            }
            else
            {
                fx += candidate.vx * forward_sim_step_ * cos(fyaw);
                fy += candidate.vx * forward_sim_step_ * sin(fyaw);
            }
        }

        double left_clr, right_clr;
        checkSideClearance(fx, fy, fyaw, &left_clr, &right_clr);

        // 在窄道中, 侧向间隙不足是严重问题
        double min_clr = std::min(left_clr, right_clr);
        if (min_clr < min_side_clearance_ * 0.5)
        {
            // 只有不到一半的最小间隙 → 此轨迹危险
            candidate.is_valid = false;
            candidate.obstacle_cost = 1e6;
            candidate.score = 1e9;
            return;
        }
        else if (min_clr < min_side_clearance_)
        {
            // 侧向间隙不足 → 额外惩罚
            double lr = min_clr / min_side_clearance_;
            candidate.obstacle_cost += weight_obstacle_ * 2.0 * (1.0 - lr);
        }
    }

    // ---- 维度2: 路径偏差代价 ----
    // 基于新的速度指令计算预瞄点角度偏差
    // 偏差越大 → 代价越高
    double path_deviation = 0.0;
    {
        // 用类似 selectLookaheadTarget 的方式评估
        // 简化: 用当前 target_index 对应的全局路径方向作为参考
        if (!global_plan_.empty() && target_index_ < (int)global_plan_.size())
        {
            geometry_msgs::PoseStamped path_pose;
            global_plan_[target_index_].header.stamp = ros::Time(0);
            try
            {
                tf_listener_->transformPose("base_link", global_plan_[target_index_], path_pose);
                double desired_heading = std::atan2(path_pose.pose.position.y,
                                                     path_pose.pose.position.x);

                // 用圆模型: 轨迹方向 ≈ wz 引起的朝向变化
                // 简化: 把角速度偏差作为路径偏差的代理
                // wz越大 → 转弯越剧烈 → 偏离路径越多
                double heading_from_traj = candidate.wz * forward_sim_time_ * 0.5;
                path_deviation = std::fabs(heading_from_traj - desired_heading);
            }
            catch (...) { path_deviation = std::fabs(candidate.wz); }
        }
        else
        {
            path_deviation = std::fabs(candidate.wz);
        }
    }
    candidate.path_cost = weight_path_ * path_deviation;

    // ---- 维度3: 目标进度代价 ----
    // 偏好向目标方向移动的速度
    double goal_progress = 0.0;
    {
        // 越接近目标(高速朝前) → 代价越低
        double effective_vx = candidate.vx;
        if (reverse_mode_) effective_vx = -effective_vx;
        goal_progress = -(effective_vx * 0.1);  // 负值=奖励
    }
    candidate.goal_cost = weight_goal_ * goal_progress;

    // ---- 总分 ----
    candidate.score = candidate.obstacle_cost
                    + candidate.path_cost
                    + candidate.goal_cost;
}

// ============================================================
//  【新增5】选择最佳候选
// ============================================================

TrajectoryCandidate IdenPlannerV2::selectBestCandidate(
    const std::vector<TrajectoryCandidate>& candidates)
{
    TrajectoryCandidate best;
    best.score = 1e9;
    best.is_valid = false;

    for (const auto& c : candidates)
    {
        if (c.is_valid && c.score < best.score)
            best = c;
    }

    // 如果所有候选都不合法, 选一个碰撞距离最远的
    if (!best.is_valid)
    {
        ROS_WARN_THROTTLE(1.0, "IdenPlannerV2: 所有轨迹候选都碰撞! 选最优的减速通过");
        best.score = 1e9;
        for (const auto& c : candidates)
        {
            if (c.score < best.score)
                best = c;
        }
        // 强制大幅减速
        best.vx *= 0.2;
        best.wz *= 0.2;
    }

    return best;
}

// ============================================================
//  【新增6】速度平滑
//   用加/减速度约束平滑速度指令
// ============================================================

geometry_msgs::Twist IdenPlannerV2::applyVelocitySmoothing(
    const geometry_msgs::Twist& desired,
    const geometry_msgs::Twist& current)
{
    if (!enable_vel_smooth_)
        return desired;

    geometry_msgs::Twist smoothed = desired;

    // 假设控制周期 ~0.1s (配合 move_base 的 controller_frequency)
    double dt = 0.1;

    double dvx = desired.linear.x - current.linear.x;

    // 线加速度/减速度限制
    if (dvx > 0)
    {
        // 加速
        double max_dv = max_linear_accel_ * dt;
        if (dvx > max_dv)
            smoothed.linear.x = current.linear.x + max_dv;
    }
    else
    {
        // 减速 (允许更快减速)
        double max_dv = max_linear_decel_ * dt;
        if (-dvx > max_dv)
            smoothed.linear.x = current.linear.x - max_dv;
    }

    // 角加/减速度限制
    double dwz = desired.angular.z - current.angular.z;
    double max_dw = max_angular_accel_ * dt;
    if (std::fabs(dwz) > max_dw)
        smoothed.angular.z = current.angular.z + (dwz > 0 ? max_dw : -max_dw);

    return smoothed;
}

// ============================================================
//  【新增7】智能恢复：寻找安全旋转方向
//   扫描不同角度寻找无障碍方向
// ============================================================

double IdenPlannerV2::findSafeRotationDirection()
{
    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (!costmap) return 0.0;

    tf::StampedTransform transform;
    try
    {
        tf_listener_->lookupTransform(costmap_ros_->getGlobalFrameID(),
                                      costmap_ros_->getBaseFrameID(),
                                      ros::Time(0), transform);
    }
    catch (...) { return 0.0; }

    double cx = transform.getOrigin().x();
    double cy = transform.getOrigin().y();
    double cyaw = tf::getYaw(transform.getRotation());

    // 窄道模式: 用 side clearance 直接扫描
    // 非窄道模式: 检查 30°/60°/90° 方向上代价
    if (narrow_track_mode_)
    {
        // 检查当前朝向的侧向间隙
        double left_clr = 999.0, right_clr = 999.0;
        checkSideClearance(cx, cy, cyaw, &left_clr, &right_clr);

        // 检查微转后的侧向间隙
        double wiggle = recovery_rotation_angle_;  // ~10°
        double left_after_rotate = 999.0, right_after_rotate = 999.0;

        // 左转 wiggle° 后的间隙
        double new_yaw_left = cyaw + wiggle;
        double lx_l = cx + 0.05 * cos(new_yaw_left);
        double ly_l = cy + 0.05 * sin(new_yaw_left);
        checkSideClearance(lx_l, ly_l, new_yaw_left, nullptr, &right_after_rotate);

        // 右转 wiggle° 后的间隙
        double new_yaw_right = cyaw - wiggle;
        double lx_r = cx + 0.05 * cos(new_yaw_right);
        double ly_r = cy + 0.05 * sin(new_yaw_right);
        checkSideClearance(lx_r, ly_r, new_yaw_right, &left_after_rotate, nullptr);

        // 决策: 哪侧转后有足够间隙
        bool left_safe  = (right_after_rotate > min_side_clearance_);
        bool right_safe = (left_after_rotate  > min_side_clearance_);

        if (left_safe && right_safe)
            return (left_clr > right_clr) ? recovery_rotation_speed_ : -recovery_rotation_speed_;
        else if (left_safe)
            return recovery_rotation_speed_;   // 向左微转安全
        else if (right_safe)
            return -recovery_rotation_speed_;  // 向右微转安全
        else
        {
            ROS_WARN_THROTTLE(1.0, "IdenPlannerV2: 窄道两侧都无法安全旋转, 微转左");
            return recovery_rotation_speed_ * 0.3;  // 极小幅度旋转
        }
    }
    else
    {
        // 宽道模式: 原逻辑
        double scan_angles[] = { 0.52, 1.05, 1.57 };
        double left_cost = 0.0, right_cost = 0.0;
        double lookahead_m = 0.4;

        for (double da : scan_angles)
        {
            {
                double ax = cx + lookahead_m * cos(cyaw + da);
                double ay = cy + lookahead_m * sin(cyaw + da);
                unsigned int mx, my;
                if (costmap->worldToMap(ax, ay, mx, my))
                    left_cost += costmap->getCost(mx, my);
                else left_cost += 254;
            }
            {
                double ax = cx + lookahead_m * cos(cyaw - da);
                double ay = cy + lookahead_m * sin(cyaw - da);
                unsigned int mx, my;
                if (costmap->worldToMap(ax, ay, mx, my))
                    right_cost += costmap->getCost(mx, my);
                else right_cost += 254;
            }
        }

        if (left_cost < right_cost)
            return recovery_rotation_speed_;
        else
            return -recovery_rotation_speed_;
    }
}

// ============================================================
//  【新增8】恢复阶段
//   原地旋转寻找无障碍方向
// ============================================================

bool IdenPlannerV2::computeRecoveryPhase(geometry_msgs::Twist& cmd_vel)
{
    // ========================================================
    //  恢复阶段分两步:
    //    1) 先后退约 recovery_backup_distance_ (默认 10cm)
    //    2) 再做小角度旋转，返回 false 触发 move_base 重新规划
    //  这样比原地一直转更适合“靠墙点 + 锥桶区”的任务。
    // ========================================================

    if (recovery_backup_distance_ < 0.0)
        recovery_backup_distance_ = 0.0;
    if (recovery_backup_speed_ < 0.02)
        recovery_backup_speed_ = 0.02;

    // 第一阶段：倒退固定距离
    if (!recovery_backup_phase_ && !recovery_yaw_initialized_)
    {
        recovery_backup_phase_ = true;
        recovery_backup_start_ = ros::Time::now();
        ROS_WARN("IdenPlannerV2: 恢复模式启动，先倒退 %.2fm", recovery_backup_distance_);
    }

    if (recovery_backup_phase_)
    {
        double backup_time = recovery_backup_distance_ / recovery_backup_speed_;
        double elapsed = (ros::Time::now() - recovery_backup_start_).toSec();

        if (elapsed < backup_time)
        {
            cmd_vel.linear.x  = -recovery_backup_speed_;
            cmd_vel.angular.z = 0.0;
            return true;
        }

        recovery_backup_phase_ = false;
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        ROS_WARN("IdenPlannerV2: 倒退恢复完成，开始小角度旋转");
    }

    // 获取当前 yaw (用于累计旋转角度)
    tf::StampedTransform transform;
    try
    {
        tf_listener_->lookupTransform(costmap_ros_->getGlobalFrameID(),
                                      costmap_ros_->getBaseFrameID(),
                                      ros::Time(0), transform);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR_THROTTLE(1.0, "Recovery TF error: %s", ex.what());
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        return false;
    }

    double current_yaw = tf::getYaw(transform.getRotation());

    if (!recovery_yaw_initialized_)
    {
        recovery_yaw_initialized_ = true;
        recovery_prev_yaw_ = current_yaw;
        recovery_yaw_accum_ = 0.0;
        recovery_yaw_target_ = findSafeRotationDirection();

        if (std::fabs(recovery_yaw_target_) < 0.01)
        {
            // 找不到安全方向, 尝试左右交替
            recovery_yaw_target_ = recovery_rotation_speed_;  // 先左转
        }
        ROS_WARN("IdenPlannerV2: 恢复旋转方向=%s, 最大旋转 %.1f°",
                 recovery_yaw_target_ > 0 ? "LEFT" : "RIGHT",
                 recovery_rotation_angle_ * 57.3);
    }

    // 累计旋转角度
    double delta = current_yaw - recovery_prev_yaw_;
    if (delta >  M_PI) delta -= 2.0 * M_PI;
    if (delta < -M_PI) delta += 2.0 * M_PI;
    recovery_yaw_accum_ += delta;
    recovery_prev_yaw_ = current_yaw;

    // 检查是否旋转了足够的弧度
    if (std::fabs(recovery_yaw_accum_) >= recovery_rotation_angle_)
    {
        ROS_WARN("IdenPlannerV2: 恢复完成: 后退 %.2fm + 旋转 %.1f°，请求重新规划",
                 recovery_backup_distance_,
                 std::fabs(recovery_yaw_accum_) * 57.3);
        in_recovery_ = false;
        recovery_backup_phase_ = false;
        recovery_yaw_initialized_ = false;
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;

        // 返回 false 通知 move_base 当前局部规划失败，从而触发重新规划
        return false;
    }

    // 第二阶段：小角度原地旋转
    cmd_vel.linear.x  = 0.0;
    cmd_vel.angular.z = recovery_yaw_target_;

    // 若前方已经安全，也不要长时间原地转；提前结束并触发重规划
    double collision_dist = 0.0;
    bool blocked = forwardSimulateCollision(max_linear_vel_ * 0.4, 0.0,
                                             0.3, &collision_dist);
    if (!blocked && std::fabs(recovery_yaw_accum_) > 0.08)
    {
        ROS_WARN("IdenPlannerV2: 前方已安全，提前结束恢复并请求重新规划");
        in_recovery_ = false;
        recovery_backup_phase_ = false;
        recovery_yaw_initialized_ = false;
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        return false;
    }

    return true;
}

// ============================================================
//  代价地图可视化 (同 IdenPlanner)
// ============================================================

void IdenPlannerV2::renderCostmap()
{
    if (!costmap_ros_) return;
    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (!costmap) return;
    unsigned int sx = costmap->getSizeInCellsX();
    unsigned int sy = costmap->getSizeInCellsY();

    cv::Mat img(sy, sx, CV_8UC3, cv::Scalar(128, 128, 128));
    unsigned char* data = costmap->getCharMap();

    for (unsigned int y = 0; y < sy; y++)
    {
        for (unsigned int x = 0; x < sx; x++)
        {
            unsigned char c = data[y * sx + x];
            cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
            if      (c == 0)   p = cv::Vec3b(128, 128, 128);
            else if (c == 254) p = cv::Vec3b(0, 0, 0);
            else               p = cv::Vec3b(255 - c, 0, c);
        }
    }

    cv::circle(img, cv::Point(sx / 2, sy / 2), 5, cv::Scalar(0, 255, 0), -1);

    if (enable_visualization_)
    {
        cv::imshow("IdenPlannerV2 Costmap", img);
        cv::waitKey(1);
    }

    if (enable_costmap_pub_)
    {
        cv_bridge::CvImage cv_img;
        cv_img.header.stamp = ros::Time::now();
        cv_img.header.frame_id = costmap_ros_->getGlobalFrameID();
        cv_img.encoding = "bgr8";
        cv_img.image = img;
        costmap_pub_.publish(cv_img.toImageMsg());
    }
}

void IdenPlannerV2::publishCostmap()
{
    if (!enable_costmap_pub_ || !costmap_ros_) return;

    costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
    if (!costmap) return;
    unsigned int sx = costmap->getSizeInCellsX();
    unsigned int sy = costmap->getSizeInCellsY();

    cv::Mat img(sy, sx, CV_8UC3, cv::Scalar(128, 128, 128));
    unsigned char* data = costmap->getCharMap();

    for (unsigned int y = 0; y < sy; y++)
        for (unsigned int x = 0; x < sx; x++)
        {
            unsigned char c = data[y * sx + x];
            cv::Vec3b& p = img.at<cv::Vec3b>(y, x);
            if      (c == 0)   p = cv::Vec3b(128, 128, 128);
            else if (c == 254) p = cv::Vec3b(0, 0, 0);
            else               p = cv::Vec3b(255 - c, 0, c);
        }

    cv::circle(img, cv::Point(sx / 2, sy / 2), 5, cv::Scalar(0, 255, 0), -1);

    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = ros::Time::now();
    cv_img.header.frame_id = costmap_ros_->getGlobalFrameID();
    cv_img.encoding = "bgr8";
    cv_img.image = img;
    costmap_pub_.publish(cv_img.toImageMsg());
}

// ============================================================
//  轨迹候选可视化
// ============================================================

void IdenPlannerV2::publishTrajectoryCandidates(
    const std::vector<TrajectoryCandidate>& candidates,
    const TrajectoryCandidate& best)
{
    if (traj_viz_pub_.getNumSubscribers() == 0) return;

    visualization_msgs::Marker marker;
    marker.header.frame_id = costmap_ros_->getGlobalFrameID();
    marker.header.stamp = ros::Time::now();
    marker.ns = "traj_candidates";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::LINE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.scale.x = 0.01;  // 线宽
    marker.pose.orientation.w = 1.0;

    tf::StampedTransform transform;
    try
    {
        tf_listener_->lookupTransform(costmap_ros_->getGlobalFrameID(),
                                      costmap_ros_->getBaseFrameID(),
                                      ros::Time(0), transform);
    }
    catch (...) { return; }

    double cx = transform.getOrigin().x();
    double cy = transform.getOrigin().y();
    double cyaw = tf::getYaw(transform.getRotation());

    // 为每个候选画短线
    double sim_time = 0.5;  // 可视化50cm投影
    for (const auto& c : candidates)
    {
        double ex = cx + c.vx * sim_time * cos(cyaw);
        double ey = cy + c.vx * sim_time * sin(cyaw);

        geometry_msgs::Point p_start, p_end;
        p_start.x = cx;  p_start.y = cy;  p_start.z = 0;
        p_end.x   = ex;  p_end.y   = ey;  p_end.z   = 0;
        marker.points.push_back(p_start);
        marker.points.push_back(p_end);

        // 颜色: 越接近最优越绿,不安全越红
        std_msgs::ColorRGBA color;
        color.a = 1.0;
        if (!c.is_valid)
        {
            color.r = 1.0; color.g = 0.0; color.b = 0.0;
        }
        else if (&c == &best)
        {
            color.r = 0.0; color.g = 1.0; color.b = 0.0;
        }
        else
        {
            double ratio = std::max(0.0, std::min(1.0, c.score / (best.score + 0.01)));
            color.r = static_cast<float>(ratio);
            color.g = static_cast<float>(1.0 - ratio);
            color.b = 0.0;
        }
        marker.colors.push_back(color);
        marker.colors.push_back(color);
    }

    traj_viz_pub_.publish(marker);
}

// ============================================================
//  位姿调整阶段入口
// ============================================================

bool IdenPlannerV2::computePoseAdjustPhase(geometry_msgs::Twist& cmd_vel)
{
    int final_idx = global_plan_.size() - 1;
    geometry_msgs::PoseStamped pose_final;
    global_plan_[final_idx].header.stamp = ros::Time(0);
    try
    {
        tf_listener_->transformPose("base_link", global_plan_[final_idx], pose_final);
    }
    catch (tf::TransformException& ex)
    {
        ROS_ERROR_THROTTLE(1.0, "TF final pose: %s", ex.what());
        return true;
    }

    if (!pose_adjusting_)
    {
        double dist = std::hypot(pose_final.pose.position.x,
                                 pose_final.pose.position.y);
        if (dist < pose_dist_threshold_)
        {
            pose_adjusting_ = true;
            pose_adjust_start_ = ros::Time::now();
        }
    }

    if (pose_adjusting_)
    {
        double adjust_elapsed = (ros::Time::now() - pose_adjust_start_).toSec();
        if (adjust_elapsed > pose_adjust_timeout_)
        {
            goal_reached_ = true;
            ROS_WARN("IdenPlannerV2: 位姿调整超时(%.1fs), 强制标记到达", adjust_elapsed);
            cmd_vel.linear.x  = 0.0;
            cmd_vel.angular.z = 0.0;
            return true;
        }

        double dist = std::hypot(pose_final.pose.position.x,
                                 pose_final.pose.position.y);
        if (dist > pose_dist_threshold_ + 0.05)
        {
            pose_adjusting_ = false;
            error_sum_  = 0.0;
            last_error_ = 0.0;
            ROS_WARN("IdenPlannerV2: 退出位姿调整, 距离=%.3f", dist);
        }
        else
        {
            return computePoseAdjust(cmd_vel, pose_final);
        }
    }

    return false;  // 不在位姿调整阶段
}

// ============================================================
//  路径追踪阶段（含轨迹采样）
// ============================================================

bool IdenPlannerV2::computeTrackingPhase(geometry_msgs::Twist& cmd_vel)
{
    // ---- 双向导航分析 ----
    if (enable_bidirectional_)
    {
        bool want_reverse = analyzePathDirection();
        updateReverseState(want_reverse);
    }

    // ---- 选择预瞄点 ----
    geometry_msgs::PoseStamped target = selectLookaheadTarget();

    // ---- 双向等效变换 ----
    double eff_dx = target.pose.position.x;
    double eff_dy = target.pose.position.y;
    if (reverse_mode_)
    {
        eff_dx = -eff_dx;
        eff_dy = -eff_dy;
    }

    // ---- 航点切换检测 → 重置积分 ----
    if (target_index_ != prev_target_index_)
    {
        error_sum_  = 0.0;
        last_error_ = 0.0;
        prev_target_index_ = target_index_;
    }

    // ---- 角度-速度耦合: 计算线速度缩放 ----
    double angle_err  = std::atan2(eff_dy, eff_dx);
    double speed_scale = computeAngleSpeedScale(angle_err);
    double desired_linear = eff_dx * linear_gain_ * speed_scale;

    // ---- 角速度 PID ----
    double desired_angular = computeAngularPID(eff_dy);

    // ---- 方向符号 ----
    if (reverse_mode_)
        desired_linear = -desired_linear;

    // ---- 速度限幅(第一次) ----
    if (desired_linear > max_linear_vel_)
        desired_linear = max_linear_vel_;
    else if (desired_linear < min_linear_vel_)
        desired_linear = min_linear_vel_;

    if (desired_angular > max_angular_vel_)
        desired_angular = max_angular_vel_;
    else if (desired_angular < -max_angular_vel_)
        desired_angular = -max_angular_vel_;

    geometry_msgs::Twist desired_pid;
    desired_pid.linear.x  = desired_linear;
    desired_pid.angular.z = desired_angular;

    // ========================================================
    //  【核心增强】轨迹采样 + 评分
    //  如果 enable_traj_sampling_=true, 用采样-评分范式替代直接 PID 输出
    // ========================================================
    geometry_msgs::Twist desired_vel;

    if (enable_traj_sampling_ && !global_plan_.empty())
    {
        // 生成以 PID 输出为中心的候选集
        std::vector<TrajectoryCandidate> candidates =
            generateTrajectoryCandidates(desired_linear, desired_angular);

        // 对每个候选评分
        for (auto& c : candidates)
            scoreTrajectory(c, last_cmd_vel_.linear.x, last_cmd_vel_.angular.z);

        // 选最优
        TrajectoryCandidate best = selectBestCandidate(candidates);

        desired_vel.linear.x  = best.vx;
        desired_vel.angular.z = best.wz;

        // 可视化
        publishTrajectoryCandidates(candidates, best);
    }
    else
    {
        desired_vel = desired_pid;
    }

    // ========================================================
    //  【核心增强】分级速度策略 (参照 avoid.py 的思路)
    //
    //  基于前向仿真的"安全距离"动态调整速度和方向:
    //    clear_dist > danger_dist_far_  (0.80m) → 正常速度
    //    clear_dist ∈ (mid, far]        (0.40~0.80m) → 减速50%
    //    clear_dist ∈ (near, mid]       (0.18~0.40m) → 减速80% + 朝开阔侧偏转
    //    clear_dist < danger_dist_near_ (0.18m)     → 停车 + 智能恢复
    //
    //  与 avoid.py 的区别: 这里基于代价地图的前向足印仿真,
    //  而非原始LiDAR, 兼顾了规划层的路径信息。
    // ========================================================
    double clear_dist = forward_sim_time_;  // 默认: 仿真时间内全清

    if (enable_forward_sim_)
    {
        double collision_dist = 0.0;
        bool will_collide = forwardSimulateCollision(
            desired_vel.linear.x, desired_vel.angular.z,
            forward_sim_time_, &collision_dist);

        if (will_collide)
            clear_dist = collision_dist;  // 碰撞距离 = 安全距离上限

        // 即使没有足印碰撞, 也检查路径上是否有高代价区域(锥桶附近)
        if (!will_collide && !global_plan_.empty())
        {
            // 沿正向投影, 查代价地图中的最高代价
            costmap_2d::Costmap2D* costmap = costmap_ros_->getCostmap();
            if (costmap)
            {
                tf::StampedTransform transform;
                try
                {
                    tf_listener_->lookupTransform(costmap_ros_->getGlobalFrameID(),
                                                  costmap_ros_->getBaseFrameID(),
                                                  ros::Time(0), transform);
                }
                catch (...) {}
                double cx = transform.getOrigin().x();
                double cy = transform.getOrigin().y();
                double cyaw = tf::getYaw(transform.getRotation());
                double step = forward_sim_step_;
                double accum = 0.0;
                double max_cost_along_path = 0.0;

                for (double t = step; t <= forward_sim_time_; t += step)
                {
                    if (std::fabs(desired_vel.angular.z) > 1e-6)
                    {
                        double r = desired_vel.linear.x / desired_vel.angular.z;
                        cx += r * (sin(cyaw + desired_vel.angular.z * step) - sin(cyaw));
                        cy -= r * (cos(cyaw + desired_vel.angular.z * step) - cos(cyaw));
                        cyaw += desired_vel.angular.z * step;
                    }
                    else
                    {
                        cx += desired_vel.linear.x * step * cos(cyaw);
                        cy += desired_vel.linear.x * step * sin(cyaw);
                    }
                    accum += std::fabs(desired_vel.linear.x) * step;

                    unsigned int mx, my;
                    if (costmap->worldToMap(cx, cy, mx, my))
                    {
                        unsigned char c = costmap->getCost(mx, my);
                        if (c > max_cost_along_path) max_cost_along_path = c;
                        // 代价 >= 200 表示靠近障碍物（膨胀区内层）
                        if (c >= 200 && accum < clear_dist)
                            clear_dist = accum;
                    }
                }
            }
        }

        if (enable_graded_speed_ && clear_dist < forward_sim_time_)
        {
            if (clear_dist < danger_dist_near_)
            {
                // === 危险区 (< 0.18m): 停车 + 触发智能恢复 ===
                ROS_WARN_THROTTLE(0.5,
                    "IdenPlannerV2: ⛔ 危险! 安全距离=%.2fm → 停车",
                    clear_dist);
                desired_vel.linear.x  = 0.0;
                desired_vel.angular.z = 0.0;

                if (enable_smart_recovery_ && !in_recovery_)
                {
                    in_recovery_ = true;
                    recovery_backup_phase_ = false;
                    recovery_yaw_initialized_ = false;
                    recovery_yaw_accum_ = 0.0;
                }
            }
            else if (clear_dist < danger_dist_mid_)
            {
                // === 谨慎区 (0.18~0.40m): 大幅减速 + 朝开阔侧偏转 ===
                double ratio = speed_ratio_near_;
                // 找左右哪边更开阔, 偏转过去
                double open_bias = findSafeRotationDirection();
                double biased_wz = desired_vel.angular.z;
                if (std::fabs(open_bias) > 0.01)
                    biased_wz = desired_vel.angular.z * 0.3 + open_bias * 0.7;

                ROS_WARN_THROTTLE(0.5,
                    "IdenPlannerV2: ↗ 谨慎 安全距离=%.2fm 减速到%.0f%% 偏向=%+.2f",
                    clear_dist, ratio * 100, biased_wz);

                desired_vel.linear.x  *= ratio;
                desired_vel.angular.z  = biased_wz;
            }
            else if (clear_dist < danger_dist_far_)
            {
                // === 注意区 (0.40~0.80m): 适度减速 ===
                double t = (clear_dist - danger_dist_mid_) /
                           (danger_dist_far_ - danger_dist_mid_);
                double ratio = speed_ratio_mid_ + (1.0 - speed_ratio_mid_) * t;

                ROS_INFO_THROTTLE(1.0,
                    "IdenPlannerV2: → 注意 安全距离=%.2fm 减速到%.0f%%",
                    clear_dist, ratio * 100);

                desired_vel.linear.x  *= ratio;
                desired_vel.angular.z *= ratio;
            }
            // else: clear_dist >= danger_dist_far_ → 正常速度, 不改动
        }
        else if (will_collide && !enable_graded_speed_)
        {
            // 回退: 旧版简单碰撞响应 (enable_graded_speed_=false 时)
            if (collision_dist < 0.15)
            {
                desired_vel.linear.x  = 0.0;
                desired_vel.angular.z = 0.0;
                if (enable_smart_recovery_ && !in_recovery_)
                {
                    in_recovery_ = true;
                    recovery_backup_phase_ = false;
                    recovery_yaw_initialized_ = false;
                    recovery_yaw_accum_ = 0.0;
                }
            }
            else
            {
                double ratio = collision_dist / 0.5;
                desired_vel.linear.x  *= ratio;
                desired_vel.angular.z *= ratio;
            }
        }
    }

    // ========================================================
    //  速度平滑 (加/减速限制)
    // ========================================================
    cmd_vel = applyVelocitySmoothing(desired_vel, last_cmd_vel_);

    // ---- 保存当前指令用于下一帧平滑 ----
    last_cmd_vel_ = cmd_vel;

    // ---- 路径可视化 ----
    if (enable_visualization_)
    {
        cv::Mat plan_img(600, 600, CV_8UC3, cv::Scalar(0, 0, 0));
        for (size_t i = 0; i < global_plan_.size(); i++)
        {
            geometry_msgs::PoseStamped pb;
            global_plan_[i].header.stamp = ros::Time(0);
            try { tf_listener_->transformPose("base_link", global_plan_[i], pb); }
            catch (...) { continue; }
            int px = 300 - pb.pose.position.y * 100;
            int py = 300 - pb.pose.position.x * 100;
            cv::circle(plan_img, cv::Point(px, py), 1, cv::Scalar(255, 0, 255));
        }
        {
            int tx = 300 - target.pose.position.y * 100;
            int ty = 300 - target.pose.position.x * 100;
            cv::circle(plan_img, cv::Point(tx, ty), 5, cv::Scalar(0, 255, 255), -1);
        }
        cv::circle(plan_img, cv::Point(300, 300), 15, cv::Scalar(0, 255, 0));
        cv::line(plan_img, cv::Point(65, 300), cv::Point(510, 300), cv::Scalar(0, 255, 0), 1);
        cv::line(plan_img, cv::Point(300, 45), cv::Point(300, 555), cv::Scalar(0, 255, 0), 1);
        cv::putText(plan_img,
                    reverse_mode_ ? "REVERSE" : "FORWARD",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, reverse_mode_ ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0), 2);
        cv::imshow("IdenPlannerV2 Path", plan_img);
        cv::waitKey(1);
    }

    return true;
}

// ============================================================
//  computeVelocityCommands — 主控制循环
// ============================================================

bool IdenPlannerV2::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
    // ---- 每秒热重载参数 ----
    param_reload_cnt_++;
    if (param_reload_cnt_ >= 10)
    {
        param_reload_cnt_ = 0;
        reloadParams();
    }

    if (global_plan_.empty())
    {
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        in_recovery_ = false;
        recovery_backup_phase_ = false;
        return true;
    }

    // ---- 代价地图可视化 ----
    if (enable_visualization_ || enable_costmap_pub_)
        renderCostmap();

    // ========================================================
    //  阶段0: 如果处于恢复模式, 执行恢复旋转
    // ========================================================
    if (in_recovery_)
    {
        bool recovery_ok = computeRecoveryPhase(cmd_vel);

        // 速度平滑
        cmd_vel = applyVelocitySmoothing(cmd_vel, last_cmd_vel_);
        last_cmd_vel_ = cmd_vel;

        if (!recovery_ok)
        {
            // 恢复完成或需要重规划
            in_recovery_ = false;
            recovery_backup_phase_ = false;
            recovery_yaw_initialized_ = false;
        }
        return true;
    }

    // ========================================================
    //  阶段1: 位姿最终调整
    // ========================================================
    if (computePoseAdjustPhase(cmd_vel))
    {
        cmd_vel = applyVelocitySmoothing(cmd_vel, last_cmd_vel_);
        last_cmd_vel_ = cmd_vel;
        return true;
    }

    // ========================================================
    //  阶段2: 路径追踪 (含轨迹采样+前向碰撞检测)
    // ========================================================
    return computeTrackingPhase(cmd_vel);
}

}  // namespace iden_planner_v2