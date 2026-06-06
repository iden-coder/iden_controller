#include <iden_controller/iden_planner.h>
#include <pluginlib/class_list_macros.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <tf/tf.h>
#include <tf/transform_datatypes.h>
#include <cmath>

PLUGINLIB_EXPORT_CLASS(iden_planner::IdenPlanner, nav_core::BaseLocalPlanner)

namespace iden_planner
{

// ============================================================
//  构造 / 析构
// ============================================================

IdenPlanner::IdenPlanner()
{
    setlocale(LC_ALL, "");
}

IdenPlanner::~IdenPlanner()
{
    if (tf_listener_)
        delete tf_listener_;
}

// ============================================================
//  initialize — 加载全部参数
// ============================================================

void IdenPlanner::initialize(std::string name, tf2_ros::Buffer* tf,
                             costmap_2d::Costmap2DROS* costmap_ros)
{
    tf_listener_ = new tf::TransformListener();
    costmap_ros_ = costmap_ros;

    param_ns_ = "~/" + name;  // 存储命名空间供热重载使用
    ros::NodeHandle nh(param_ns_);

    // PID
    nh.param("kp",                  kp_,                  3.0);
    nh.param("ki",                  ki_,                  0.02);
    nh.param("kd",                  kd_,                  0.02);
    nh.param("integral_limit",      integral_limit_,      0.5);

    // 预瞄
    nh.param("lookahead_dist",      lookahead_dist_,      0.25);

    // 角度-速度耦合
    nh.param("angle_power",         angle_power_,         1.5);
    nh.param("angle_full_speed_deg", angle_full_speed_deg_, 0.0);
    nh.param("angle_zero_speed_deg", angle_zero_speed_deg_, 90.0);

    // 双向导航
    nh.param("enable_bidirectional", enable_bidirectional_, true);
    nh.param("long_lookahead",      long_lookahead_,      0.5);
    nh.param("reverse_vote_ratio",  reverse_vote_ratio_,  0.7);
    nh.param("reverse_confirm_frames", reverse_confirm_frames_, 3);

    // 速度限制
    nh.param("max_linear_vel",      max_linear_vel_,      0.5);
    nh.param("min_linear_vel",      min_linear_vel_,     -0.5);
    nh.param("max_angular_vel",     max_angular_vel_,     1.0);
    nh.param("linear_gain",         linear_gain_,         3.0);

    // 碰撞检测
    nh.param("collision_check_count",   collision_check_count_,   10);
    nh.param("collision_cooldown_max",  collision_cooldown_max_,  20);

    // 可视化
    nh.param("enable_visualization",  enable_visualization_, false);
    nh.param("enable_costmap_pub",    enable_costmap_pub_,   false);

    // 位姿调整
    nh.param("pose_dist_threshold", pose_dist_threshold_, 0.1);
    nh.param("pos_deadband",        pos_deadband_,        0.015);
    nh.param("yaw_deadband",        yaw_deadband_,        0.04);
    nh.param("angle_adjust_gain",   angle_adjust_gain_,   1.0);
    nh.param("slow_zone",           slow_zone_,           0.2);
    nh.param("min_adjust_speed",    min_adjust_speed_,    0.08);
    nh.param("pose_tolerance",      pose_tolerance_,      0.015);
    nh.param("pose_adjust_timeout", pose_adjust_timeout_, 15.0);
    nh.param("pose_adjust_max_linear", pose_adjust_max_linear_, 0.10);

    // 状态初始化
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
    collision_cooldown_max_ = 20;  // ~2秒 @10Hz，防止频繁触发重规划
    collision_replan_count_ = 0;
    collision_replan_max_   = 2;   // 最多触发 2 次重规划，之后仅减速

    // 代价地图可视化发布
    if (enable_costmap_pub_)
    {
        ros::NodeHandle nh_pub;  // 全局命名空间发布 topic
        costmap_pub_ = nh_pub.advertise<sensor_msgs::Image>("/iden_planner/costmap", 1);
    }

    ROS_WARN("IdenPlanner 启动! 预瞄=%.2fm, Kp=%.1f, 双向=%s, 可视化=%s, costmap_topic=%s",
             lookahead_dist_, kp_,
             enable_bidirectional_ ? "ON" : "OFF",
             enable_visualization_ ? "ON" : "OFF",
             enable_costmap_pub_ ? "ON" : "OFF");
}

// ============================================================
//  setPlan — 接收新全局路径
// ============================================================

bool IdenPlanner::setPlan(const std::vector<geometry_msgs::PoseStamped>& plan)
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
    collision_replan_count_ = 0;  // 新路径到来，重置碰撞重规划计数
    return true;
}

// ============================================================
//  isGoalReached
// ============================================================

bool IdenPlanner::isGoalReached()
{
    return goal_reached_;
}

// ============================================================
//  参数热重载（每秒一次，配合 rosparam set 运行时调参）
// ============================================================

void IdenPlanner::reloadParams()
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
    nh.getParam("collision_replan_max", collision_replan_max_);
}

// ============================================================
//  【核心1】角度-速度耦合：角度偏差大 → 线速度降低
//  使用幂函数曲线: scale = 1 - t^N
//  t ∈ [0,1] 是归一化角度误差
// ============================================================

double IdenPlanner::computeAngleSpeedScale(double angle_error_rad)
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
//  【核心2】角速度 PID（带积分抗饱和 + 航点切换重置）
// ============================================================

double IdenPlanner::computeAngularPID(double lateral_error)
{
    error_sum_ += lateral_error;

    // 积分抗饱和
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
//  【核心3】选择预瞄路径点（基于 base_link 坐标系）
// ============================================================

geometry_msgs::PoseStamped IdenPlanner::selectLookaheadTarget()
{
    geometry_msgs::PoseStamped target;
    target.pose.orientation.w = 1.0; // 安全默认值

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
//  【核心4】双向导航：分析路径方向，决定前进/倒车
//  长预瞄 (long_lookahead_) 内统计路径点分布
// ============================================================

bool IdenPlanner::analyzePathDirection()
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

    if (total < 3) return reverse_mode_;  // 数据不足，保持当前方向

    double behind_ratio = (double)behind / total;
    double ahead_ratio  = (double)ahead  / total;

    if (behind_ratio > reverse_vote_ratio_) return true;   // 应倒车
    if (ahead_ratio  > reverse_vote_ratio_) return false;  // 应前进
    return reverse_mode_;  // 不确定 → 保持
}

void IdenPlanner::updateReverseState(bool want_reverse)
{
    if (want_reverse != reverse_mode_)
    {
        reverse_confirm_cnt_++;
        if (reverse_confirm_cnt_ >= reverse_confirm_frames_)
        {
            reverse_mode_ = want_reverse;
            reverse_confirm_cnt_ = 0;
            // 切方向时重置 PID 状态
            error_sum_  = 0.0;
            last_error_ = 0.0;
            ROS_WARN("IdenPlanner: 切换为 %s 模式", reverse_mode_ ? "倒车" : "前进");
        }
    }
    else
    {
        reverse_confirm_cnt_ = 0;
    }
}

// ============================================================
//  【核心5】位姿最终调整
// ============================================================

bool IdenPlanner::computePoseAdjust(geometry_msgs::Twist& cmd_vel,
                                    const geometry_msgs::PoseStamped& pose_final)
{
    double dx       = pose_final.pose.position.x;
    double dy       = pose_final.pose.position.y;
    double final_yaw = tf::getYaw(pose_final.pose.orientation);

    double desired_linear  = 0.0;
    double desired_angular = 0.0;

    // 位置死区：允许双向微调
    if (std::fabs(dx) > pos_deadband_)
        desired_linear = dx * 0.8;  // 低增益微调
    desired_linear = std::max(-pose_adjust_max_linear_,
                      std::min(pose_adjust_max_linear_, desired_linear));

    // 角度死区：渐变降速
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

    // 收敛判断
    if (std::fabs(final_yaw) < pose_tolerance_ &&
        std::fabs(dx) < pos_deadband_)
    {
        goal_reached_ = true;
        ROS_WARN("IdenPlanner: 精确到达目标点!");
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        return true;
    }

    cmd_vel.linear.x  = desired_linear;
    cmd_vel.angular.z = desired_angular;
    return true;
}

// ============================================================
//  代价地图可视化（OpenCV 窗口，用于调试）
// ============================================================

void IdenPlanner::renderCostmap()
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
            if      (c == 0)   p = cv::Vec3b(128, 128, 128);  // 自由
            else if (c == 254) p = cv::Vec3b(0, 0, 0);        // 障碍
            else               p = cv::Vec3b(255 - c, 0, c);  // 膨胀
        }
    }

    cv::circle(img, cv::Point(sx / 2, sy / 2), 5, cv::Scalar(0, 255, 0), -1);

    // 本地 OpenCV 窗口（仅在 enable_visualization_ 时弹出）
    if (enable_visualization_)
    {
        cv::imshow("IdenPlanner Costmap", img);
        cv::waitKey(1);
    }

    // cv_bridge 发布（仅在 enable_costmap_pub_ 时发布到 ROS topic）
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

// ============================================================
//  publishCostmap — 独立的代价地图 topic 发布（供外部调用）
// ============================================================

void IdenPlanner::publishCostmap()
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
//  computeVelocityCommands — 主控制循环
// ============================================================

bool IdenPlanner::computeVelocityCommands(geometry_msgs::Twist& cmd_vel)
{
    // ---- 每秒热重载参数 ----
    param_reload_cnt_++;
    if (param_reload_cnt_ >= 10)  // ~10Hz 调用频率 → 1秒一次
    {
        param_reload_cnt_ = 0;
        reloadParams();
    }

    if (global_plan_.empty())
    {
        cmd_vel.linear.x  = 0.0;
        cmd_vel.angular.z = 0.0;
        return true;
    }

    // ---- 代价地图可视化（仅在开启可视化或 topic 发布时执行）----
    if (enable_visualization_ || enable_costmap_pub_)
        renderCostmap();

    // ========================================================
    //  阶段1：判断是否进入最终位姿调整
    // ========================================================
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
            pose_adjust_start_ = ros::Time::now();  // 记录进入时刻
        }
    }

    if (pose_adjusting_)
    {
        // 超时保护：超过 pose_adjust_timeout_ 秒仍未收敛，直接标记到达
        double adjust_elapsed = (ros::Time::now() - pose_adjust_start_).toSec();
        if (adjust_elapsed > pose_adjust_timeout_)
        {
            goal_reached_ = true;
            ROS_WARN("IdenPlanner: 位姿调整超时(%.1fs)，强制标记到达", adjust_elapsed);
            cmd_vel.linear.x  = 0.0;
            cmd_vel.angular.z = 0.0;
            return true;
        }

        // 实时检查：如果又被推远了，退出位姿调整
        double dist = std::hypot(pose_final.pose.position.x,
                                 pose_final.pose.position.y);
        if (dist > pose_dist_threshold_ + 0.05)
        {
            pose_adjusting_ = false;
            error_sum_  = 0.0;  // 重置 PID 积分，防止退出时角速度跳变
            last_error_ = 0.0;
            ROS_WARN("IdenPlanner: 退出位姿调整，距离=%.3f", dist);
        }
        else
        {
            return computePoseAdjust(cmd_vel, pose_final);
        }
    }

    // ========================================================
    //  阶段2：路径追踪
    // ========================================================

    // 2a. 双向导航分析
    if (enable_bidirectional_)
    {
        bool want_reverse = analyzePathDirection();
        updateReverseState(want_reverse);
    }

    // 2b. 选择预瞄点
    geometry_msgs::PoseStamped target = selectLookaheadTarget();

    // 2b2. 两级碰撞检测：
    //       致命区 (cost>=254 LETHAL): 停车+重规划(限2次)，超过→减速到30%
    //       危险区 (cost>=253 INSCRIBED): 仅减速到40%，不触发重规划，避免死锁
    double collision_speed_factor = 1.0;  // 1.0=全速, 0.4=危险区, 0.3=致命区超限
    {
        costmap_2d::Costmap2D* costmap = costmap_ros_ ? costmap_ros_->getCostmap() : nullptr;
        if (costmap)
        {
        double resolution  = costmap->getResolution();
        double origin_x    = costmap->getOriginX();
        double origin_y    = costmap->getOriginY();
        unsigned int size_x = costmap->getSizeInCellsX();
        unsigned int size_y = costmap->getSizeInCellsY();
        unsigned char* costmap_data = costmap->getCharMap();
        std::string global_frame = costmap_ros_->getGlobalFrameID();

        // 冷却递减
        if (collision_cooldown_ > 0)
            collision_cooldown_--;

        int check_end = std::min((int)global_plan_.size(),
                                 target_index_ + collision_check_count_);
        bool collision_lethal = false;
        bool collision_danger = false;
        for (int i = target_index_; i < check_end; i++)
        {
            geometry_msgs::PoseStamped pose_global;
            global_plan_[i].header.stamp = ros::Time(0);
            try
            {
                tf_listener_->transformPose(global_frame, global_plan_[i], pose_global);
            }
            catch (...) { continue; }

            double wx = pose_global.pose.position.x;
            double wy = pose_global.pose.position.y;
            unsigned int mx = (unsigned int)((wx - origin_x) / resolution);
            unsigned int my = (unsigned int)((wy - origin_y) / resolution);

            if (mx < size_x && my < size_y)
            {
                unsigned char cost = costmap_data[my * size_x + mx];
                // 致命区：LETHAL_OBSTACLE (254) — 优先处理
                if (cost >= 254)
                {
                    collision_lethal = true;
                    break;
                }
                // 危险区：INSCRIBED_INFLATED (253) — 减速但不重规划
                if (cost >= 253)
                {
                    collision_danger = true;
                }
            }
        }

        if (collision_lethal)
        {
            // 致命区：冷却期内不处理
            if (collision_cooldown_ > 0)
            {
                // 冷却中，继续正常行驶
            }
            else if (collision_replan_count_ < collision_replan_max_)
            {
                // 前 N 次：停车 + 触发重规划
                ROS_WARN("IdenPlanner: 致命障碍物! 触发重规划 (%d/%d)",
                         collision_replan_count_ + 1, collision_replan_max_);
                cmd_vel.linear.x  = 0.0;
                cmd_vel.linear.y  = 0.0;
                cmd_vel.angular.z = 0.0;
                collision_cooldown_ = collision_cooldown_max_;
                collision_replan_count_++;
                return false;  // 通知 move_base 重新全局规划
            }
            else
            {
                // 重规划次数用尽：减速到 30%
                ROS_WARN_THROTTLE(1.0,
                    "IdenPlanner: 致命区重规划用尽，减速通过 (replan=%d)",
                    collision_replan_count_);
                collision_speed_factor = 0.3;
            }
        }
        else if (collision_danger)
        {
            // 危险区 (INSCRIBED_INFLATED): 减速到 40%，不触发重规划
            ROS_WARN_THROTTLE(1.0,
                "IdenPlanner: 进入障碍物膨胀区，减速缓行");
            collision_speed_factor = 0.4;
        }
        else
        {
            // 路径前方安全，重置重规划计数器
            if (collision_cooldown_ <= 0)
                collision_replan_count_ = 0;
        }
        } // if (costmap)
    }

    // 2c. 双向等效变换
    double eff_dx = target.pose.position.x;
    double eff_dy = target.pose.position.y;
    if (reverse_mode_)
    {
        eff_dx = -eff_dx;
        eff_dy = -eff_dy;
    }

    // 2d. 航点切换检测 → 重置积分
    if (target_index_ != prev_target_index_)
    {
        error_sum_  = 0.0;
        last_error_ = 0.0;
        prev_target_index_ = target_index_;
    }

    // 2e. 角度-速度耦合：计算线速度缩放
    double angle_err  = std::atan2(eff_dy, eff_dx);
    double speed_scale = computeAngleSpeedScale(angle_err);
    double desired_linear = eff_dx * linear_gain_ * speed_scale;

    // 2f. 角速度 PID
    double desired_angular = computeAngularPID(eff_dy);

    // 2g. 方向符号
    if (reverse_mode_)
        desired_linear = -desired_linear;

    // 2h. 速度限幅
    if (desired_linear > max_linear_vel_)
        desired_linear = max_linear_vel_;
    else if (desired_linear < min_linear_vel_)
        desired_linear = min_linear_vel_;

    if (desired_angular > max_angular_vel_)
        desired_angular = max_angular_vel_;
    else if (desired_angular < -max_angular_vel_)
        desired_angular = -max_angular_vel_;

    cmd_vel.linear.x  = desired_linear;
    cmd_vel.angular.z = desired_angular;

    // 碰撞减速：致命区超限→30%, 危险区→40%
    if (collision_speed_factor < 1.0)
    {
        cmd_vel.linear.x  *= collision_speed_factor;
        cmd_vel.angular.z *= collision_speed_factor;
    }

    // 2i. 路径可视化（仅在 enable_visualization_ 时弹出 OpenCV 窗口）
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
        // 预瞄目标点（黄色大点）
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
        cv::imshow("IdenPlanner Path", plan_img);
        cv::waitKey(1);
    }

    return true;
}

}  // namespace iden_planner
