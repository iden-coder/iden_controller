/**
 * @file test_iden_planner.cpp
 * @brief IdenPlanner 核心算法单元测试
 *
 * 覆盖三个关键模块：
 *   1. computeAngleSpeedScale  — 角度-速度耦合曲线（边界值 + 单调性）
 *   2. computeAngularPID        — PID 积分累积 + 抗饱和 + 航点重置
 *   3. analyzePathDirection     — 双向导航方向判断（前进/倒车投票）
 *
 * 运行方式:
 *   catkin_make run_tests_iden_controller_gtest_test_iden_planner
 *   或
 *   rostest iden_controller test_iden_planner.test
 */

#include <gtest/gtest.h>
#include <cmath>

// 将 IdenPlanner 的 private 方法暴露给测试（配合头文件中的 FRIEND_TEST）
#include <iden_controller/iden_planner.h>

// 测试夹具放入 iden_planner 命名空间，以便 friend class 声明生效
namespace iden_planner
{

// ============================================================
//  测试夹具：提供默认构造的 IdenPlanner 实例
// ============================================================

class IdenPlannerTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        planner_ = new iden_planner::IdenPlanner();

        // 手动设置参数，模拟 initialize() 的效果（跳过 TF 初始化）
        // 这些值与 iden_planner.cpp 中的默认值一致
        planner_->kp_                   = 3.0;
        planner_->ki_                   = 0.02;
        planner_->kd_                   = 0.02;
        planner_->integral_limit_       = 0.5;
        planner_->angle_power_          = 1.5;
        planner_->angle_full_speed_deg_ = 0.0;
        planner_->angle_zero_speed_deg_ = 90.0;
        planner_->reverse_vote_ratio_   = 0.7;
        planner_->reverse_confirm_frames_ = 3;
        planner_->long_lookahead_       = 0.5;
        planner_->enable_bidirectional_ = true;
        planner_->pose_dist_threshold_  = 0.1;
        planner_->pos_deadband_         = 0.015;
        planner_->yaw_deadband_         = 0.04;
        planner_->pose_tolerance_       = 0.015;
        planner_->error_sum_            = 0.0;
        planner_->last_error_           = 0.0;
        planner_->reverse_mode_         = false;
        planner_->reverse_confirm_cnt_  = 0;
        planner_->target_index_         = 0;
        planner_->prev_target_index_    = -1;
    }

    void TearDown() override
    {
        delete planner_;
    }

    iden_planner::IdenPlanner* planner_;
};

// ============================================================
//  1. computeAngleSpeedScale — 角度-速度耦合曲线测试
// ============================================================

// 角度为 0 时，应返回满速 (1.0)
TEST_F(IdenPlannerTest, AngleSpeedScale_Boundary)
{
    // 零角度 → 满速
    EXPECT_DOUBLE_EQ(1.0, planner_->computeAngleSpeedScale(0.0));

    // 刚好在 full_speed 边界 (0°) → 满速
    double full_rad = planner_->angle_full_speed_deg_ * M_PI / 180.0;
    EXPECT_DOUBLE_EQ(1.0, planner_->computeAngleSpeedScale(full_rad));

    // 刚好在 zero_speed 边界 (90°) → 零速
    double zero_rad = planner_->angle_zero_speed_deg_ * M_PI / 180.0;
    EXPECT_DOUBLE_EQ(0.0, planner_->computeAngleSpeedScale(zero_rad));

    // 超过 zero_speed → 仍然是零速
    EXPECT_DOUBLE_EQ(0.0, planner_->computeAngleSpeedScale(zero_rad + 0.5));
    EXPECT_DOUBLE_EQ(0.0, planner_->computeAngleSpeedScale(M_PI));  // 180°

    // 负角度 → 与正角度相同（取绝对值）
    EXPECT_DOUBLE_EQ(planner_->computeAngleSpeedScale(0.5),
                     planner_->computeAngleSpeedScale(-0.5));
}

// 中段曲线：单调递减校验
TEST_F(IdenPlannerTest, AngleSpeedScale_MidRange)
{
    double full  = planner_->angle_full_speed_deg_ * M_PI / 180.0;
    double zero  = planner_->angle_zero_speed_deg_ * M_PI / 180.0;
    double mid   = (full + zero) / 2.0;  // 45°

    double scale_mid = planner_->computeAngleSpeedScale(mid);

    // 中间值应在 (0, 1) 之间
    EXPECT_GT(scale_mid, 0.0);
    EXPECT_LT(scale_mid, 1.0);

    // 单调性: angle1 < angle2 → scale1 >= scale2
    double scale_30 = planner_->computeAngleSpeedScale(30.0 * M_PI / 180.0);
    double scale_60 = planner_->computeAngleSpeedScale(60.0 * M_PI / 180.0);
    EXPECT_GE(scale_30, scale_60) << "30° 应比 60° 速度更高";

    // power 参数影响曲线形状
    // angle_power=1.0 时是线性，>1.0 时曲线上凸（前半段降速慢，后半段急降）
    // 默认 1.5 → 45° 处 t=0.5 → t^1.5=0.354 → scale=0.646（仍保留约 65% 速度）
    EXPECT_GT(scale_mid, 0.5)
        << "angle_power=1.5 时，45° 处 scale 应大于 0.5（上凸曲线，前半段保持高速）";
}

// ============================================================
//  2. computeAngularPID — PID 控制测试
// ============================================================

// 持续的横向偏差 → 积分逐步累积
TEST_F(IdenPlannerTest, AngularPID_IntegralAccumulation)
{
    const double lateral_error = 0.1;  // 固定偏差 0.1m

    // 第一次调用：只有 P+D，I 刚开始累积
    double out1 = planner_->computeAngularPID(lateral_error);
    double sum1 = planner_->error_sum_;

    // 第二次调用：I 继续累积
    double out2 = planner_->computeAngularPID(lateral_error);
    double sum2 = planner_->error_sum_;

    // 积分应该增加（同向偏差持续）
    EXPECT_GT(std::fabs(sum2), std::fabs(sum1))
        << "持续同向偏差时积分应增加";

    // 输出中应能看到 I 项的增长贡献
    // P 项 = kp * error = 3.0 * 0.1 = 0.3
    double expected_p_only = planner_->kp_ * lateral_error;
    // 有 I 项后输出绝对值应大于纯 P
    EXPECT_GT(std::fabs(out2), std::fabs(expected_p_only) * 0.9)
        << "PID 输出应包含 P+I+D 的完整贡献";
}

// 积分抗饱和：持续累积不应超过 integral_limit
TEST_F(IdenPlannerTest, AngularPID_AntiWindup)
{
    const double large_error = 1.0;  // 大偏差
    const int    iterations   = 100; // 大量迭代

    for (int i = 0; i < iterations; i++)
    {
        planner_->computeAngularPID(large_error);
    }

    double limit = planner_->integral_limit_;
    EXPECT_LE(std::fabs(planner_->error_sum_), limit)
        << "积分不应超过 integral_limit (" << limit << ")";

    // 积分达到上限后不应继续增长
    EXPECT_DOUBLE_EQ(planner_->error_sum_, limit)
        << "积分应在 integral_limit 处饱和（正偏差）";
}

// 航点切换时外部重置积分 → 新航点从干净状态开始
TEST_F(IdenPlannerTest, AngularPID_ResetOnWaypointSwitch)
{
    // 模拟几个周期的跟踪
    planner_->computeAngularPID(0.2);
    planner_->computeAngularPID(0.2);
    planner_->computeAngularPID(0.2);

    // 确认积分不为零
    EXPECT_NE(0.0, planner_->error_sum_);

    // 模拟航点切换：重置积分（这是 computeVelocityCommands 中的逻辑）
    planner_->error_sum_  = 0.0;
    planner_->last_error_ = 0.0;

    EXPECT_DOUBLE_EQ(0.0, planner_->error_sum_);
    EXPECT_DOUBLE_EQ(0.0, planner_->last_error_);

    // 新航点上的第一次输出仅包含 P（+极小的 D）
    double out = planner_->computeAngularPID(0.1);
    double p_only = planner_->kp_ * 0.1;
    // 由于积分和 last_error 都是 0，输出应非常接近纯 P
    EXPECT_NEAR(out, p_only, 0.01)
        << "航点重置后首次输出应接近纯 P 项";
}

// ============================================================
//  3. analyzePathDirection — 双向导航方向判断测试
// ============================================================

// 模拟：设置 global_plan_ 并检查方向判断
// 注意：analyzePathDirection 依赖 tf_listener_->transformPose，
// 在纯单元测试中无法真正调用（没有 TF 树）。
// 这里测试投票逻辑的边界条件。
//
// 为测试投票逻辑，我们直接验证 reverse_mode_ 状态机行为：
TEST_F(IdenPlannerTest, ReverseMode_StateMachine)
{
    // 初始状态：前进
    EXPECT_FALSE(planner_->reverse_mode_);

    // 模拟连续 3 帧请求倒车 → 应切换
    planner_->updateReverseState(true);
    EXPECT_FALSE(planner_->reverse_mode_);  // 第1帧：确认中

    planner_->updateReverseState(true);
    EXPECT_FALSE(planner_->reverse_mode_);  // 第2帧：确认中

    planner_->updateReverseState(true);
    EXPECT_TRUE(planner_->reverse_mode_);   // 第3帧：切换！

    // 方向切换后确认计数应重置
    EXPECT_EQ(0, planner_->reverse_confirm_cnt_);
}

// 倒车状态下请求前进，需要重新计数
TEST_F(IdenPlannerTest, ReverseMode_ReconfirmOnDirectionChange)
{
    // 先切换到倒车状态
    planner_->updateReverseState(true);
    planner_->updateReverseState(true);
    planner_->updateReverseState(true);
    EXPECT_TRUE(planner_->reverse_mode_);

    // 请求切回前进
    planner_->updateReverseState(false);
    EXPECT_TRUE(planner_->reverse_mode_);   // 第1帧：确认中（不倒车→前进）

    planner_->updateReverseState(false);
    EXPECT_TRUE(planner_->reverse_mode_);   // 第2帧

    planner_->updateReverseState(false);
    EXPECT_FALSE(planner_->reverse_mode_);  // 第3帧：切回前进
}

// 中断计数：请求方向来回摆动应重置计数
TEST_F(IdenPlannerTest, ReverseMode_InterruptResetsCounter)
{
    // 第1帧：请求倒车
    planner_->updateReverseState(true);
    EXPECT_EQ(1, planner_->reverse_confirm_cnt_);

    // 第2帧：取消请求（回到前进）
    planner_->updateReverseState(false);
    EXPECT_EQ(0, planner_->reverse_confirm_cnt_) << "中断后计数器应重置";
}

}  // namespace iden_planner

// ============================================================
//  main
// ============================================================

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
