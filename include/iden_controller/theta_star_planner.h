#ifndef THETA_STAR_PLANNER_ROS_H_
#define THETA_STAR_PLANNER_ROS_H_

#include <ros/ros.h>
#include <nav_core/base_global_planner.h>
#include <costmap_2d/costmap_2d_ros.h>
#include <costmap_2d/cost_values.h>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>

namespace theta_star_ros
{

// ===== 数据结构 (同 navigation2 Theta*) =====
struct Coords { double x, y; };

struct TreeNode
{
    int x, y;
    double g = std::numeric_limits<double>::max();
    double h = std::numeric_limits<double>::max();
    const TreeNode* parent = nullptr;
    bool in_queue = false;
    double f = std::numeric_limits<double>::max();
};

struct NodeCompare
{
    bool operator()(const TreeNode* a, const TreeNode* b) { return a->f > b->f; }
};

// ===== Theta* 核心算法 =====
class ThetaStarCore
{
public:
    costmap_2d::Costmap2D* costmap_;
    int size_x_, size_y_;
    Coords src_, dst_;
    int nodes_opened_;

    double w_euc_cost_;       // 欧几里得距离权重 (default 1.0)
    double w_traversal_cost_; // 遍历代价权重 (default 0.1)
    double w_heuristic_cost_; // 启发式权重 (default 1.0)
    bool   allow_unknown_;    // 允许穿越未知区域 (default true)
    int    max_non_obstacle_; // 最大非障碍代价 (default 252)

    ThetaStarCore();

    bool generatePath(std::vector<Coords>& raw_path);

private:
    std::vector<TreeNode*> node_positions_;
    std::vector<TreeNode>  node_data_;
    int index_generated_;
    std::priority_queue<TreeNode*, std::vector<TreeNode*>, NodeCompare> queue_;

    static const int moves_[8][2];

    void resetContainers();
    void initializePositions(int size_inc = 0);
    void addToNodeData(int id);
    void addIndex(int cx, int cy, TreeNode* n);
    TreeNode* getIndex(int cx, int cy);
    bool withinLimits(int cx, int cy) const;
    bool isGoal(const TreeNode& n) const;
    double getTraversalCost(int cx, int cy);
    double getEuclideanCost(int ax, int ay, int bx, int by);
    double getHCost(int cx, int cy);
    bool isSafe(int cx, int cy) const;
    bool isSafe(int cx, int cy, double& cost) const;
    bool losCheck(int x0, int y0, int x1, int y1, double& sl_cost) const;
    void resetParent(TreeNode* curr);
    void setNeighbors(const TreeNode* curr);
    void backtrace(std::vector<Coords>& raw, const TreeNode* curr) const;
    void clearQueue();
};

// ===== ROS1 全局规划器插件 =====
class ThetaStarPlanner : public nav_core::BaseGlobalPlanner
{
public:
    ThetaStarPlanner();
    ThetaStarPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros);

    void initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros) override;
    bool makePlan(const geometry_msgs::PoseStamped& start,
                  const geometry_msgs::PoseStamped& goal,
                  std::vector<geometry_msgs::PoseStamped>& plan) override;

private:
    costmap_2d::Costmap2DROS* costmap_ros_;
    ThetaStarCore core_;
    bool initialized_;
};

}  // namespace theta_star_ros
#endif
