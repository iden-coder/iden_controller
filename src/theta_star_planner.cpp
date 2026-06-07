#include <iden_controller/theta_star_planner.h>
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(theta_star_ros::ThetaStarPlanner, nav_core::BaseGlobalPlanner)

namespace theta_star_ros
{

// 8-connected neighbors: up, down, left, right, + 4 diagonals
const int ThetaStarCore::moves_[8][2] = {
    {0,1}, {0,-1}, {1,0}, {-1,0}, {1,-1}, {-1,1}, {1,1}, {-1,-1}
};

// ============================================================
//  ThetaStarCore — 纯算法
// ============================================================

ThetaStarCore::ThetaStarCore()
  : costmap_(nullptr), size_x_(0), size_y_(0), nodes_opened_(0),
    w_euc_cost_(1.0), w_traversal_cost_(0.1), w_heuristic_cost_(1.0),
    allow_unknown_(true), max_non_obstacle_(252), index_generated_(0)
{}

void ThetaStarCore::resetContainers()
{
    index_generated_ = 0;
    int new_sx = static_cast<int>(costmap_->getSizeInCellsX());
    int new_sy = static_cast<int>(costmap_->getSizeInCellsY());
    int total = new_sx * new_sy;

    if (static_cast<int>(node_positions_.size()) < total)
    {
        node_positions_.resize(total, nullptr);
        node_data_.clear();
        node_data_.reserve(total);
    }
    else
    {
        for (int i = 0; i < size_x_ * size_y_; i++)
            node_positions_[i] = nullptr;
    }
    size_x_ = new_sx;
    size_y_ = new_sy;
    clearQueue();
}

void ThetaStarCore::addToNodeData(int id)
{
    if (static_cast<int>(node_data_.size()) <= id)
        node_data_.push_back({});
    else
        node_data_[id] = {};
}

void ThetaStarCore::addIndex(int cx, int cy, TreeNode* n)
{
    node_positions_[size_x_ * cy + cx] = n;
}

TreeNode* ThetaStarCore::getIndex(int cx, int cy)
{
    return node_positions_[size_x_ * cy + cx];
}

bool ThetaStarCore::withinLimits(int cx, int cy) const
{
    return cx >= 0 && cx < size_x_ && cy >= 0 && cy < size_y_;
}

bool ThetaStarCore::isGoal(const TreeNode& n) const
{
    return n.x == dst_.x && n.y == dst_.y;
}

bool ThetaStarCore::isSafe(int cx, int cy) const
{
    unsigned char c = costmap_->getCost(cx, cy);
    if (c == costmap_2d::NO_INFORMATION && allow_unknown_) return true;
    return c <= static_cast<unsigned char>(max_non_obstacle_);
}

bool ThetaStarCore::isSafe(int cx, int cy, double& cost) const
{
    unsigned char c = costmap_->getCost(cx, cy);
    if ((c == costmap_2d::NO_INFORMATION && allow_unknown_) || c <= static_cast<unsigned char>(max_non_obstacle_))
    {
        double cc = (c == costmap_2d::NO_INFORMATION) ? 253.0 : static_cast<double>(c);
        cost += w_traversal_cost_ * cc * cc / (max_non_obstacle_ * max_non_obstacle_);
        return true;
    }
    return false;
}

double ThetaStarCore::getTraversalCost(int cx, int cy)
{
    double c = static_cast<double>(costmap_->getCost(cx, cy));
    return w_traversal_cost_ * c * c / (max_non_obstacle_ * max_non_obstacle_);
}

double ThetaStarCore::getEuclideanCost(int ax, int ay, int bx, int by)
{
    return w_euc_cost_ * std::hypot(ax - bx, ay - by);
}

double ThetaStarCore::getHCost(int cx, int cy)
{
    return w_heuristic_cost_ * std::hypot(cx - dst_.x, cy - dst_.y);
}

// Bresenham 直线检测: 从(x0,y0)到(x1,y1)之间是否全是安全的
bool ThetaStarCore::losCheck(int x0, int y0, int x1, int y1, double& sl_cost) const
{
    sl_cost = 0;
    int dx = abs(x1 - x0), sx = (x0 < x1) ? 1 : -1;
    int dy = abs(y1 - y0), sy = (y0 < y1) ? 1 : -1;
    int cx = x0, cy = y0, e = dx - dy;

    while (cx != x1 || cy != y1)
    {
        if (!isSafe(cx, cy, sl_cost)) return false;
        int e2 = 2 * e;
        if (e2 > -dy && e2 <= dx)
        {
            if (!isSafe(cx + sx, cy) || !isSafe(cx, cy + sy)) return false;
            cx += sx; cy += sy; e += dx - dy;
        }
        else if (e2 > -dy) { cx += sx; e -= dy; }
        else               { cy += sy; e += dx; }
    }
    return true;
}

// Theta* 核心: 检查当前节点能否直接连到祖父节点
void ThetaStarCore::resetParent(TreeNode* curr)
{
    curr->in_queue = false;
    const TreeNode* parent = curr->parent;
    const TreeNode* grandparent = parent->parent;
    double los_cost = 0;

    if (losCheck(curr->x, curr->y, grandparent->x, grandparent->y, los_cost))
    {
        double g = grandparent->g +
                   getEuclideanCost(curr->x, curr->y, grandparent->x, grandparent->y) + los_cost;
        if (g < curr->g)
        {
            curr->parent = grandparent;
            curr->g = g;
            curr->f = g + curr->h;
        }
    }
}

void ThetaStarCore::setNeighbors(const TreeNode* curr)
{
    for (int i = 0; i < 8; i++)
    {
        int mx = curr->x + moves_[i][0];
        int my = curr->y + moves_[i][1];

        if (!withinLimits(mx, my)) continue;
        if (!isSafe(mx, my)) continue;

        double g = curr->g + getEuclideanCost(curr->x, curr->y, mx, my) + getTraversalCost(mx, my);
        TreeNode* n = getIndex(mx, my);

        if (n == nullptr)
        {
            addToNodeData(index_generated_);
            n = &node_data_[index_generated_];
            addIndex(mx, my, n);
            index_generated_++;
        }

        double h = getHCost(mx, my);
        double f = g + h;
        if (n->f > f)
        {
            n->g = g; n->h = h; n->f = f;
            n->parent = curr;
            if (!n->in_queue)
            {
                n->x = mx; n->y = my;
                n->in_queue = true;
                queue_.push(n);
            }
        }
    }
}

bool ThetaStarCore::generatePath(std::vector<Coords>& raw_path)
{
    resetContainers();

    // 检查起点和终点安全
    if (!isSafe(src_.x, src_.y) || !isSafe(dst_.x, dst_.y))
    {
        ROS_WARN("Theta*: start or goal is unsafe");
        return false;
    }

    // 初始化起点
    addToNodeData(index_generated_);
    double sg = getTraversalCost(src_.x, src_.y);
    double sh = getHCost(src_.x, src_.y);
    TreeNode start_node;
    start_node.x = src_.x; start_node.y = src_.y;
    start_node.g = sg; start_node.h = sh;
    start_node.parent = &node_data_[index_generated_];
    start_node.in_queue = true;
    start_node.f = sg + sh;
    node_data_[index_generated_] = start_node;
    queue_.push(&node_data_[index_generated_]);
    addIndex(src_.x, src_.y, &node_data_[index_generated_]);
    TreeNode* curr = &node_data_[index_generated_];
    index_generated_++;
    nodes_opened_ = 0;

    while (!queue_.empty())
    {
        nodes_opened_++;
        if (isGoal(*curr)) break;

        resetParent(curr);
        setNeighbors(curr);

        curr = queue_.top();
        queue_.pop();
    }

    if (queue_.empty() && !isGoal(*curr))
    {
        ROS_WARN("Theta*: no path found (opened %d nodes)", nodes_opened_);
        return false;
    }

    backtrace(raw_path, curr);
    return true;
}

void ThetaStarCore::backtrace(std::vector<Coords>& raw, const TreeNode* curr) const
{
    std::vector<Coords> rev;
    while (curr->parent != curr)
    {
        double wx, wy;
        costmap_->mapToWorld(curr->x, curr->y, wx, wy);
        rev.push_back({wx, wy});
        curr = curr->parent;
    }
    double wx, wy;
    costmap_->mapToWorld(curr->x, curr->y, wx, wy);
    rev.push_back({wx, wy});

    for (int i = static_cast<int>(rev.size()) - 1; i >= 0; i--)
        raw.push_back(rev[i]);
}

void ThetaStarCore::clearQueue()
{
    queue_ = std::priority_queue<TreeNode*, std::vector<TreeNode*>, NodeCompare>();
}

// ============================================================
//  ThetaStarPlanner — ROS1 插件
// ============================================================

ThetaStarPlanner::ThetaStarPlanner()
  : costmap_ros_(nullptr), initialized_(false) {}

ThetaStarPlanner::ThetaStarPlanner(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
  : costmap_ros_(nullptr), initialized_(false)
{
    initialize(name, costmap_ros);
}

void ThetaStarPlanner::initialize(std::string name, costmap_2d::Costmap2DROS* costmap_ros)
{
    if (initialized_) return;

    costmap_ros_ = costmap_ros;
    core_.costmap_ = costmap_ros_->getCostmap();

    ros::NodeHandle nh("~/" + name);

    // Theta* 参数
    nh.param("w_euc_cost",       core_.w_euc_cost_,       1.0);
    nh.param("w_traversal_cost", core_.w_traversal_cost_, 0.1);
    nh.param("w_heuristic_cost", core_.w_heuristic_cost_, 1.0);
    nh.param("allow_unknown",    core_.allow_unknown_,    true);
    nh.param("max_non_obstacle", core_.max_non_obstacle_, 252);

    initialized_ = true;
    ROS_WARN("Theta* 全局规划器启动! euc=%.1f trav=%.2f heur=%.1f unknown=%s",
             core_.w_euc_cost_, core_.w_traversal_cost_, core_.w_heuristic_cost_,
             core_.allow_unknown_ ? "YES" : "NO");
}

bool ThetaStarPlanner::makePlan(const geometry_msgs::PoseStamped& start,
                                 const geometry_msgs::PoseStamped& goal,
                                 std::vector<geometry_msgs::PoseStamped>& plan)
{
    plan.clear();
    if (!costmap_ros_) return false;

    costmap_2d::Costmap2D* cm = costmap_ros_->getCostmap();
    core_.costmap_ = cm;

    // world → map 坐标
    unsigned int mx_s, my_s, mx_g, my_g;
    if (!cm->worldToMap(start.pose.position.x, start.pose.position.y, mx_s, my_s))
    {
        ROS_ERROR("Theta*: start outside map");
        return false;
    }
    if (!cm->worldToMap(goal.pose.position.x, goal.pose.position.y, mx_g, my_g))
    {
        ROS_ERROR("Theta*: goal outside map");
        return false;
    }

    core_.src_ = {static_cast<double>(mx_s), static_cast<double>(my_s)};
    core_.dst_ = {static_cast<double>(mx_g), static_cast<double>(my_g)};

    std::vector<Coords> raw;
    if (!core_.generatePath(raw))
    {
        ROS_WARN("Theta*: makePlan failed (%d nodes searched)", core_.nodes_opened_);
        return false;
    }

    // 转换为 ROS poseStamped 格式 + 线性插值
    double resolution = cm->getResolution();
    plan.reserve(raw.size() * 2);

    for (size_t j = 0; j < raw.size(); j++)
    {
        geometry_msgs::PoseStamped p;
        p.header.frame_id = costmap_ros_->getGlobalFrameID();
        p.pose.position.x = raw[j].x;
        p.pose.position.y = raw[j].y;
        p.pose.position.z = 0;
        p.pose.orientation.w = 1.0;
        plan.push_back(p);

        // 两点之间线性插值
        if (j + 1 < raw.size())
        {
            double dist = std::hypot(raw[j+1].x - raw[j].x, raw[j+1].y - raw[j].y);
            int loops = static_cast<int>(dist / resolution);
            if (loops > 1)
            {
                double sin_a = (raw[j+1].y - raw[j].y) / dist;
                double cos_a = (raw[j+1].x - raw[j].x) / dist;
                for (int k = 1; k < loops; k++)
                {
                    geometry_msgs::PoseStamped pi = p;
                    pi.pose.position.x = raw[j].x + k * resolution * cos_a;
                    pi.pose.position.y = raw[j].y + k * resolution * sin_a;
                    plan.push_back(pi);
                }
            }
        }
    }

    // 终点朝向
    if (!plan.empty())
        plan.back().pose.orientation = goal.pose.orientation;

    ROS_INFO("Theta*: 找到路径! %zu点, %d节点展开",
             plan.size(), core_.nodes_opened_);
    return true;
}

}  // namespace theta_star_ros
