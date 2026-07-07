#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os
import signal
import sys
import time

import actionlib
import rospy
from actionlib_msgs.msg import GoalID, GoalStatus
from geometry_msgs.msg import Pose, Point, Quaternion, PoseWithCovarianceStamped, Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from preflight_navfn_success1_guarded import load_waypoints, run_preflight


PARAM_NS = "/navfn_success1_guarded"
DEFAULT_WAYPOINTS_YAML = (
    "/home/ucar/instant_ws/src/iden_controller/config/"
    "navfn_success1_guarded_waypoints.yaml"
)
DEFAULT_MAP_YAML = "/home/ucar/instant_ws/src/ucar_nav/maps/6.3.1.yaml"


STATUS_NAMES = {
    GoalStatus.PENDING: "PENDING",
    GoalStatus.ACTIVE: "ACTIVE",
    GoalStatus.PREEMPTED: "PREEMPTED",
    GoalStatus.SUCCEEDED: "SUCCEEDED",
    GoalStatus.ABORTED: "ABORTED",
    GoalStatus.REJECTED: "REJECTED",
    GoalStatus.RECALLED: "RECALLED",
    GoalStatus.LOST: "LOST",
}


class State:
    def __init__(self):
        self.client = None
        self.nav_points = {}
        self.patrol_path = []
        self.force_moves = {}
        self.narrow_points = set()
        self.event_log = None

        self.odom_received = False
        self.odom_x = 0.0
        self.odom_y = 0.0
        self.odom_yaw = 0.0
        self.odom_prev = (0.0, 0.0)
        self.odom_yaw_prev = None
        self.odom_dist = 0.0
        self.odom_yaw_delta = 0.0

        self.amcl_received = False
        self.amcl_stamp = rospy.Time(0)
        self.amcl_x = 0.0
        self.amcl_y = 0.0
        self.amcl_yaw = 0.0
        self.amcl_cov_xy = float("inf")

        self.scan_received = False
        self.scan_stamp = rospy.Time(0)
        self.front_min = float("inf")
        self.left_min = float("inf")
        self.right_min = float("inf")
        self.rear_min = float("inf")


S = State()


def p(name, default):
    private_name = "~" + name
    if rospy.has_param(private_name):
        return rospy.get_param(private_name)
    global_name = PARAM_NS + "/" + name
    if rospy.has_param(global_name):
        return rospy.get_param(global_name)
    return default


def event(message, *args):
    text = message % args if args else message
    rospy.loginfo(text)
    if S.event_log:
        try:
            S.event_log.write("[%8.3f] %s\n" % (time.time(), text))
            S.event_log.flush()
        except Exception:
            pass


def warn(message, *args):
    text = message % args if args else message
    rospy.logwarn(text)
    if S.event_log:
        try:
            S.event_log.write("[%8.3f] WARN %s\n" % (time.time(), text))
            S.event_log.flush()
        except Exception:
            pass


def setup_event_log():
    log_path = p("event_log_path", "")
    if not log_path:
        log_dir = "/home/ucar/instant_ws/src/iden_controller/log_info"
        log_path = os.path.join(log_dir, "navfn_success1_guarded_%s.log" % time.strftime("%Y%m%d_%H%M%S"))
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        S.event_log = open(log_path, "w", encoding="utf-8")
        event("event_log=%s", log_path)
    except Exception as exc:
        S.event_log = None
        rospy.logwarn("event log unavailable: %s", exc)


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def angle_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


def goal_state_name(state):
    return STATUS_NAMES.get(state, "UNKNOWN(%s)" % state)


def cb_odom(msg):
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    yaw = yaw_from_quat(msg.pose.pose.orientation)
    if not S.odom_received:
        S.odom_prev = (x, y)
        S.odom_yaw_prev = yaw
    px, py = S.odom_prev
    S.odom_dist += math.hypot(x - px, y - py)
    if S.odom_yaw_prev is not None:
        S.odom_yaw_delta += abs(angle_diff(yaw, S.odom_yaw_prev))
    S.odom_prev = (x, y)
    S.odom_x = x
    S.odom_y = y
    S.odom_yaw = yaw
    S.odom_yaw_prev = yaw
    S.odom_received = True


def cb_amcl(msg):
    S.amcl_received = True
    S.amcl_stamp = rospy.Time.now()
    S.amcl_x = msg.pose.pose.position.x
    S.amcl_y = msg.pose.pose.position.y
    S.amcl_yaw = yaw_from_quat(msg.pose.pose.orientation)
    cov = msg.pose.covariance
    S.amcl_cov_xy = max(abs(cov[0]), abs(cov[7]))


def scan_min_in_range(msg, min_deg, max_deg):
    lo = math.radians(min_deg)
    hi = math.radians(max_deg)
    best = float("inf")
    for i, value in enumerate(msg.ranges):
        if math.isnan(value) or math.isinf(value):
            continue
        if not (msg.range_min <= value <= msg.range_max):
            continue
        angle = msg.angle_min + i * msg.angle_increment
        if lo <= angle <= hi and value < best:
            best = value
    return best


def cb_scan(msg):
    S.scan_received = True
    S.scan_stamp = rospy.Time.now()
    S.front_min = scan_min_in_range(msg, -30.0, 30.0)
    S.left_min = scan_min_in_range(msg, 35.0, 75.0)
    S.right_min = scan_min_in_range(msg, -75.0, -35.0)
    S.rear_min = min(
        scan_min_in_range(msg, 145.0, 180.0),
        scan_min_in_range(msg, -180.0, -145.0),
    )
    if p("swap_scan_left_right", False):
        S.left_min, S.right_min = S.right_min, S.left_min


def scan_is_fresh(max_age=None):
    if max_age is None:
        max_age = p("scan_fresh_timeout", 0.8)
    return S.scan_received and (rospy.Time.now() - S.scan_stamp).to_sec() <= max_age


def amcl_is_fresh(max_age=None):
    if max_age is None:
        max_age = p("amcl_fresh_timeout", 2.0)
    return S.amcl_received and (rospy.Time.now() - S.amcl_stamp).to_sec() <= max_age


def stop_robot(duration=0.35):
    pubs = [
        rospy.Publisher("/cmd_vel_raw", Twist, queue_size=1),
        rospy.Publisher("/cmd_vel", Twist, queue_size=1),
    ]
    rospy.sleep(0.05)
    zero = Twist()
    end_t = time.time() + duration
    rate = rospy.Rate(20)
    while time.time() < end_t and not rospy.is_shutdown():
        for pub in pubs:
            pub.publish(zero)
        rate.sleep()


def cancel_all():
    if S.client:
        S.client.cancel_all_goals()
    pub = rospy.Publisher("/move_base/cancel", GoalID, queue_size=10)
    rospy.sleep(0.08)
    pub.publish(GoalID())


def clear_costmaps():
    try:
        rospy.wait_for_service("/move_base/clear_costmaps", timeout=2.0)
        rospy.ServiceProxy("/move_base/clear_costmaps", Empty)()
        event("costmaps cleared")
        return True
    except Exception as exc:
        warn("clear_costmaps failed: %s", exc)
        return False


def publish_initial_pose():
    initial = p("initial_pose", [0.0, 0.0, 0.0])
    pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=5)
    rospy.sleep(0.3)
    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.pose.pose.position.x = float(initial[0])
    msg.pose.pose.position.y = float(initial[1])
    yaw = float(initial[2])
    msg.pose.pose.orientation.z = math.sin(yaw * 0.5)
    msg.pose.pose.orientation.w = math.cos(yaw * 0.5)
    cov_xy = float(p("initial_cov_xy", 0.12))
    cov_yaw = float(p("initial_cov_yaw", 0.20))
    msg.pose.covariance[0] = cov_xy
    msg.pose.covariance[7] = cov_xy
    msg.pose.covariance[35] = cov_yaw
    for _ in range(5):
        msg.header.stamp = rospy.Time.now()
        pub.publish(msg)
        rospy.sleep(0.12)
    event("initial pose sent x=%.3f y=%.3f yaw=%.2f", initial[0], initial[1], initial[2])


def wait_until(label, predicate, timeout):
    start = time.time()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() and time.time() - start < timeout:
        if predicate():
            event("%s ready", label)
            return True
        rate.sleep()
    warn("%s not ready after %.1fs", label, timeout)
    return False


def load_route():
    waypoints_yaml = p("waypoints_yaml", DEFAULT_WAYPOINTS_YAML)
    if rospy.has_param(PARAM_NS + "/nav_points"):
        S.nav_points = rospy.get_param(PARAM_NS + "/nav_points")
        S.patrol_path = rospy.get_param(PARAM_NS + "/patrol_path")
        S.force_moves = rospy.get_param(PARAM_NS + "/force_moves", {})
        S.narrow_points = set(rospy.get_param(PARAM_NS + "/narrow_points", []))
    else:
        data = load_waypoints(waypoints_yaml)
        S.nav_points = data["nav_points"]
        S.patrol_path = data["patrol_path"]
        S.force_moves = data["force_moves"]
        S.narrow_points = set(data["narrow_points"])
    for key, pose in list(S.nav_points.items()):
        S.nav_points[key] = [float(v) for v in pose]
    event("route loaded: %d points, %d force moves", len(S.patrol_path), len(S.force_moves))


def run_startup_preflight():
    if not p("enable_preflight", True):
        warn("preflight disabled by param")
        return True
    map_yaml = p("map_yaml", DEFAULT_MAP_YAML)
    lines = []

    def capture(line):
        lines.append(line)
        rospy.loginfo("[preflight] %s", line)

    ok = run_preflight(
        map_yaml, S.nav_points, S.patrol_path, S.force_moves,
        {
            "fail_goal_clearance": float(p("fail_goal_clearance", 0.13)),
            "warn_goal_clearance": float(p("warn_goal_clearance", 0.20)),
            "fail_force_clearance": float(p("fail_force_clearance", 0.18)),
            "warn_segment_clearance": float(p("warn_segment_clearance", 0.14)),
            "strict_segment_fail": bool(p("strict_segment_fail", False)),
            "unknown_is_obstacle": bool(p("unknown_is_obstacle", False)),
        },
        emit=capture,
    )
    for line in lines:
        if S.event_log:
            S.event_log.write("[preflight] %s\n" % line)
    if not ok:
        warn("preflight failed; no goal will be sent")
    return ok


def make_goal(pose7):
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose = Pose(
        Point(pose7[0], pose7[1], pose7[2]),
        Quaternion(pose7[3], pose7[4], pose7[5], pose7[6]))
    return goal


def target_yaw(pose7):
    q = Quaternion(pose7[3], pose7[4], pose7[5], pose7[6])
    return yaw_from_quat(q)


def goal_error(pose7):
    if not amcl_is_fresh():
        return None, None
    dist = math.hypot(S.amcl_x - pose7[0], S.amcl_y - pose7[1])
    yaw_err = abs(angle_diff(S.amcl_yaw, target_yaw(pose7)))
    return dist, yaw_err


def goal_tolerances(name):
    pos_tol = float(p("goal_pos_tolerance", 0.28))
    yaw_tol = float(p("goal_yaw_tolerance", 0.80))
    if name in S.narrow_points:
        pos_tol = max(pos_tol, float(p("narrow_goal_pos_tolerance", 0.32)))
        yaw_tol = max(yaw_tol, float(p("narrow_goal_yaw_tolerance", 0.95)))
    if S.patrol_path and name == S.patrol_path[-1]:
        pos_tol = float(p("final_goal_pos_tolerance", 0.30))
        yaw_tol = float(p("final_goal_yaw_tolerance", 0.90))
    return pos_tol, yaw_tol


def goal_is_verified(name, pose7):
    if not amcl_is_fresh():
        return False, "amcl pose missing or stale"
    max_cov = float(p("max_amcl_cov_xy", 0.45))
    if S.amcl_cov_xy > max_cov:
        return False, "amcl covariance %.3f > %.3f" % (S.amcl_cov_xy, max_cov)
    dist, yaw_err = goal_error(pose7)
    pos_tol, yaw_tol = goal_tolerances(name)
    if dist <= pos_tol and yaw_err <= yaw_tol:
        return True, "pos %.2fm yaw %.1fdeg" % (dist, math.degrees(yaw_err))
    return False, "pos %.2f/%.2fm yaw %.1f/%.1fdeg" % (
        dist, pos_tol, math.degrees(yaw_err), math.degrees(yaw_tol))


def reset_progress_counters():
    S.odom_dist = 0.0
    S.odom_prev = (S.odom_x, S.odom_y)
    S.odom_yaw_delta = 0.0
    S.odom_yaw_prev = S.odom_yaw


def attempt_timeout_for(pose7):
    if amcl_is_fresh():
        dist = math.hypot(S.amcl_x - pose7[0], S.amcl_y - pose7[1])
    else:
        dist = 1.5
    base = float(p("goal_timeout_base", 14.0))
    per_meter = float(p("goal_timeout_per_meter", 8.0))
    max_t = float(p("goal_timeout_max", 34.0))
    min_t = float(p("goal_timeout_min", 12.0))
    return max(min_t, min(max_t, base + per_meter * dist))


def goto_point_once(name, pose7):
    reset_progress_counters()
    timeout = attempt_timeout_for(pose7)
    stuck_window = float(p("stuck_window", 4.0))
    stuck_min_dist = float(p("stuck_min_dist", 0.045))
    stuck_min_yaw = math.radians(float(p("stuck_min_yaw_deg", 7.0)))
    blocked_front = float(p("blocked_front_dist", 0.17))

    stuck_dist_at = 0.0
    stuck_yaw_at = 0.0
    stuck_time_at = time.time()
    start = time.time()

    event("[%s] send goal x=%.3f y=%.3f timeout=%.1fs", name, pose7[0], pose7[1], timeout)
    S.client.send_goal(make_goal(pose7))
    rate = rospy.Rate(8)
    while not rospy.is_shutdown():
        elapsed = time.time() - start
        state = S.client.get_state()

        if state == GoalStatus.SUCCEEDED:
            ok, reason = goal_is_verified(name, pose7)
            if ok:
                event("[%s] verified success: %s", name, reason)
                return True, "success"
            warn("[%s] move_base succeeded but verification failed: %s", name, reason)
            return False, "not_verified"

        if state in (GoalStatus.ABORTED, GoalStatus.REJECTED, GoalStatus.RECALLED,
                     GoalStatus.PREEMPTED, GoalStatus.LOST):
            warn("[%s] move_base ended: %s", name, goal_state_name(state))
            ok, reason = goal_is_verified(name, pose7)
            if ok:
                event("[%s] accepted after action end: %s", name, reason)
                return True, "verified_after_end"
            return False, goal_state_name(state)

        if scan_is_fresh() and S.front_min < blocked_front:
            warn("[%s] blocked front %.2fm < %.2fm", name, S.front_min, blocked_front)
            return False, "blocked_front"

        if elapsed >= timeout:
            ok, reason = goal_is_verified(name, pose7)
            if ok:
                event("[%s] accepted on timeout: %s", name, reason)
                return True, "verified_timeout"
            warn("[%s] timeout; %s", name, reason)
            return False, "timeout"

        now = time.time()
        if now - stuck_time_at >= stuck_window:
            moved = S.odom_dist - stuck_dist_at
            yaw = S.odom_yaw_delta - stuck_yaw_at
            ok, reason = goal_is_verified(name, pose7)
            if ok:
                event("[%s] accepted during progress check: %s", name, reason)
                return True, "verified_progress"
            if moved < stuck_min_dist and yaw < stuck_min_yaw:
                warn("[%s] stuck moved=%.3fm yaw=%.1fdeg", name, moved, math.degrees(yaw))
                return False, "stuck"
            stuck_dist_at = S.odom_dist
            stuck_yaw_at = S.odom_yaw_delta
            stuck_time_at = now

        rate.sleep()
    return False, "shutdown"


def reverse_escape(distance=0.08, speed=0.04):
    if not bool(p("enable_reverse_escape", False)):
        warn("reverse escape disabled")
        stop_robot(0.25)
        return False
    min_rear = float(p("reverse_min_rear_clear", 0.28))
    if (not scan_is_fresh()) or S.rear_min < min_rear:
        warn("reverse escape blocked: rear %.2fm < %.2fm", S.rear_min, min_rear)
        stop_robot(0.25)
        return False
    pub = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=3)
    rospy.sleep(0.1)
    start_x = S.odom_x
    start_y = S.odom_y
    max_duration = distance / max(speed, 1e-3) + 0.5
    twist = Twist()
    twist.linear.x = -abs(speed)
    start = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if (not scan_is_fresh()) or S.rear_min < min_rear:
            warn("reverse escape stopped: rear %.2fm < %.2fm", S.rear_min, min_rear)
            break
        moved = math.hypot(S.odom_x - start_x, S.odom_y - start_y)
        if moved >= distance or time.time() - start >= max_duration:
            break
        pub.publish(twist)
        rate.sleep()
    stop_robot(0.2)
    event("reverse escape moved %.2fm", math.hypot(S.odom_x - start_x, S.odom_y - start_y))
    return True


def micro_wiggle():
    if not scan_is_fresh():
        return
    if S.front_min < float(p("wiggle_min_front", 0.24)):
        return
    direction = -1.0 if S.left_min < S.right_min else 1.0
    pub = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=3)
    twist = Twist()
    twist.angular.z = direction * float(p("wiggle_angular_speed", 0.16))
    duration = float(p("wiggle_duration", 0.55))
    start = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.time() - start < duration:
        pub.publish(twist)
        rate.sleep()
    stop_robot(0.15)
    event("micro wiggle direction=%+.0f", direction)


def scan_clear_for_motion(label, direction, min_front, min_side, min_rear):
    if not scan_is_fresh():
        warn("[%s] motion blocked: scan stale", label)
        return False
    if direction > 0.0 and S.front_min < min_front:
        warn("[%s] motion blocked: front %.2fm < %.2fm", label, S.front_min, min_front)
        return False
    if direction < 0.0 and S.rear_min < min_rear:
        warn("[%s] motion blocked: rear %.2fm < %.2fm", label, S.rear_min, min_rear)
        return False
    side = min(S.left_min, S.right_min)
    if side < min_side:
        warn("[%s] motion blocked: side %.2fm < %.2fm", label, side, min_side)
        return False
    return True


def target_heading_error(pose7):
    if not amcl_is_fresh():
        return None
    desired = math.atan2(pose7[1] - S.amcl_y, pose7[0] - S.amcl_x)
    return angle_diff(desired, S.amcl_yaw)


def turn_toward_target(name, pose7):
    if not bool(p("enable_target_turn", True)):
        return False
    if not amcl_is_fresh():
        warn("[%s] target turn skipped: amcl stale", name)
        return False
    min_any = float(p("target_turn_min_clear", 0.16))
    if not scan_is_fresh() or min(S.front_min, S.left_min, S.right_min, S.rear_min) < min_any:
        warn("[%s] target turn skipped: clearance too small", name)
        return False
    err = target_heading_error(pose7)
    if err is None:
        return False
    deadband = math.radians(float(p("target_turn_deadband_deg", 8.0)))
    if abs(err) <= deadband:
        return True
    speed = abs(float(p("target_turn_speed", 0.18)))
    max_turn = math.radians(float(p("target_turn_max_deg", 28.0)))
    duration = min(abs(err), max_turn) / max(speed, 1e-3)
    direction = 1.0 if err > 0.0 else -1.0
    pub = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=3)
    twist = Twist()
    twist.angular.z = direction * speed
    start = time.time()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown() and time.time() - start < duration:
        if not scan_is_fresh() or min(S.front_min, S.left_min, S.right_min, S.rear_min) < min_any:
            warn("[%s] target turn stopped: clearance too small", name)
            stop_robot(0.15)
            return False
        pub.publish(twist)
        rate.sleep()
    stop_robot(0.12)
    event("[%s] target turn err=%.1fdeg applied=%.1fdeg", name, math.degrees(err), math.degrees(min(abs(err), max_turn)))
    return True


def guarded_forward_unstick(name, pose7, round_index):
    if not bool(p("enable_forward_unstick", True)):
        return False
    every = max(1, int(p("unstick_every_n_rounds", 1)))
    if round_index % every != 0:
        return False
    err = target_heading_error(pose7)
    if err is None:
        warn("[%s] forward unstick skipped: amcl stale", name)
        return False
    max_err = math.radians(float(p("unstick_forward_max_heading_error_deg", 35.0)))
    if abs(err) > max_err:
        warn("[%s] forward unstick skipped: heading error %.1fdeg > %.1fdeg",
             name, math.degrees(abs(err)), math.degrees(max_err))
        return False
    move = {
        "distance": float(p("unstick_forward_distance", 0.10)),
        "speed": float(p("unstick_forward_speed", 0.055)),
        "direction": 1.0,
        "step_distance": float(p("unstick_step_distance", 0.035)),
        "min_front_clear": float(p("unstick_min_front_clear", 0.32)),
        "min_side_clear": float(p("unstick_min_side_clear", 0.16)),
        "min_rear_clear": float(p("unstick_min_rear_clear", 0.22)),
    }
    return guarded_force_move("%s:unstick%d" % (name, round_index), move)


def recover_before_retry(name, pose7, reason, round_index):
    warn("[%s] recovery round %d after %s", name, round_index, reason)
    cancel_all()
    stop_robot(0.25)
    turned = turn_toward_target(name, pose7)
    nudged = False
    if reason in ("stuck", "timeout", "not_verified", "PREEMPTED", "ABORTED", "LOST"):
        nudged = guarded_forward_unstick(name, pose7, round_index)
    if not nudged and scan_is_fresh() and S.front_min < float(p("escape_front_trigger", 0.23)):
        reverse_escape(float(p("escape_distance", 0.06)), float(p("escape_speed", 0.035)))
    elif not nudged and not turned:
        micro_wiggle()
    clear_costmaps()
    rospy.sleep(float(p("recovery_settle_sec", 0.45)))


def goto_point(name, pose7, is_final=False):
    max_rounds = int(p("max_recovery_rounds", 6))
    keep_trying_final = bool(p("keep_trying_final", True))
    keep_trying_all = bool(p("keep_trying_all_goals", True))
    allow_skip = bool(p("allow_skip_after_recovery", False))
    round_index = 0
    while not rospy.is_shutdown():
        ok, reason = goto_point_once(name, pose7)
        if ok:
            return "OK"
        round_index += 1
        recover_before_retry(name, pose7, reason, round_index)
        if round_index >= max_rounds:
            if reason == "blocked_front" and not bool(p("keep_trying_blocked_front", False)):
                warn("[%s] blocked in front after %d recovery rounds; safe pause", name, max_rounds)
                return "FAIL"
            if keep_trying_all or (is_final and keep_trying_final):
                warn("[%s] still not reached after %d recovery rounds; continuing same goal",
                     name, max_rounds)
                round_index = 0
                continue
            if (not is_final) and allow_skip:
                warn("[%s] skip after %d recovery rounds; trying later waypoints", name, max_rounds)
                return "SKIP"
            return "FAIL"
    return "FAIL"


def guarded_force_move(name, move):
    distance = abs(float(move.get("distance", 0.0)))
    if distance <= 0.0:
        return True
    direction = 1.0 if float(move.get("direction", 1.0)) >= 0.0 else -1.0
    speed = abs(float(move.get("speed", 0.20)))
    step_distance = abs(float(move.get("step_distance", 0.06)))
    min_front = float(move.get("min_front_clear", p("force_min_front_clear", 0.26)))
    min_side = float(move.get("min_side_clear", p("force_min_side_clear", 0.14)))
    min_rear = float(move.get("min_rear_clear", p("force_min_rear_clear", 0.28)))
    pub = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=3)
    travelled = 0.0
    event("[%s] guarded force move distance=%.2fm speed=%.2fm/s", name, distance, speed)
    while travelled < distance and not rospy.is_shutdown():
        if not scan_clear_for_motion(name, direction, min_front, min_side, min_rear):
            stop_robot(0.25)
            return False
        step = min(step_distance, distance - travelled)
        duration = step / max(speed, 1e-3)
        twist = Twist()
        twist.linear.x = direction * speed
        start = time.time()
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() - start < duration:
            if not scan_clear_for_motion(name, direction, min_front, min_side, min_rear):
                stop_robot(0.25)
                return False
            pub.publish(twist)
            rate.sleep()
        travelled += step
        stop_robot(0.08)
    event("[%s] guarded force move done", name)
    return True


def cruise():
    total = len(S.patrol_path)
    ok_count = 0
    skipped = []
    failed = []
    for index, name in enumerate(S.patrol_path):
        if rospy.is_shutdown():
            break
        if name not in S.nav_points:
            warn("[%s] missing from nav_points", name)
            skipped.append(name)
            continue
        is_final = index == total - 1
        event("--- [%d/%d] %s ---", index + 1, total, name)
        result = goto_point(name, S.nav_points[name], is_final=is_final)
        if result == "OK":
            ok_count += 1
            if name in S.force_moves:
                if not guarded_force_move(name, S.force_moves[name]):
                    warn("[%s] guarded force move failed; continuing via move_base route", name)
                    clear_costmaps()
        elif result == "SKIP":
            skipped.append(name)
        else:
            failed.append(name)
            if is_final:
                warn("final goal failed unexpectedly")
            elif not bool(p("continue_after_failure", True)):
                break
        rospy.sleep(float(p("between_goals_sleep", 0.18)))

    stop_robot(0.4)
    event("patrol loop finished: ok=%d/%d skipped=%d failed=%d", ok_count, total, len(skipped), len(failed))
    if skipped:
        warn("skipped points: %s", ", ".join(skipped))
    if failed:
        warn("failed points: %s", ", ".join(failed))


def on_shutdown(sig=None, frame=None):
    try:
        warn("shutdown requested")
        cancel_all()
        stop_robot(0.35)
    finally:
        if S.event_log:
            try:
                S.event_log.close()
            except Exception:
                pass
        rospy.signal_shutdown("user_exit")
        sys.exit(0)


def main():
    rospy.init_node("process_navfn_success1_guarded")
    signal.signal(signal.SIGINT, on_shutdown)
    setup_event_log()

    rospy.Subscriber("/odom", Odometry, cb_odom, queue_size=1)
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, cb_amcl, queue_size=1)
    rospy.Subscriber("/scan", LaserScan, cb_scan, queue_size=1)

    load_route()
    if not run_startup_preflight():
        stop_robot(0.3)
        return 2

    event("connecting move_base action server...")
    S.client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
    if not S.client.wait_for_server(rospy.Duration(float(p("move_base_wait_sec", 45.0)))):
        warn("move_base unavailable")
        stop_robot(0.3)
        return 3
    event("move_base connected")

    if not wait_until("scan", lambda: scan_is_fresh(), float(p("scan_wait_sec", 12.0))):
        warn("scan is required; no goal will be sent")
        stop_robot(0.3)
        return 5
    if not wait_until("odom", lambda: S.odom_received, float(p("odom_wait_sec", 8.0))):
        warn("odom is required; no goal will be sent")
        stop_robot(0.3)
        return 6
    try:
        rospy.wait_for_service("/static_map", timeout=float(p("map_wait_sec", 8.0)))
        event("static_map service ready")
    except Exception as exc:
        warn("static_map service not ready: %s", exc)

    if bool(p("publish_initial_pose_on_start", True)):
        publish_initial_pose()
    else:
        event("initial pose publish skipped by param")
    if not wait_until("amcl", lambda: amcl_is_fresh(), float(p("amcl_wait_sec", 18.0))):
        warn("AMCL is required; no goal will be sent")
        stop_robot(0.3)
        return 4

    clear_costmaps()
    event("start guarded patrol")
    cruise()
    event("task loop done; node stays alive for inspection")
    rospy.spin()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
