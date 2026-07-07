# Global-First Planner Design Notes - 2026-07-06

This is a new navigation flow for `iden_controller`. It does not modify old
planner source files, launch files, or plugin XML files.

## Existing planner references checked

- `my_planner`: early local planner. It tracks a global plan with pure-pursuit
  style target selection, but it allows reverse motion and has an unsafe final
  success condition with very loose tolerances.
- `iden_planner/IdenPlanner`: local planner plugin. It improves target tracking,
  PID state handling, speed-angle coupling, and parameter reloads, but it still
  contains bidirectional/reverse logic and can force goal success after pose
  adjustment timeout.
- `iden_planner_v2/IdenPlannerV2`: local planner plugin. Useful ideas include
  footprint checking, forward simulation, trajectory sampling, side-clearance
  checks, and velocity smoothing. The dangerous part for this task is recovery
  by backing up, which already caused a wall collision at launch.
- `theta_star_ros/ThetaStarPlanner`: global planner plugin. Useful ideas include
  8-connected search and line-of-sight smoothing. The new planner keeps this
  global-first spirit but runs as a standalone Python node to avoid modifying
  old build files or plugin XML.
- DWA/TEB/global planner configs: useful safety ideas are low speed limits,
  strong forward-drive preference, dense path via-points, obstacle inflation,
  and continuous visualization/status output. Reverse allowance is not reused.

## New behavior

- One goal only: current AMCL pose to final goal, no intermediate waypoints.
- Static map planning is primary. The node reads `/static_map` or `/map`, then
  computes a full-map distance field and A* path with obstacle-proximity cost.
- Local behavior is auxiliary. It follows the global path, slows down from scan
  data, stops if the front/side is unsafe, and requests global replanning.
- The node never emits negative `linear.x`.
- Stuck is not success. If progress is too small for `stuck_timeout_s`, the node
  stops and replans from the current AMCL pose.
- Goal success requires position tolerance plus a short hold time. The node
  stays alive and continues publishing zero velocity after success.
- If a similar map has slightly shifted walls, the planner first tries the
  conservative clearance and can relax toward an emergency clearance instead of
  immediately failing.

## Files added

- `scripts/global_first_nav.py`
- `config/global_first_nav.yaml`
- `launch/global_first_nav.launch`
- `audit_reports/global_first_planner_design_20260706.md`

## Main command

```bash
roslaunch iden_controller global_first_nav.launch
```

Override map or final goal:

```bash
roslaunch iden_controller global_first_nav.launch \
  map_yaml:=/path/to/new_map.yaml \
  goal_x:=-1.50 goal_y:=-0.40 goal_yaw:=3.1415926
```

## Offline check on the current map

The local offline check on `6.3.1.yaml` from start `(1.07, 0.0)` to final goal
`(-1.5, -0.4)` passed with:

- hard clearance tested: `0.17 m`
- minimum map clearance on raw path: about `0.21 m`
- path length: about `7.57 m`
- no intermediate waypoints

Actual collision safety still depends on correct map scale, AMCL localization,
laser frame alignment, and the real robot footprint.
