# Global-first graph navigation with optional vision guard

This is a new-file-only upgrade for `iden_controller`.

## Purpose

The navigation stack is designed around one final goal, not intermediate
waypoints. Global planning is dominant; local behavior only tracks the selected
path and filters unsafe velocity commands.

## Command Chain

1. `global_first_graph_nav.py`
   - Loads the static map.
   - Builds a clearance distance field.
   - Plans first on a sparse safe roadmap/graph.
   - Falls back to dense grid A* if the graph cannot connect.
   - Publishes forward-only commands to `/cmd_vel_graph_raw`.

2. `global_first_vision_guard.py`
   - Subscribes to `/ucar_camera/image_raw`.
   - Looks at the lower image region for strong boundary/line evidence.
   - Only when confidence is high, applies small heading correction and speed
     reduction.
   - If the camera is missing or unclear, passes commands through.
   - Publishes to `/cmd_vel_raw`.

3. `global_first_safety_monitor.py`
   - Final lidar safety layer.
   - Blocks reverse commands.
   - Stops/slows/turns away when laser sectors are too close.
   - Publishes to `/cmd_vel`.

## Why not directly use the existing vision followers?

The workspace already contains `line_follower` and `flow_end`, but both are task
controllers that publish `/cmd_vel` directly. Running them together with this
global navigation would cause command contention. Their line/corridor processing
ideas are useful, but for this mission vision must stay auxiliary.

## Safety Constraints

- No reverse command is emitted by the graph planner.
- The vision guard never increases linear speed.
- Visual stop is disabled by default; vision slows/corrects, lidar stops.
- If localization, scan, or map data is missing, the planner publishes zero.
- Stuck does not mean success; the node replans and keeps trying.

## Similar-map Robustness

The roadmap planner is map tolerant because it samples safe high-clearance
cells and connects them with line-of-sight checks. If a similar YAML map is
slightly noisier or narrower, the planner relaxes clearance down to the
emergency limit and falls back to dense grid A*.

## Launch

Use:

```bash
roslaunch iden_controller global_first_graph_nav.launch
```

The new launch defaults the initial pose to `(1.07, 0.0, 0.0)` and therefore
does not overwrite the YAML start pose with `(0, 0)`.

If a camera is already running:

```bash
roslaunch iden_controller global_first_graph_nav.launch start_camera:=false
```

If visual correction looks harmful in a new scene:

```bash
roslaunch iden_controller global_first_graph_nav.launch vision_guard_enabled:=false
```
