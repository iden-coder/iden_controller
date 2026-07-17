#!/usr/bin/env bash
set -euo pipefail

WS="${INSTANT_WS:-/home/ucar/instant_ws}"
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

running_nodes="$(rosnode list 2>/dev/null || true)"
if printf '%s\n' "$running_nodes" \
    | grep -Eq '^/(amcl|map_server|move_base|my_base_driver|ydlidar_node|global_first_safety_monitor|global_first_graph_nav|front_first_smooth_navigation|factory_room.*)$'; then
  echo "[ERROR] Navigation/base nodes are already running. Stop the old launch first." >&2
  exit 2
fi

exec roslaunch iden_controller xunfei2026_first_stage_test_v1.launch "$@"
