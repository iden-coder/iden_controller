#!/usr/bin/env bash
set -euo pipefail

WS="${INSTANT_WS:-/home/ucar/instant_ws}"
PKG_DIR="$WS/src/iden_controller"
CREDENTIALS="$PKG_DIR/config/spark_credentials.env"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

if [[ ! -f "$CREDENTIALS" ]]; then
  echo "[ERROR] Missing Spark credential file: $CREDENTIALS" >&2
  exit 1
fi

set -a
source "$CREDENTIALS"
set +a

if [[ -z "${SPARK_API_KEY:-}" || -z "${SPARK_API_SECRET:-}" ]]; then
  echo "[ERROR] SPARK_API_KEY or SPARK_API_SECRET is empty in $CREDENTIALS" >&2
  exit 1
fi

CONFLICT_PATTERN='^/(amcl|map_server|move_base|my_base_driver|ydlidar_node|ucar_camera|cloud_asr_test2|xunfei2026_|subtask1_|factory_room_ocr)'
CLEANUP_DONE=0
FLOW_STARTED=0

flow_nodes() {
  rosnode list 2>/dev/null | grep -E "$CONFLICT_PATTERN" || true
}

stop_robot_once() {
  timeout 1.2 rostopic pub -1 /cmd_vel geometry_msgs/Twist \
    '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
    >/dev/null 2>&1 || true
}

cleanup_flow() {
  [[ "$FLOW_STARTED" -eq 0 ]] && return
  [[ "$CLEANUP_DONE" -eq 1 ]] && return
  CLEANUP_DONE=1
  set +e
  echo "[INFO] Cleaning up task nodes..."
  stop_robot_once
  sleep 0.25
  local nodes
  nodes="$(flow_nodes)"
  if [[ -n "$nodes" ]]; then
    # Startup rejected pre-existing nodes, so matching nodes at this point
    # belong to this flow and are safe to remove.
    rosnode kill $nodes >/dev/null 2>&1 || true
  fi
  for _ in {1..12}; do
    [[ -z "$(flow_nodes)" ]] && break
    sleep 0.15
  done
  echo "[INFO] Task cleanup complete."
}

on_signal() {
  exit 130
}

trap cleanup_flow EXIT
trap on_signal INT TERM

FILTER_PATTERN='check crc16 faild|header_crc8 error|check frame end|head_len error'

echo "[INFO] Complete flow: voice -> first nav -> 3 unique QR -> Spark/TTS -> lidar wall inspection -> OCR -> centerline parking -> final TTS"

set +e
FLOW_STARTED=1
roslaunch iden_controller subtask1_xunfei2026_complete_delivery_v1.launch "$@" 2>&1 \
  | grep --line-buffered -v -E "$FILTER_PATTERN"
LAUNCH_STATUS="${PIPESTATUS[0]}"
exit "$LAUNCH_STATUS"
