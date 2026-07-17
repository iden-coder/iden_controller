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

CONFLICT_PATTERN='^/(amcl|map_server|move_base|base_driver|my_base_driver|ydlidar_node|ucar_camera|cloud_asr_test2|xunfei2026_|subtask1_|factory_room_ocr)'
BASE_PORT="${BASE_SERIAL_PORT:-/dev/ttyS0}"
CLEANUP_DONE=0
FLOW_STARTED=0

flow_nodes() {
  timeout 0.35 rosnode list 2>/dev/null | grep -E "$CONFLICT_PATTERN" || true
}

stop_robot_once() {
  timeout 0.45 rostopic pub -1 /cmd_vel geometry_msgs/Twist \
    '{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}' \
    >/dev/null 2>&1 || true
}

base_port_owners() {
  fuser "$BASE_PORT" 2>/dev/null || true
}

wait_for_base_port_release() {
  for _ in {1..20}; do
    [[ -z "$(base_port_owners)" ]] && return 0
    sleep 0.10
  done
  return 1
}

cleanup_before_start() {
  local nodes owners pid cmd
  nodes="$(flow_nodes)"
  if [[ -n "$nodes" ]]; then
    echo "[INFO] Dropping stale ROS task registrations..."
    stop_robot_once
    # A dead node can remain registered in rosmaster and make an unrestricted
    # rosnode kill block for many seconds. New nodes safely replace stale
    # registrations, so cleanup here is strictly time-bounded.
    timeout 0.65 rosnode kill $nodes >/dev/null 2>&1 || true
    timeout 0.35 rosnode cleanup -y >/dev/null 2>&1 || true
  fi

  owners="$(base_port_owners)"
  if [[ -n "$owners" ]]; then
    for pid in $owners; do
      cmd="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
      case "$cmd" in
        *base_driver*|*iden_my_base_driver*)
          echo "[INFO] Stopping stale base driver PID $pid on $BASE_PORT..."
          kill -TERM "$pid" 2>/dev/null || true
          ;;
        *)
          echo "[ERROR] $BASE_PORT is occupied by an unknown process: PID=$pid CMD=$cmd" >&2
          return 1
          ;;
      esac
    done
  fi
  if ! wait_for_base_port_release; then
    owners="$(base_port_owners)"
    echo "[ERROR] Base serial port $BASE_PORT is still busy: PID=$owners" >&2
    return 1
  fi
}

cleanup_flow() {
  [[ "$FLOW_STARTED" -eq 0 ]] && return
  [[ "$CLEANUP_DONE" -eq 1 ]] && return
  CLEANUP_DONE=1
  set +e
  echo "[INFO] Cleaning up task nodes..."
  stop_robot_once
  sleep 0.05
  local nodes
  nodes="$(flow_nodes)"
  if [[ -n "$nodes" ]]; then
    # Startup rejected pre-existing nodes, so matching nodes at this point
    # belong to this flow and are safe to remove.
    timeout 0.65 rosnode kill $nodes >/dev/null 2>&1 || true
  fi
  timeout 0.35 rosnode cleanup -y >/dev/null 2>&1 || true
  echo "[INFO] Task cleanup complete."
}

on_signal() {
  exit 130
}

trap cleanup_flow EXIT
trap on_signal INT TERM

FILTER_PATTERN='check crc16 faild|header_crc8 error|check frame end|head_len error'

echo "[INFO] Complete flow: voice -> first nav -> 3 unique QR -> Spark/TTS -> lidar wall inspection -> OCR -> centerline parking -> final TTS"

cleanup_before_start

set +e
FLOW_STARTED=1
roslaunch iden_controller subtask1_xunfei2026_complete_delivery_v1.launch "$@" 2>&1 \
  | grep --line-buffered -v -E "$FILTER_PATTERN"
LAUNCH_STATUS="${PIPESTATUS[0]}"
exit "$LAUNCH_STATUS"
