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

CONFLICT_PATTERN='^/(amcl|map_server|move_base|my_base_driver|ydlidar_node|ucar_camera|cloud_asr_test2|xunfei2026_|subtask1_)'
CONFLICTS="$(rosnode list 2>/dev/null | grep -E "$CONFLICT_PATTERN" || true)"
if [[ -n "$CONFLICTS" ]]; then
  echo "[INFO] Previous ROS nodes are still shutting down; waiting up to 6 seconds..."
  for _ in {1..6}; do
    sleep 1
    CONFLICTS="$(rosnode list 2>/dev/null | grep -E "$CONFLICT_PATTERN" || true)"
    [[ -z "$CONFLICTS" ]] && break
  done
fi
if [[ -n "$CONFLICTS" ]]; then
  echo "[ERROR] Conflicting ROS nodes are already running:" >&2
  echo "$CONFLICTS" >&2
  echo "[ERROR] Stop the old navigation/task launch before starting this flow." >&2
  exit 2
fi

FILTER_PATTERN='check crc16 faild|header_crc8 error|check frame end|head_len error'

echo "[INFO] Flow: wake/order -> Xunfei2026 first navigation -> continuous hybrid QR -> Spark -> TTS"
echo "[INFO] Large-room navigation is disabled."

if [[ "${SHOW_SERIAL_WARNINGS:-0}" == "1" ]]; then
  exec roslaunch iden_controller subtask1_xunfei2026_voice_nav_qr_llm_v1.launch "$@"
fi

set +e
roslaunch iden_controller subtask1_xunfei2026_voice_nav_qr_llm_v1.launch "$@" 2>&1 \
  | grep --line-buffered -v -E "$FILTER_PATTERN"
exit "${PIPESTATUS[0]}"
