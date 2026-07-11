#!/usr/bin/env bash
set -euo pipefail

WS="${INSTANT_WS:-/home/ucar/instant_ws}"
PKG_DIR="$WS/src/iden_controller"
CRED_FILE="$PKG_DIR/config/spark_credentials.env"

source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

if [[ ! -f "$CRED_FILE" ]]; then
  echo "[ERROR] Missing Spark credential file: $CRED_FILE" >&2
  echo "[ERROR] Create it once from $PKG_DIR/config/spark_credentials.env.example" >&2
  exit 1
fi

set -a
source "$CRED_FILE"
set +a

if [[ -z "${SPARK_API_KEY:-}" || -z "${SPARK_API_SECRET:-}" ]]; then
  echo "[ERROR] SPARK_API_KEY or SPARK_API_SECRET is empty in $CRED_FILE" >&2
  exit 1
fi

FILTER_PATTERN='check crc16 faild|header_crc8 error|check frame end|head_len error'

if [[ "${SHOW_SERIAL_WARNINGS:-0}" == "1" ]]; then
  exec roslaunch iden_controller subtask1_real_factory_route.launch "$@"
fi

set +e
roslaunch iden_controller subtask1_real_factory_route.launch "$@" 2>&1 \
  | grep --line-buffered -v -E "$FILTER_PATTERN"
exit "${PIPESTATUS[0]}"
