#!/usr/bin/env bash
set -euo pipefail

WS="${INSTANT_WS:-/home/ucar/instant_ws}"
source /opt/ros/noetic/setup.bash
source "$WS/devel/setup.bash"

FILTER_PATTERN='check crc16 faild|header_crc8 error|check frame end|head_len error'
set +e
roslaunch iden_controller front_first_stable_dynamic_route_test_v1.launch "$@" 2>&1 \
  | grep --line-buffered -v -E "$FILTER_PATTERN"
exit "${PIPESTATUS[0]}"
