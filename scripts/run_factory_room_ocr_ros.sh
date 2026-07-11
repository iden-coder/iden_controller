#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/ucar/instant_ws/src/iden_controller/scripts/factory_room_ocr_ros.py"
exec /home/ucar/venv3.9/bin/python "$SCRIPT" "$@"

