#!/usr/bin/env bash
set -euo pipefail

OCR_DIR="/home/ucar/instant_ws/src/iden_controller/factory_ocr_car_deploy"
PYTHON_BIN="/home/ucar/venv3.9/bin/python"
ENTRYPOINT="camera_det_rec_final_working.py"

cd "$OCR_DIR"
exec "$PYTHON_BIN" "$ENTRYPOINT"
