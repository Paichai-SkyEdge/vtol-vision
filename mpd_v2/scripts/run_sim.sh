#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="${PROJECT_DIR}/build/realsense_live_detect"
MODEL="${PROJECT_DIR}/weights/yolo11n_shadow_v1_best.onnx"

# --- Build if needed ---
if [ ! -f "$BINARY" ]; then
  echo "=== Building (sim mode: ONNX + OpenCV DNN) ==="
  mkdir -p "${PROJECT_DIR}/build"
  cd "${PROJECT_DIR}/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_REALSENSE=OFF -DWITH_TENSORRT=OFF
  make -j$(nproc)
  cd "${PROJECT_DIR}"
fi

# --- Check model ---
if [ ! -f "$MODEL" ]; then
  echo "ONNX model not found. Run ./scripts/export_onnx.sh first."
  exit 1
fi

# --- Run ---
echo "=== Starting Sim Mode ==="
exec "$BINARY" \
  --sim \
  --model "$MODEL" \
  --width 848 --height 480 \
  "$@"
