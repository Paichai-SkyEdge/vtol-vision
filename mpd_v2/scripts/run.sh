#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BINARY="${PROJECT_DIR}/build/realsense_live_detect"
ENGINE="${PROJECT_DIR}/weights/yolo11n_shadow_v1_best.engine"

# --- Build if needed ---
if [ ! -f "$BINARY" ]; then
  echo "=== Building (RealSense + TensorRT) ==="
  mkdir -p "${PROJECT_DIR}/build"
  cd "${PROJECT_DIR}/build"
  cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_REALSENSE=ON -DWITH_TENSORRT=ON
  make -j$(nproc)
  cd "${PROJECT_DIR}"
fi

# --- Check engine ---
if [ ! -f "$ENGINE" ]; then
  echo "TensorRT engine not found."
  echo "  For Jetson: ./scripts/export_engine.sh"
  echo "  For PC/sim: ./scripts/run_sim.sh"
  exit 1
fi

# --- Run ---
echo "=== Starting RealSense + TensorRT ==="
exec "$BINARY" \
  --model "$ENGINE" \
  --width 848 --height 480 \
  --depth-width 848 --depth-height 480 \
  "$@"
