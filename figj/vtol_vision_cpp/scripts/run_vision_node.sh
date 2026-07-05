#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_PATH="${1:-$ROOT_DIR/weights/basket_mannequin_yolo11n_best.engine}"
CAMERA_URI="${CAMERA_URI:-0}"

if [[ ! -f "$ENGINE_PATH" ]]; then
  echo "error: TensorRT engine not found: $ENGINE_PATH" >&2
  echo "run: $ROOT_DIR/scripts/export_engine.sh" >&2
  exit 1
fi

ros2 launch vtol_vision vision.launch.py \
  camera_uri:="$CAMERA_URI" \
  trt_engine_path:="$ENGINE_PATH" \
  class_map_yaml:="$ROOT_DIR/config/basket_mannequin_class_map.yaml"
