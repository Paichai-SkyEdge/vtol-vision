#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_PT="${PROJECT_DIR}/weights/yolo11n_shadow_v1_best.pt"
ENGINE_OUT="${PROJECT_DIR}/weights/yolo11n_shadow_v1_best.engine"

echo "=== Exporting YOLO model to TensorRT engine ==="
echo "  Input : ${MODEL_PT}"
echo "  Output: ${ENGINE_OUT}"

if ! command -v yolo &>/dev/null; then
  echo "ERROR: 'yolo' CLI not found. Install: pip install ultralytics"
  exit 1
fi

if [ ! -f "$MODEL_PT" ]; then
  echo "ERROR: model not found at ${MODEL_PT}"
  echo "Copy your best.pt to ${PROJECT_DIR}/weights/ first."
  exit 1
fi

yolo export model="$MODEL_PT" format=engine device=0 imgsz=640 half=True

echo ""
echo "=== Done ==="
echo "Engine saved to: ${ENGINE_OUT:-check above}"
echo "Now run: ./run.sh"
