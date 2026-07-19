#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_ONNX="${PROJECT_DIR}/weights/yolo11n_shadow_v1_best.onnx"

echo "=== Exporting YOLO model to ONNX ==="
echo "  Output: ${MODEL_ONNX}"

if ! command -v yolo &>/dev/null; then
  echo "ERROR: 'yolo' CLI not found. Install: pip install ultralytics"
  exit 1
fi

PT="${PROJECT_DIR}/weights/yolo11n_shadow_v1_best.pt"
if [ ! -f "$PT" ]; then
  echo "ERROR: model not found at ${PT}"
  exit 1
fi

yolo export model="$PT" format=onnx imgsz=640
mv "${PT%.pt}.onnx" "$MODEL_ONNX" 2>/dev/null || true
echo "Done: $MODEL_ONNX"
