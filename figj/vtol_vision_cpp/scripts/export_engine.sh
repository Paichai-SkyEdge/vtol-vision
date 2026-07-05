#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${1:-$ROOT_DIR/weights/basket_mannequin_yolo11n_best.pt}"

if ! command -v yolo >/dev/null 2>&1; then
  echo "error: yolo command not found. Install ultralytics on the target first." >&2
  echo "       python3 -m pip install ultralytics" >&2
  exit 1
fi

yolo export \
  model="$MODEL_PATH" \
  format=engine \
  device=0 \
  imgsz=640 \
  half=True

echo "engine: ${MODEL_PATH%.pt}.engine"
