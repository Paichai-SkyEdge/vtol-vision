#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ -f .venv-yolo/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-yolo/bin/activate
fi

MODEL_PATH="${MODEL_PATH:-runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt}"
DEVICE="${DEVICE:-0}"
IMG_SIZE="${IMG_SIZE:-640}"
HALF="${HALF:-True}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi

yolo export \
  model="$MODEL_PATH" \
  format=engine \
  device="$DEVICE" \
  imgsz="$IMG_SIZE" \
  half="$HALF"
