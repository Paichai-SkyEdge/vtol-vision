#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_VENV="${USE_VENV:-1}"
VENV_DIR="${VENV_DIR:-.venv-yolo}"

if [[ "$USE_VENV" == "1" ]]; then
  if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

MODEL="${MODEL:-yolo11n.pt}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-4}"
IMG_SIZE="${IMG_SIZE:-640}"
RUN_NAME="${RUN_NAME:-yolo11n_v1}"

yolo detect train \
  model="$MODEL" \
  data=dataset.yaml \
  epochs="$EPOCHS" \
  imgsz="$IMG_SIZE" \
  batch="$BATCH" \
  workers="$WORKERS" \
  device="$DEVICE" \
  project=runs/detect/basket_mannequin_detect \
  name="$RUN_NAME" \
  exist_ok=True \
  degrees=25 \
  fliplr=0.5 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  scale=0.5 \
  mosaic=1.0 \
  patience=20 \
  plots=False \
  "$@"
