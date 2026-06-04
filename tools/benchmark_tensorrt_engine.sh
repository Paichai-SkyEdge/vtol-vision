#!/usr/bin/env bash
set -euo pipefail

ENGINE_PATH="${1:-weights/mannequin_yolo11n/best.engine}"
SHAPES="${SHAPES:-images:1x3x640x640}"
ITERATIONS="${ITERATIONS:-200}"
WARMUP="${WARMUP:-50}"

if [[ ! -f "$ENGINE_PATH" ]]; then
  echo "engine not found: $ENGINE_PATH" >&2
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT first, then rerun on the target machine." >&2
  exit 2
fi

echo "benchmarking TensorRT engine"
echo "  engine:     $ENGINE_PATH"
echo "  shapes:     $SHAPES"
echo "  warmup:     $WARMUP"
echo "  iterations: $ITERATIONS"

trtexec \
  --loadEngine="$ENGINE_PATH" \
  --shapes="$SHAPES" \
  --warmUp="$WARMUP" \
  --iterations="$ITERATIONS" \
  --useSpinWait
