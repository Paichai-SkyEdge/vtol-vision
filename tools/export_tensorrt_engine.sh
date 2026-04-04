#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${1:-/home/dev/vtol-vision/runs/detect/mannequin_detect/yolo11n_v1_gpu/weights/best.onnx}"
ENGINE_PATH="${2:-${MODEL_PATH%.onnx}.engine}"
FP16_FLAG="${FP16_FLAG:---fp16}"
WORKSPACE_MB="${WORKSPACE_MB:-1024}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi

if ! command -v trtexec >/dev/null 2>&1; then
  echo "trtexec not found. Install TensorRT first, then rerun on the target machine." >&2
  exit 2
fi

echo "building TensorRT engine"
echo "  model:  $MODEL_PATH"
echo "  engine: $ENGINE_PATH"
echo "  flags:  $FP16_FLAG --workspace=${WORKSPACE_MB}"

trtexec \
  --onnx="$MODEL_PATH" \
  --saveEngine="$ENGINE_PATH" \
  $FP16_FLAG \
  --workspace="$WORKSPACE_MB"

echo "done: $ENGINE_PATH"
