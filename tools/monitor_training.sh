#!/usr/bin/env bash

set -euo pipefail

RUN_NAME="${1:-yolo11n_v1_gpu}"
INTERVAL="${2:-5}"
RUN_DIR="runs/detect/mannequin_detect/${RUN_NAME}"
RESULTS_CSV="${RUN_DIR}/results.csv"
BEST_PT="${RUN_DIR}/weights/best.pt"
LAST_PT="${RUN_DIR}/weights/last.pt"

print_header() {
  echo "YOLO Training Monitor"
  echo "run      : ${RUN_NAME}"
  echo "interval : ${INTERVAL}s"
  echo "results  : ${RESULTS_CSV}"
  echo
}

print_run_status() {
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "results.csv not found yet."
    return
  fi

  local line
  line="$(tail -n 1 "${RESULTS_CSV}")"

  awk -F',' -v row="${line}" '
    BEGIN {
      split(row, a, ",")
      if (length(a) < 15) {
        print "results.csv format not ready yet."
        exit
      }
      printf "epoch        : %s\n", a[1]
      printf "elapsed_sec  : %s\n", a[2]
      printf "train losses : box=%s cls=%s dfl=%s\n", a[3], a[4], a[5]
      printf "val losses   : box=%s cls=%s dfl=%s\n", a[10], a[11], a[12]
      printf "metrics      : P=%s R=%s mAP50=%s mAP50-95=%s\n", a[6], a[7], a[8], a[9]
      printf "lr           : pg0=%s pg1=%s pg2=%s\n", a[13], a[14], a[15]
    }
  '

  if [[ -f "${BEST_PT}" ]]; then
    echo "best.pt      : present"
  else
    echo "best.pt      : missing"
  fi

  if [[ -f "${LAST_PT}" ]]; then
    echo "last.pt      : present"
  else
    echo "last.pt      : missing"
  fi
}

print_process_status() {
  local proc
  proc="$(ps -eo pid,etime,%cpu,%mem,cmd | rg "tools/train_mannequin_yolo.py|ultralytics" || true)"

  echo
  echo "process:"
  if [[ -n "${proc}" ]]; then
    echo "${proc}"
  else
    echo "training process not found in current shell namespace."
  fi
}

print_gpu_status() {
  echo
  echo "gpu:"
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi --query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null; then
      :
    else
      echo "nvidia-smi is installed but GPU status is not accessible in this session."
    fi
  else
    echo "nvidia-smi not found."
  fi
}

while true; do
  clear
  print_header
  print_run_status
  print_process_status
  print_gpu_status
  sleep "${INTERVAL}"
done
