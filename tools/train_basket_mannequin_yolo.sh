#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"
cmd="python3 tools/train_basket_mannequin_yolo.py"
for arg in "$@"; do
  printf -v quoted ' %q' "$arg"
  cmd+="$quoted"
done

exec nix-shell -p python3Packages.ultralytics --run "$cmd"
