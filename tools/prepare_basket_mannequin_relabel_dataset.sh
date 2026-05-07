#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"
python3 tools/prepare_basket_mannequin_dataset.py \
  --src images/skyedge_vision_relabel \
  --out datasets/basket_mannequin_relabel \
  --allow-missing
