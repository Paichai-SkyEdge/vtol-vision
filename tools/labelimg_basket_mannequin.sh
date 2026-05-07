#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"
exec nix-shell -p labelImg --run \
  "labelImg images/skyedge_vision config/basket_mannequin_classes.txt images/skyedge_vision"
