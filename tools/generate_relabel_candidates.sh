#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$repo_root"
cmd="python3 tools/generate_relabel_candidates.py"
for arg in "$@"; do
  printf -v quoted ' %q' "$arg"
  cmd+="$quoted"
done

exec nix-shell -p 'python3.withPackages (ps: with ps; [ opencv4 numpy ])' --run "$cmd"
