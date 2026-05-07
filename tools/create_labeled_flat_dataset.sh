#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_root="${1:-$repo_root/datasets/basket_mannequin_final}"
out_root="${2:-$repo_root/datasets/basket_mannequin_labeled_flat}"

rm -rf "$out_root"
mkdir -p "$out_root"

for split in train val; do
  label_dir="$src_root/$split/labels"
  image_dir="$src_root/$split/images"
  [[ -d "$label_dir" && -d "$image_dir" ]] || continue

  for label in "$label_dir"/*.txt; do
    [[ -f "$label" && -s "$label" ]] || continue
    stem="$(basename "$label" .txt)"
    image="$image_dir/$stem.jpg"
    [[ -f "$image" ]] || continue

    cp "$image" "$out_root/$stem.jpg"
    cp "$label" "$out_root/$stem.txt"
  done
done

printf 'basket\nmannequin\n' > "$out_root/classes.txt"

cat > "$out_root/README.md" <<'EOF'
# Basket + Mannequin Labeled Flat Dataset

One flat directory containing only labeled image/YOLO-label pairs.

Class order:

```txt
0 basket
1 mannequin
```

Each usable sample has:

```txt
<stem>.jpg
<stem>.txt
```

YOLO label format:

```txt
class_id center_x center_y width height
```

Coordinates are normalized to `[0, 1]`.
EOF

python3 - "$out_root" <<'PY'
from pathlib import Path
from collections import Counter
import sys

root = Path(sys.argv[1])
images = sorted(root.glob("*.jpg"))
labels = sorted(p for p in root.glob("*.txt") if p.name != "classes.txt")

missing_labels = [p for p in images if not (root / f"{p.stem}.txt").exists()]
missing_images = [p for p in labels if not (root / f"{p.stem}.jpg").exists()]

counts = Counter()
for label in labels:
    for line in label.read_text().splitlines():
        if line.strip():
            counts[line.split()[0]] += 1

print(f"Created: {root}")
print(f"  images: {len(images)}")
print(f"  labels: {len(labels)}")
print(f"  missing labels: {len(missing_labels)}")
print(f"  missing images: {len(missing_images)}")
print(f"  basket: {counts['0']}")
print(f"  mannequin: {counts['1']}")

if missing_labels or missing_images:
    raise SystemExit(1)
PY
