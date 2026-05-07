#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src_dataset="$repo_root/datasets/basket_mannequin_final"
bundle_root="${1:-$repo_root/training_bundle/basket_mannequin_yolo}"

if [[ ! -d "$src_dataset/train/images" || ! -d "$src_dataset/val/images" ]]; then
  echo "missing source dataset: $src_dataset" >&2
  echo "run: python3 tools/prepare_basket_mannequin_final_dataset.py" >&2
  exit 1
fi

rm -rf "$bundle_root"
mkdir -p "$bundle_root/dataset" "$bundle_root/scripts" "$bundle_root/config"

cp -a "$src_dataset/train" "$bundle_root/dataset/"
cp -a "$src_dataset/val" "$bundle_root/dataset/"
cp -a "$src_dataset/classes.txt" "$bundle_root/dataset/classes.txt"
cp -a "$repo_root/config/basket_mannequin_class_map.yaml" "$bundle_root/config/class_map.yaml"
find "$bundle_root/dataset" -name '*.cache' -delete

cat > "$bundle_root/dataset.yaml" <<'YAML'
path: dataset
train: train/images
val: val/images

nc: 2
names:
  0: basket
  1: mannequin
YAML

cat > "$bundle_root/dataset/dataset.yaml" <<'YAML'
path: .
train: train/images
val: val/images

nc: 2
names:
  0: basket
  1: mannequin
YAML

cat > "$bundle_root/requirements.txt" <<'REQ'
ultralytics>=8.3.0
REQ

cat > "$bundle_root/scripts/train.sh" <<'SH'
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
SH

cat > "$bundle_root/scripts/export_engine.sh" <<'SH'
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
SH

cat > "$bundle_root/scripts/export_onnx.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

if [[ -f .venv-yolo/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv-yolo/bin/activate
fi

MODEL_PATH="${MODEL_PATH:-runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt}"
IMG_SIZE="${IMG_SIZE:-640}"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "model not found: $MODEL_PATH" >&2
  exit 1
fi

yolo export \
  model="$MODEL_PATH" \
  format=onnx \
  imgsz="$IMG_SIZE" \
  simplify=True
SH

cat > "$bundle_root/README.md" <<'MD'
# Basket + Mannequin YOLO Training Bundle

This folder is self-contained. Copy it to a GPU machine or Jetson and train from here.

## Dataset

- Classes: `0 basket`, `1 mannequin`
- Dataset YAML: `dataset.yaml`
- ROS class map after export: `config/class_map.yaml`

## Train

```bash
cd basket_mannequin_yolo
DEVICE=0 BATCH=8 EPOCHS=100 scripts/train.sh
```

For a quick check:

```bash
DEVICE=cpu BATCH=4 EPOCHS=1 RUN_NAME=smoke scripts/train.sh
```

The best checkpoint is written to:

```txt
runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt
```

## Export

Create TensorRT engine on the Jetson or target GPU machine:

```bash
scripts/export_engine.sh
```

Create ONNX:

```bash
scripts/export_onnx.sh
```

Override the model path if needed:

```bash
MODEL_PATH=runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt scripts/export_engine.sh
```

TensorRT `.engine` files are hardware/JetPack specific. Export them on the machine that will run inference.
MD

chmod +x "$bundle_root/scripts/"*.sh

train_count="$(find "$bundle_root/dataset/train/images" -type f | wc -l)"
val_count="$(find "$bundle_root/dataset/val/images" -type f | wc -l)"
archive_path="${bundle_root}.tar.gz"
tar -C "$(dirname "$bundle_root")" -czf "$archive_path" "$(basename "$bundle_root")"

echo "Created training bundle:"
echo "  dir     : $bundle_root"
echo "  archive : $archive_path"
echo "  train   : $train_count images"
echo "  val     : $val_count images"
