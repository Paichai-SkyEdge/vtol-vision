#!/usr/bin/env python3
"""
Prepare a clean YOLO detection dataset from the Roboflow skyedge YOLO11 export.

The source export contains mixed image extensions and mixed box/segmentation
YOLO labels. This script normalizes all images to JPG and all labels to
YOLO-detect 5-column boxes.
"""

import argparse
import hashlib
import re
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageOps

try:
    from pillow_heif import register_heif_opener
except ImportError:  # pragma: no cover - optional dependency for HEIC inputs
    register_heif_opener = None


SPLITS = ("train", "valid", "test")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare clean SkyEdge YOLO11 dataset")
    parser.add_argument(
        "--src",
        default="/media/dev/WORKOUT/skyedge yolo11 image.yolov11 (1)",
        help="Roboflow YOLO dataset root",
    )
    parser.add_argument(
        "--out",
        default="datasets/skyedge_yolo11_clean",
        help="Output YOLO dataset root",
    )
    parser.add_argument(
        "--flatten",
        action="store_true",
        help="Also write all/images and all/labels with every sample.",
    )
    return parser.parse_args()


def safe_stem(path: Path, split: str, index: int) -> str:
    stem = path.stem
    stem = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{split}_{index:05d}_{stem}_{digest}"


def convert_label_line(line: str, label_path: Path) -> str | None:
    parts = line.split()
    if not parts:
        return None
    if len(parts) < 5:
        raise ValueError(f"invalid YOLO label in {label_path}: {line!r}")

    cls = int(float(parts[0]))
    nums = [float(v) for v in parts[1:]]

    if len(nums) == 4:
        cx, cy, width, height = nums
    else:
        if len(nums) % 2 != 0:
            raise ValueError(f"invalid polygon coordinate count in {label_path}: {line!r}")
        xs = nums[0::2]
        ys = nums[1::2]
        xmin = max(0.0, min(xs))
        xmax = min(1.0, max(xs))
        ymin = max(0.0, min(ys))
        ymax = min(1.0, max(ys))
        width = xmax - xmin
        height = ymax - ymin
        if width <= 0.0 or height <= 0.0:
            return None
        cx = xmin + width / 2.0
        cy = ymin + height / 2.0

    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    width = min(1.0, max(0.0, width))
    height = min(1.0, max(0.0, height))
    if width <= 0.0 or height <= 0.0:
        return None
    return f"{cls} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}"


def read_clean_labels(label_path: Path) -> list[str]:
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        converted = convert_label_line(line.strip(), label_path)
        if converted:
            labels.append(converted)
    return labels


def write_jpg(src: Path, dst: Path):
    image = Image.open(src)
    image = ImageOps.exif_transpose(image)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    elif image.mode == "L":
        image = image.convert("RGB")
    image.save(dst, "JPEG", quality=95)


def reset_output(out_dir: Path, flatten: bool):
    targets = [(split, kind) for split in SPLITS for kind in ("images", "labels")]
    if flatten:
        targets.extend(("all", kind) for kind in ("images", "labels"))
    for split, kind in targets:
        target = out_dir / split / kind
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)


def load_names(src_dir: Path) -> list[str]:
    data = yaml.safe_load((src_dir / "data.yaml").read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def write_yaml(out_dir: Path, names: list[str]):
    name_lines = "\n".join(f"  {i}: {name}" for i, name in enumerate(names))
    (out_dir / "data.yaml").write_text(
        f"""path: {out_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: {len(names)}
names:
{name_lines}
""",
        encoding="utf-8",
    )
    (out_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)
    if register_heif_opener is not None:
        register_heif_opener()

    if not src_dir.exists():
        raise FileNotFoundError(f"source dataset not found: {src_dir}")

    names = load_names(src_dir)
    reset_output(out_dir, args.flatten)
    write_yaml(out_dir, names)

    stats = {split: {"images": 0, "labels": 0, "objects": 0} for split in SPLITS}
    skipped = []

    for split in SPLITS:
        image_dir = src_dir / split / "images"
        label_dir = src_dir / split / "labels"
        images = sorted(p for p in image_dir.iterdir() if p.is_file())
        for index, image_path in enumerate(images, start=1):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                skipped.append(f"missing label: {image_path}")
                continue

            labels = read_clean_labels(label_path)
            stem = safe_stem(image_path, split, index)
            out_image = out_dir / split / "images" / f"{stem}.jpg"
            out_label = out_dir / split / "labels" / f"{stem}.txt"
            write_jpg(image_path, out_image)
            label_text = "\n".join(labels)
            if label_text:
                label_text += "\n"
            out_label.write_text(label_text, encoding="utf-8")

            if args.flatten:
                shutil.copy2(out_image, out_dir / "all" / "images" / out_image.name)
                shutil.copy2(out_label, out_dir / "all" / "labels" / out_label.name)

            stats[split]["images"] += 1
            stats[split]["labels"] += 1
            stats[split]["objects"] += len(labels)

    for split in SPLITS:
        print(
            f"{split}: {stats[split]['images']} images, "
            f"{stats[split]['labels']} labels, {stats[split]['objects']} objects"
        )
    print(f"yaml: {out_dir / 'data.yaml'}")
    if skipped:
        print(f"skipped: {len(skipped)}")
        for item in skipped[:20]:
            print(f"  {item}")


if __name__ == "__main__":
    main()
