#!/usr/bin/env python3
"""Build a controlled shadow-robustness training dataset from a YOLO dataset.

Original train images are retained and shadow variants are added only to train.
Validation and test files are linked or copied unchanged, so their metrics stay
comparable with the baseline model.
"""

import argparse
import json
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create object-aware shadow augmentations for YOLO")
    parser.add_argument("--src", default="datasets/skyedge_all_yolo")
    parser.add_argument("--out", default="datasets/skyedge_all_yolo_shadow")
    parser.add_argument("--fraction", type=float, default=0.4, help="Fraction of train images to augment")
    parser.add_argument("--copies", type=int, default=1, help="Shadow variants per selected image")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1600,
        help="Resize shadow variants so their longest side is at most this value; 0 disables resizing",
    )
    parser.add_argument("--copy", action="store_true", help="Copy unchanged files instead of hard-linking them")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_boxes(label_path: Path, width: int, height: int):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _, cx, cy, bw, bh = map(float, parts)
        boxes.append(
            (
                (cx - bw / 2.0) * width,
                (cy - bh / 2.0) * height,
                (cx + bw / 2.0) * width,
                (cy + bh / 2.0) * height,
            )
        )
    return boxes


def rotated_band_mask(shape, box, rng, crossing):
    height, width = shape[:2]
    x1, y1, x2, y2 = box
    object_w = max(2.0, x2 - x1)
    object_h = max(2.0, y2 - y1)

    if crossing:
        side = rng.randrange(4)
        if side == 0:
            center = (x1, rng.uniform(y1, y2))
        elif side == 1:
            center = (x2, rng.uniform(y1, y2))
        elif side == 2:
            center = (rng.uniform(x1, x2), y1)
        else:
            center = (rng.uniform(x1, x2), y2)
    else:
        center = (rng.uniform(x1, x2), rng.uniform(y1, y2))

    angle = rng.uniform(0.0, math.pi)
    direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    half_length = math.hypot(width, height)
    half_width = rng.uniform(0.12, 0.42) * max(object_w, object_h)
    center_vec = np.array(center, dtype=np.float32)
    polygon = np.array(
        [
            center_vec - direction * half_length - normal * half_width,
            center_vec + direction * half_length - normal * half_width,
            center_vec + direction * half_length + normal * half_width,
            center_vec - direction * half_length + normal * half_width,
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def add_leaf_gaps(mask, rng):
    height, width = mask.shape
    if rng.random() >= 0.35:
        return mask
    result = mask.copy()
    for _ in range(rng.randint(4, 12)):
        center = (rng.randrange(width), rng.randrange(height))
        axes = (
            rng.randint(max(4, width // 100), max(5, width // 25)),
            rng.randint(max(4, height // 100), max(5, height // 25)),
        )
        cv2.ellipse(result, center, axes, rng.uniform(0, 180), 0, 360, 0, -1)
    return result


def apply_shadow(image, boxes, rng):
    target = rng.choice(boxes)
    crossing = rng.random() < 0.7
    mask = rotated_band_mask(image.shape, target, rng, crossing)
    mask = add_leaf_gaps(mask, rng)

    sigma = rng.uniform(0.008, 0.03) * max(image.shape[:2])
    soft_mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, None]

    darkness = rng.uniform(0.38, 0.72)
    color_scale = np.array(
        [darkness * rng.uniform(0.92, 1.04), darkness, darkness * rng.uniform(0.86, 1.0)],
        dtype=np.float32,
    )
    shadowed = image.astype(np.float32) * color_scale
    output = image.astype(np.float32) * (1.0 - alpha) + shadowed * alpha
    metadata = {
        "mode": "boundary_crossing" if crossing else "interior",
        "darkness": round(darkness, 4),
        "blur_sigma": round(sigma, 4),
    }
    return np.clip(output, 0, 255).astype(np.uint8), metadata


def resize_for_augmentation(image, max_side):
    height, width = image.shape[:2]
    if max_side <= 0 or max(height, width) <= max_side:
        return image
    scale = max_side / float(max(height, width))
    return cv2.resize(
        image,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def link_or_copy(source: Path, destination: Path, copy_files: bool):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_files:
        shutil.copy2(source, destination)
    else:
        os.link(source.resolve(), destination)


def copy_split(source_root: Path, output_root: Path, split: str, copy_files: bool):
    image_dir = source_root / split / "images"
    label_dir = source_root / split / "labels"
    if not image_dir.exists():
        return 0
    count = 0
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            raise FileNotFoundError(f"missing label: {label_path}")
        link_or_copy(image_path, output_root / split / "images" / image_path.name, copy_files)
        link_or_copy(label_path, output_root / split / "labels" / label_path.name, copy_files)
        count += 1
    return count


def write_yaml(output_root: Path):
    (output_root / "data.yaml").write_text(
        f"""path: {output_root.resolve()}
train: train/images
val: valid/images
test: test/images

nc: 2
names:
  0: basket
  1: mannequin
""",
        encoding="utf-8",
    )
    (output_root / "classes.txt").write_text("basket\nmannequin\n", encoding="utf-8")


def main():
    args = parse_args()
    if not 0.0 <= args.fraction <= 1.0:
        raise ValueError("--fraction must be between 0 and 1")
    if args.copies < 1:
        raise ValueError("--copies must be at least 1")

    source_root = Path(args.src)
    output_root = Path(args.out)
    if not (source_root / "train" / "images").exists():
        raise FileNotFoundError(f"invalid YOLO dataset: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output_root}")
        shutil.rmtree(output_root)

    rng = random.Random(args.seed)
    counts = {
        split: copy_split(source_root, output_root, split, args.copy)
        for split in ("train", "valid", "test")
    }
    train_images = sorted(
        p for p in (source_root / "train" / "images").iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    selected_count = round(len(train_images) * args.fraction)
    selected = rng.sample(train_images, selected_count)
    manifest = []

    for image_path in selected:
        label_path = source_root / "train" / "labels" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read: {image_path}")
        image = resize_for_augmentation(image, args.max_side)
        boxes = load_boxes(label_path, image.shape[1], image.shape[0])
        if not boxes:
            continue
        for copy_index in range(args.copies):
            augmented, metadata = apply_shadow(image, boxes, rng)
            stem = f"{image_path.stem}_shadow{copy_index:02d}"
            output_image = output_root / "train" / "images" / f"{stem}.jpg"
            output_label = output_root / "train" / "labels" / f"{stem}.txt"
            if not cv2.imwrite(str(output_image), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 94]):
                raise RuntimeError(f"failed to write: {output_image}")
            shutil.copy2(label_path, output_label)
            manifest.append({"source": image_path.name, "output": output_image.name, **metadata})

    write_yaml(output_root)
    summary = {
        "source": str(source_root),
        "seed": args.seed,
        "fraction": args.fraction,
        "copies": args.copies,
        "max_side": args.max_side,
        "original_counts": counts,
        "shadow_images": len(manifest),
        "train_total": counts["train"] + len(manifest),
        "augmentations": manifest,
    }
    (output_root / "shadow_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "augmentations"}, indent=2))
    print(f"yaml: {output_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
