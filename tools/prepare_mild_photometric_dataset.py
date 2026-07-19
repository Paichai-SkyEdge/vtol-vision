#!/usr/bin/env python3
"""Add conservative brightness/saturation variants to train only.

The source dataset is never modified. Unchanged files are hard-linked, and a
separate held-out photometric test split is generated with an independent RNG.
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from prepare_shadow_augmented_dataset import IMAGE_EXTS, resize_for_augmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Create a mild photometric YOLO experiment dataset")
    parser.add_argument("--src", default="datasets/skyedge_all_yolo_shadow")
    parser.add_argument("--out", default="datasets/skyedge_all_yolo_shadow_mild")
    parser.add_argument("--fraction", type=float, default=0.2)
    parser.add_argument("--brightness", type=float, default=0.1, help="Maximum V multiplier delta")
    parser.add_argument("--saturation", type=float, default=0.1, help="Maximum S multiplier delta")
    parser.add_argument("--seed", type=int, default=142)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def adjust_mild(image, rng, brightness_delta, saturation_delta):
    brightness = rng.uniform(1.0 - brightness_delta, 1.0 + brightness_delta)
    saturation = rng.uniform(1.0 - saturation_delta, 1.0 + saturation_delta)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return result, {"brightness": round(brightness, 4), "saturation": round(saturation, 4)}


def hardlink_split(source, output, split):
    count = 0
    for kind in ("images", "labels"):
        directory = source / split / kind
        if not directory.exists():
            continue
        destination = output / split / kind
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(p for p in directory.iterdir() if p.is_file()):
            os.link(path.resolve(), destination / path.name)
            if kind == "images":
                count += 1
    return count


def write_yaml(output):
    class_block = "nc: 2\nnames:\n  0: basket\n  1: mannequin\n"
    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\ntrain: train/images\nval: valid/images\ntest: test/images\n{class_block}",
        encoding="utf-8",
    )
    (output / "photometric_test.yaml").write_text(
        f"path: {output.resolve()}\ntrain: photometric_test/images\nval: photometric_test/images\n"
        f"test: photometric_test/images\n{class_block}",
        encoding="utf-8",
    )


def main():
    args = parse_args()
    if not 0.0 <= args.fraction <= 1.0:
        raise ValueError("--fraction must be between 0 and 1")
    if not 0.0 <= args.brightness <= 0.2 or not 0.0 <= args.saturation <= 0.2:
        raise ValueError("brightness and saturation deltas must be between 0 and 0.2")

    source = Path(args.src)
    output = Path(args.out)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)

    counts = {split: hardlink_split(source, output, split) for split in ("train", "valid", "test")}
    originals = sorted(
        p for p in (source / "train" / "images").iterdir()
        if p.suffix.lower() in IMAGE_EXTS and "_shadow" not in p.stem and "_mild" not in p.stem
    )
    train_rng = random.Random(args.seed)
    selected = train_rng.sample(originals, round(len(originals) * args.fraction))
    manifest = []
    for image_path in selected:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = resize_for_augmentation(image, args.max_side)
        augmented, metadata = adjust_mild(image, train_rng, args.brightness, args.saturation)
        stem = f"{image_path.stem}_mild00"
        destination = output / "train" / "images" / f"{stem}.jpg"
        cv2.imwrite(str(destination), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        shutil.copy2(source / "train" / "labels" / f"{image_path.stem}.txt", output / "train" / "labels" / f"{stem}.txt")
        manifest.append({"source": image_path.name, "output": destination.name, **metadata})

    test_rng = random.Random(args.seed + 10_000)
    stress_dir = output / "photometric_test"
    (stress_dir / "images").mkdir(parents=True)
    (stress_dir / "labels").mkdir(parents=True)
    stress_manifest = []
    for image_path in sorted((source / "test" / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = resize_for_augmentation(image, args.max_side)
        augmented, metadata = adjust_mild(image, test_rng, args.brightness, args.saturation)
        stem = f"{image_path.stem}_mildtest"
        cv2.imwrite(str(stress_dir / "images" / f"{stem}.jpg"), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        shutil.copy2(source / "test" / "labels" / f"{image_path.stem}.txt", stress_dir / "labels" / f"{stem}.txt")
        stress_manifest.append({"source": image_path.name, **metadata})

    write_yaml(output)
    summary = {
        "source": str(source), "seed": args.seed, "fraction": args.fraction,
        "brightness_delta": args.brightness, "saturation_delta": args.saturation,
        "original_counts": counts, "train_variants": len(manifest),
        "train_total": counts["train"] + len(manifest), "photometric_test": len(stress_manifest),
        "augmentations": manifest, "test_augmentations": stress_manifest,
    }
    (output / "photometric_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith("augmentations")}, indent=2))


if __name__ == "__main__":
    main()
