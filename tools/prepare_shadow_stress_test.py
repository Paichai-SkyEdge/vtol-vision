#!/usr/bin/env python3
"""Create a deterministic shadow-only YOLO evaluation split."""

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2

from prepare_shadow_augmented_dataset import IMAGE_EXTS, apply_shadow, load_boxes, resize_for_augmentation


def main():
    parser = argparse.ArgumentParser(description="Create a held-out shadow stress test")
    parser.add_argument("--src", default="datasets/skyedge_all_yolo")
    parser.add_argument("--out", default="datasets/skyedge_shadow_stress_test")
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = Path(args.src)
    output = Path(args.out)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    (output / "images").mkdir(parents=True)
    (output / "labels").mkdir(parents=True)

    rng = random.Random(args.seed)
    manifest = []
    for image_path in sorted((source / "test" / "images").iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = source / "test" / "labels" / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read: {image_path}")
        image = resize_for_augmentation(image, args.max_side)
        boxes = load_boxes(label_path, image.shape[1], image.shape[0])
        if not boxes:
            continue
        augmented, metadata = apply_shadow(image, boxes, rng)
        destination = output / "images" / f"{image_path.stem}_shadow.jpg"
        cv2.imwrite(str(destination), augmented, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        shutil.copy2(label_path, output / "labels" / f"{image_path.stem}_shadow.txt")
        manifest.append({"source": image_path.name, "output": destination.name, **metadata})

    (output / "data.yaml").write_text(
        f"""path: {output.resolve()}
train: images
val: images
test: images
nc: 2
names:
  0: basket
  1: mannequin
""",
        encoding="utf-8",
    )
    (output / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "images": len(manifest), "augmentations": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"shadow stress images: {len(manifest)}")
    print(f"yaml: {output / 'data.yaml'}")


if __name__ == "__main__":
    main()
