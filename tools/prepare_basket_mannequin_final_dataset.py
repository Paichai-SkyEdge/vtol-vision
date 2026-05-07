#!/usr/bin/env python3
"""
Build a final YOLO dataset from all usable labeled basket/mannequin images.

Sources used by default:
  - images/skyedge_vision_aug       (photometric augmentations with carried labels)
  - images/skyedge_vision_relabel   (manual labels for geometric candidates)

The train/val split is grouped by original capture stem, so variants such as
*_augNN and *_geoNN do not leak between train and validation.
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


CLASSES = ["basket", "mannequin"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare final basket/mannequin YOLO dataset")
    parser.add_argument(
        "--src",
        action="append",
        default=[],
        help="Source image/XML dir. Can be provided multiple times.",
    )
    parser.add_argument("--out", default="datasets/basket_mannequin_final")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def group_key(stem: str) -> str:
    for token in ("_aug", "_geo"):
        if token in stem:
            return stem.split(token, 1)[0]
    return stem


def voc_to_yolo(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    labels = []
    counts = Counter()
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            raise ValueError(f"unknown class {name!r} in {xml_path}")

        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        xmin = max(0.0, min(xmin, width))
        xmax = max(0.0, min(xmax, width))
        ymin = max(0.0, min(ymin, height))
        ymax = max(0.0, min(ymax, height))
        if xmax <= xmin or ymax <= ymin:
            continue

        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        labels.append(f"{CLASSES.index(name)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        counts[name] += 1

    return labels, counts


def collect_samples(src_dirs):
    samples_by_group = defaultdict(list)
    class_counts = Counter()
    skipped = 0

    for src_dir in src_dirs:
        src = Path(src_dir)
        if not src.exists():
            continue

        for image_path in sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS):
            xml_path = image_path.with_suffix(".xml")
            if not xml_path.exists():
                skipped += 1
                continue

            labels, counts = voc_to_yolo(xml_path)
            if not labels:
                skipped += 1
                continue

            samples_by_group[group_key(image_path.stem)].append((image_path, labels))
            class_counts.update(counts)

    return samples_by_group, class_counts, skipped


def reset_out(out_dir):
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            target = out_dir / split / kind
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def write_yaml(out_dir):
    (out_dir / "dataset.yaml").write_text(
        f"""path: {out_dir.resolve()}
train: train/images
val: val/images

nc: {len(CLASSES)}
names:
  0: basket
  1: mannequin
""",
        encoding="utf-8",
    )
    (out_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def copy_samples(out_dir, split, samples):
    for image_path, labels in samples:
        dst_image = out_dir / split / "images" / image_path.name
        shutil.copy2(image_path, dst_image)
        (out_dir / split / "labels" / f"{image_path.stem}.txt").write_text(
            "\n".join(labels) + "\n",
            encoding="utf-8",
        )


def main():
    args = parse_args()
    src_dirs = args.src or ["images/skyedge_vision_aug", "images/skyedge_vision_relabel"]
    out_dir = Path(args.out)

    samples_by_group, class_counts, skipped = collect_samples(src_dirs)
    groups = list(samples_by_group)
    if len(groups) < 2:
        raise RuntimeError("need at least 2 labeled source groups")

    random.Random(args.seed).shuffle(groups)
    val_group_count = max(1, round(len(groups) * args.val_ratio))
    val_groups = set(groups[:val_group_count])

    train_samples = []
    val_samples = []
    for key, samples in samples_by_group.items():
        if key in val_groups:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

    reset_out(out_dir)
    write_yaml(out_dir)
    copy_samples(out_dir, "train", train_samples)
    copy_samples(out_dir, "val", val_samples)

    print(f"Sources: {', '.join(src_dirs)}")
    print(f"Groups : {len(groups)} total, {len(groups) - len(val_groups)} train, {len(val_groups)} val")
    print(f"Images : {len(train_samples) + len(val_samples)} total")
    print(f"  train: {len(train_samples)}")
    print(f"  val  : {len(val_samples)}")
    print(f"Labels : basket={class_counts['basket']} mannequin={class_counts['mannequin']}")
    print(f"Skipped unlabeled/empty images: {skipped}")
    print(f"YAML   : {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
