#!/usr/bin/env python3
"""
Convert labelImg Pascal VOC XML annotations into a YOLO train/val dataset.

Input:
  images/skyedge_vision/*.jpg
  images/skyedge_vision/*.xml

Output:
  datasets/basket_mannequin/{train,val}/{images,labels}
  datasets/basket_mannequin/dataset.yaml
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = ["basket", "mannequin"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare basket/mannequin YOLO dataset")
    parser.add_argument("--src", default="images/skyedge_vision", help="Image/XML directory")
    parser.add_argument("--out", default="datasets/basket_mannequin", help="Output dataset dir")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip images that do not have a matching XML annotation.",
    )
    return parser.parse_args()


def voc_to_yolo(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))

    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASSES:
            raise ValueError(f"unknown class {name!r} in {xml_path}")

        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))

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

    return labels


def reset_split(out_dir: Path):
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            target = out_dir / split / kind
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def write_dataset_yaml(out_dir: Path):
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


def main():
    args = parse_args()
    src_dir = Path(args.src)
    out_dir = Path(args.out)

    if not src_dir.exists():
      raise FileNotFoundError(f"source directory not found: {src_dir}")

    samples = []
    for image_path in sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():
            if args.allow_missing:
                continue
            raise FileNotFoundError(f"missing annotation for {image_path}: {xml_path}")
        labels = voc_to_yolo(xml_path)
        if not labels:
            raise ValueError(f"no valid labels in {xml_path}")
        samples.append((image_path, labels))

    if len(samples) < 2:
        raise RuntimeError("need at least 2 labeled images to create train/val splits")

    random.Random(args.seed).shuffle(samples)
    val_count = max(1, round(len(samples) * args.val_ratio))
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    reset_split(out_dir)
    write_dataset_yaml(out_dir)

    for split, split_samples in (("train", train_samples), ("val", val_samples)):
        for image_path, labels in split_samples:
            shutil.copy2(image_path, out_dir / split / "images" / image_path.name)
            label_path = out_dir / split / "labels" / f"{image_path.stem}.txt"
            label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")

    print(f"Prepared {len(samples)} images")
    print(f"  train: {len(train_samples)}")
    print(f"  val  : {len(val_samples)}")
    print(f"  yaml : {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
