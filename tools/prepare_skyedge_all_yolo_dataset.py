#!/usr/bin/env python3
"""
Merge all local labeled data into one YOLO detection dataset.

Output classes:
  0 basket      (also maps skyedge_box)
  1 mannequin   (also maps skyedge_person)

Inputs include existing YOLO datasets, flat YOLO labels, and Pascal VOC XML
annotations. Duplicate images are detected by file hash; when duplicates exist,
the sample with more labels is kept.
"""

import argparse
import hashlib
import random
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


CLASSES = ["basket", "mannequin"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASSES)}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".heic"}


@dataclass
class Sample:
    image: Path
    labels: list[str]
    source: str
    group: str


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare combined SkyEdge YOLO dataset")
    parser.add_argument("--out", default="datasets/skyedge_all_yolo")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    return parser.parse_args()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_stem(path: Path, source: str, index: int) -> str:
    stem = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", path.stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").lower()
    src = re.sub(r"[^A-Za-z0-9]+", "_", source).strip("_").lower()
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]
    return f"{src}_{index:05d}_{stem}_{digest}"


def group_key(path: Path) -> str:
    stem = path.stem
    for token in ("_aug", "_geo"):
        if token in stem:
            return stem.split(token, 1)[0]
    return re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)


def normalize_box(cls: int, values: list[float]) -> str | None:
    if len(values) == 4:
        cx, cy, width, height = values
    else:
        if len(values) % 2 != 0:
            raise ValueError(f"invalid polygon coordinate count: {len(values)}")
        xs = values[0::2]
        ys = values[1::2]
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


def read_yolo_labels(label_path: Path, id_map: dict[int, int]) -> list[str]:
    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        src_cls = int(float(parts[0]))
        if src_cls not in id_map:
            raise ValueError(f"unknown class id {src_cls} in {label_path}")
        converted = normalize_box(id_map[src_cls], [float(v) for v in parts[1:]])
        if converted:
            labels.append(converted)
    return labels


def read_voc_labels(xml_path: Path) -> list[str]:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width"))
    height = float(size.findtext("height"))
    labels = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if name not in CLASS_TO_ID:
            raise ValueError(f"unknown class {name!r} in {xml_path}")
        box = obj.find("bndbox")
        xmin = max(0.0, min(float(box.findtext("xmin")), width))
        ymin = max(0.0, min(float(box.findtext("ymin")), height))
        xmax = max(0.0, min(float(box.findtext("xmax")), width))
        ymax = max(0.0, min(float(box.findtext("ymax")), height))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = ((xmin + xmax) / 2.0) / width
        cy = ((ymin + ymax) / 2.0) / height
        bw = (xmax - xmin) / width
        bh = (ymax - ymin) / height
        labels.append(f"{CLASS_TO_ID[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return labels


def collect_yolo_split(root: Path, split: str, source: str, id_map: dict[int, int]) -> list[Sample]:
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    if not image_dir.exists():
        return []

    samples = []
    for image in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        label = label_dir / f"{image.stem}.txt"
        if not label.exists():
            continue
        samples.append(Sample(image, read_yolo_labels(label, id_map), source, group_key(image)))
    return samples


def collect_flat_yolo(root: Path, source: str, id_map: dict[int, int]) -> list[Sample]:
    if not root.exists():
        return []
    samples = []
    for image in sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        label = image.with_suffix(".txt")
        if not label.exists():
            continue
        samples.append(Sample(image, read_yolo_labels(label, id_map), source, group_key(image)))
    return samples


def collect_voc(root: Path, source: str) -> list[Sample]:
    if not root.exists():
        return []
    samples = []
    for image in sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS):
        xml = image.with_suffix(".xml")
        if not xml.exists():
            continue
        labels = read_voc_labels(xml)
        if not labels:
            continue
        samples.append(Sample(image, labels, source, group_key(image)))
    return samples


def collect_all() -> tuple[list[Sample], Counter, int]:
    candidates = []
    basket_map = {0: CLASS_TO_ID["basket"], 1: CLASS_TO_ID["mannequin"]}
    skyedge_map = {0: CLASS_TO_ID["basket"], 1: CLASS_TO_ID["mannequin"]}

    for split in ("train", "val"):
        candidates.extend(collect_yolo_split(Path("datasets/basket_mannequin"), split, "basket_mannequin", basket_map))
        candidates.extend(
            collect_yolo_split(Path("datasets/basket_mannequin_final"), split, "basket_mannequin_final", basket_map)
        )
    for split in ("train", "valid", "test"):
        candidates.extend(collect_yolo_split(Path("datasets/skyedge_yolo11_clean"), split, "skyedge_yolo11_clean", skyedge_map))

    candidates.extend(collect_flat_yolo(Path("datasets/basket_mannequin_labeled_flat"), "basket_mannequin_labeled_flat", basket_map))
    candidates.extend(collect_voc(Path("images/skyedge_vision"), "skyedge_vision_voc"))
    candidates.extend(collect_voc(Path("images/skyedge_vision_aug"), "skyedge_vision_aug_voc"))
    candidates.extend(collect_voc(Path("images/skyedge_vision_relabel"), "skyedge_vision_relabel_voc"))

    by_hash: dict[str, Sample] = {}
    source_counts = Counter()
    duplicates = 0
    for sample in candidates:
        digest = file_hash(sample.image)
        current = by_hash.get(digest)
        if current is None or len(sample.labels) > len(current.labels):
            by_hash[digest] = sample
        if current is not None:
            duplicates += 1

    for sample in by_hash.values():
        source_counts[sample.source] += 1
    return list(by_hash.values()), source_counts, duplicates


def reset_output(out_dir: Path):
    for split in ("train", "valid", "test", "all"):
        for kind in ("images", "labels"):
            target = out_dir / split / kind
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)


def write_yaml(out_dir: Path):
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASSES))
    (out_dir / "data.yaml").write_text(
        f"""path: {out_dir.resolve()}
train: train/images
val: valid/images
test: test/images

nc: {len(CLASSES)}
names:
{names}
""",
        encoding="utf-8",
    )
    (out_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")


def split_samples(samples: list[Sample], seed: int, val_ratio: float, test_ratio: float) -> dict[str, list[Sample]]:
    by_group = defaultdict(list)
    for sample in samples:
        by_group[sample.group].append(sample)

    groups = list(by_group)
    random.Random(seed).shuffle(groups)
    test_count = max(1, round(len(groups) * test_ratio))
    val_count = max(1, round(len(groups) * val_ratio))
    test_groups = set(groups[:test_count])
    val_groups = set(groups[test_count : test_count + val_count])

    result = {"train": [], "valid": [], "test": []}
    for group, group_samples in by_group.items():
        if group in test_groups:
            result["test"].extend(group_samples)
        elif group in val_groups:
            result["valid"].extend(group_samples)
        else:
            result["train"].extend(group_samples)
    return result


def copy_sample(out_dir: Path, split: str, sample: Sample, index: int):
    stem = safe_stem(sample.image, sample.source, index)
    image_out = out_dir / split / "images" / f"{stem}{sample.image.suffix.lower()}"
    label_out = out_dir / split / "labels" / f"{stem}.txt"
    shutil.copy2(sample.image, image_out)
    label_text = "\n".join(sample.labels)
    if label_text:
        label_text += "\n"
    label_out.write_text(label_text, encoding="utf-8")
    shutil.copy2(image_out, out_dir / "all" / "images" / image_out.name)
    shutil.copy2(label_out, out_dir / "all" / "labels" / label_out.name)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    samples, source_counts, duplicates = collect_all()
    if not samples:
        raise RuntimeError("no labeled samples found")

    splits = split_samples(samples, args.seed, args.val_ratio, args.test_ratio)
    reset_output(out_dir)
    write_yaml(out_dir)

    class_counts = Counter()
    for split, samples_in_split in splits.items():
        for index, sample in enumerate(sorted(samples_in_split, key=lambda s: str(s.image)), start=1):
            copy_sample(out_dir, split, sample, index)
            for label in sample.labels:
                class_counts[CLASSES[int(label.split()[0])]] += 1

    print(f"samples: {len(samples)} unique, {duplicates} duplicates removed")
    for split in ("train", "valid", "test"):
        print(f"{split}: {len(splits[split])} images")
    print("sources:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")
    print("objects:")
    for cls in CLASSES:
        print(f"  {cls}: {class_counts[cls]}")
    print(f"yaml: {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
