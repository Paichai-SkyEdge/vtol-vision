"""
Merge COCO-format mannequin datasets into a single YOLO detection dataset.

- Dataset1: artefact-detection -> only "artist mannequin"
- Dataset2: mannequin -> all mannequin annotations
- Output: datasets/merged/{train,val}/images + labels + dataset.yaml
"""

import json
import random
import shutil
from pathlib import Path

random.seed(42)

SRC1_IMG = Path("./datasets/train")
SRC1_ANN = SRC1_IMG / "_annotations.coco.json"

SRC2_IMG = Path("./datasets/dataset2/train")
SRC2_ANN = SRC2_IMG / "_annotations.coco.json"

OUT = Path("./datasets/merged")
VAL_RATIO = 0.2
TARGET_CLASS = "mannequin"


def coco_to_yolo(bbox, img_w, img_h):
    x, y, w, h = [float(v) for v in bbox]
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    return cx, cy, w / img_w, h / img_h


def extract_samples(ann_path, img_dir, filter_names=None):
    with open(ann_path, encoding="utf-8") as f:
        coco = json.load(f)

    valid_cat_ids = {
        c["id"] for c in coco["categories"] if filter_names is None or c["name"] in filter_names
    }
    img_info = {img["id"]: img for img in coco["images"]}

    ann_by_img = {}
    for ann in coco["annotations"]:
        if ann["category_id"] not in valid_cat_ids:
            continue
        image_id = ann["image_id"]
        ann_by_img.setdefault(image_id, []).append(ann)

    samples = []
    for image_id, anns in ann_by_img.items():
        info = img_info[image_id]
        img_path = img_dir / info["file_name"]
        if not img_path.exists():
            continue
        boxes = [coco_to_yolo(ann["bbox"], info["width"], info["height"]) for ann in anns]
        samples.append((img_path, boxes))

    return samples


def write_split(samples, split_dir):
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, boxes in samples:
        shutil.copy(img_path, img_dir / img_path.name)
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        with open(lbl_path, "w", encoding="utf-8") as f:
            for cx, cy, w, h in boxes:
                f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main():
    print("Extracting mannequin samples from Dataset1...")
    samples1 = extract_samples(SRC1_ANN, SRC1_IMG, filter_names=["artist mannequin"])
    print(f"  -> {len(samples1)} images")

    print("Extracting mannequin samples from Dataset2...")
    samples2 = extract_samples(SRC2_ANN, SRC2_IMG, filter_names=None)
    print(f"  -> {len(samples2)} images")

    all_samples = samples1 + samples2
    random.shuffle(all_samples)
    print(f"Total: {len(all_samples)} images")

    n_val = max(1, int(len(all_samples) * VAL_RATIO))
    val_samples = all_samples[:n_val]
    train_samples = all_samples[n_val:]
    print(f"Train: {len(train_samples)} / Val: {len(val_samples)}")

    write_split(train_samples, OUT / "train")
    write_split(val_samples, OUT / "val")

    yaml_path = OUT / "dataset.yaml"
    yaml_path.write_text(
        f"""\
path: {OUT.resolve()}
train: train/images
val: val/images

nc: 1
names: ['{TARGET_CLASS}']
""",
        encoding="utf-8",
    )

    print("Done")
    print(f"  Train: {OUT / 'train'}")
    print(f"  Val: {OUT / 'val'}")
    print(f"  YAML: {yaml_path}")


if __name__ == "__main__":
    main()
