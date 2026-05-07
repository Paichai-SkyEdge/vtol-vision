#!/usr/bin/env python3
"""
Create photometric augmentations for labelImg Pascal VOC annotations.

The augmentations preserve object geometry, so bounding boxes remain valid:
brightness, contrast, saturation, hue, shadows, glare, vignette, blur, and noise.
Images and XML annotations are written to a separate output directory.
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Augment basket/mannequin images")
    parser.add_argument("--src", default="images/skyedge_vision", help="Source image/XML dir")
    parser.add_argument("--out", default="images/skyedge_vision_aug", help="Output dir")
    parser.add_argument("--per-image", type=int, default=8, help="Augmented copies per source image")
    parser.add_argument("--max-side", type=int, default=1600, help="Resize long side; 0 keeps original size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        bnd = obj.find("bndbox")
        boxes.append(
            (
                name,
                float(bnd.findtext("xmin")),
                float(bnd.findtext("ymin")),
                float(bnd.findtext("xmax")),
                float(bnd.findtext("ymax")),
            )
        )
    return tree, root, boxes


def scaled_image(image, boxes, max_side):
    h, w = image.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image, boxes, 1.0

    scale = max_side / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_boxes = [(name, x1 * scale, y1 * scale, x2 * scale, y2 * scale) for name, x1, y1, x2, y2 in boxes]
    return resized, scaled_boxes, scale


def write_xml(template_tree, image_name, image_path, image_shape, boxes, dst_xml: Path):
    root = template_tree.getroot()
    root.find("filename").text = image_name
    path_node = root.find("path")
    if path_node is not None:
        path_node.text = str(image_path.resolve())

    h, w = image_shape[:2]
    size = root.find("size")
    size.find("width").text = str(w)
    size.find("height").text = str(h)

    for obj, box in zip(root.findall("object"), boxes):
        name, x1, y1, x2, y2 = box
        obj.find("name").text = name
        bnd = obj.find("bndbox")
        bnd.find("xmin").text = str(int(round(max(0, min(x1, w - 1)))))
        bnd.find("ymin").text = str(int(round(max(0, min(y1, h - 1)))))
        bnd.find("xmax").text = str(int(round(max(0, min(x2, w - 1)))))
        bnd.find("ymax").text = str(int(round(max(0, min(y2, h - 1)))))

    template_tree.write(dst_xml, encoding="utf-8")


def adjust_hsv(image, rng):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
    hsv[:, :, 1] *= rng.uniform(0.45, 1.65)
    hsv[:, :, 2] *= rng.uniform(0.55, 1.55)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, rng):
    alpha = rng.uniform(0.65, 1.45)
    beta = rng.uniform(-45, 45)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def add_shadow(image, rng):
    h, w = image.shape[:2]
    overlay = image.copy()
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array(
        [
            [rng.randint(-w // 4, w), rng.randint(0, h)],
            [rng.randint(0, w + w // 4), rng.randint(0, h)],
            [rng.randint(0, w + w // 4), rng.randint(0, h)],
            [rng.randint(-w // 4, w), rng.randint(0, h)],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [points], 255)
    darkness = rng.uniform(0.35, 0.75)
    overlay[mask > 0] = (overlay[mask > 0].astype(np.float32) * darkness).astype(np.uint8)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(w, h) * 0.035)
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    return np.clip(image.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha, 0, 255).astype(np.uint8)


def add_glare(image, rng):
    h, w = image.shape[:2]
    overlay = np.zeros_like(image, dtype=np.float32)
    cx = rng.randint(0, w - 1)
    cy = rng.randint(0, h - 1)
    radius = rng.randint(max(24, min(w, h) // 18), max(25, min(w, h) // 5))
    color = np.array([rng.randint(200, 255), rng.randint(210, 255), rng.randint(220, 255)], dtype=np.float32)
    cv2.circle(overlay, (cx, cy), radius, color.tolist(), -1)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=radius * rng.uniform(0.35, 0.7))
    strength = rng.uniform(0.25, 0.65)
    return np.clip(image.astype(np.float32) + overlay * strength, 0, 255).astype(np.uint8)


def add_vignette(image, rng):
    h, w = image.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx = w * rng.uniform(0.35, 0.65)
    cy = h * rng.uniform(0.35, 0.65)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    dist /= max(dist.max(), 1.0)
    strength = rng.uniform(0.15, 0.45)
    mask = 1.0 - strength * dist ** 1.7
    return np.clip(image.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def add_noise_blur(image, rng):
    out = image.astype(np.float32)
    if rng.random() < 0.8:
        out += rng.normalvariate(0, 1) + np.random.default_rng(rng.randint(0, 1_000_000)).normal(
            0, rng.uniform(4, 18), image.shape
        )
    out = np.clip(out, 0, 255).astype(np.uint8)
    if rng.random() < 0.55:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), sigmaX=rng.uniform(0.2, 1.2))
    return out


def add_color_cast(image, rng):
    cast = np.array(
        [rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15), rng.uniform(0.85, 1.15)],
        dtype=np.float32,
    )
    return np.clip(image.astype(np.float32) * cast, 0, 255).astype(np.uint8)


def augment(image, rng):
    out = image.copy()
    steps = [adjust_hsv, adjust_contrast, add_shadow, add_glare, add_vignette, add_noise_blur, add_color_cast]
    rng.shuffle(steps)
    for step in steps:
        if rng.random() < 0.72:
            out = step(out, rng)
    if rng.random() < 0.25:
        quality = rng.randint(55, 90)
        ok, encoded = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            out = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    src_dir = Path(args.src)
    out_dir = Path(args.out)

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and "_aug" not in p.stem)
    if not image_paths:
        raise RuntimeError(f"no images found in {src_dir}")

    written = 0
    for image_path in image_paths:
        xml_path = image_path.with_suffix(".xml")
        if not xml_path.exists():
            raise FileNotFoundError(f"missing XML for {image_path}")

        tree, _, boxes = read_xml(xml_path)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read image: {image_path}")

        base_image, base_boxes, _ = scaled_image(image, boxes, args.max_side)
        base_name = image_path.name
        base_out = out_dir / base_name
        cv2.imwrite(str(base_out), base_image, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
        write_xml(tree, base_name, base_out, base_image.shape, base_boxes, base_out.with_suffix(".xml"))
        written += 1

        for idx in range(args.per_image):
            aug = augment(base_image, rng)
            aug_name = f"{image_path.stem}_aug{idx:02d}{image_path.suffix.lower()}"
            aug_out = out_dir / aug_name
            cv2.imwrite(str(aug_out), aug, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            tree, _, _ = read_xml(xml_path)
            write_xml(tree, aug_name, aug_out, aug.shape, base_boxes, aug_out.with_suffix(".xml"))
            written += 1

    print(f"Augmented {len(image_paths)} source images")
    print(f"  per image : {args.per_image}")
    print(f"  written   : {written} images + XML")
    print(f"  out       : {out_dir}")


if __name__ == "__main__":
    main()
