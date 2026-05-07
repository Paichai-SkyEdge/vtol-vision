#!/usr/bin/env python3
"""
Generate aggressive geometric augmentation candidates for manual relabeling.

This script intentionally does not copy or transform labels. It creates images
with rotation, crop, perspective, scale/pad, synthetic border/background, and
photometric changes so they can be opened in labelImg and labeled from scratch.
"""

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASSES = ["basket", "mannequin"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate relabeling image candidates")
    parser.add_argument("--src", default="images/skyedge_vision", help="Source image dir")
    parser.add_argument("--out", default="images/skyedge_vision_relabel", help="Output image dir")
    parser.add_argument("--per-image", type=int, default=10, help="Generated copies per source image")
    parser.add_argument("--max-side", type=int, default=1600, help="Resize long side before augmenting")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_long_side(image, max_side):
    h, w = image.shape[:2]
    if max_side <= 0 or max(h, w) <= max_side:
        return image
    scale = max_side / float(max(h, w))
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def random_background(h, w, rng):
    mode = rng.choice(["solid", "gradient", "noise", "floor"])
    if mode == "solid":
        color = np.array([rng.randint(20, 235), rng.randint(20, 235), rng.randint(20, 235)], dtype=np.uint8)
        return np.full((h, w, 3), color, dtype=np.uint8)

    if mode == "noise":
        base = rng.randint(35, 220)
        bg = np.random.default_rng(rng.randint(0, 1_000_000)).normal(base, rng.uniform(8, 35), (h, w, 3))
        return np.clip(bg, 0, 255).astype(np.uint8)

    if mode == "floor":
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        top = np.array([rng.randint(60, 190), rng.randint(60, 190), rng.randint(60, 190)], dtype=np.float32)
        bottom = np.array([rng.randint(25, 150), rng.randint(25, 150), rng.randint(25, 150)], dtype=np.float32)
        for y in range(h):
            t = y / max(h - 1, 1)
            bg[y, :, :] = np.clip(top * (1 - t) + bottom * t, 0, 255)
        for _ in range(rng.randint(3, 8)):
            x = rng.randint(0, w - 1)
            cv2.line(bg, (x, 0), (x + rng.randint(-80, 80), h), (rng.randint(30, 200),) * 3, 1)
        return cv2.GaussianBlur(bg, (0, 0), sigmaX=rng.uniform(1, 4))

    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    c1 = np.array([rng.randint(20, 220), rng.randint(20, 220), rng.randint(20, 220)], dtype=np.float32)
    c2 = np.array([rng.randint(20, 220), rng.randint(20, 220), rng.randint(20, 220)], dtype=np.float32)
    t = (x[None, :] * rng.random() + y * rng.random())
    t /= max(float(t.max()), 1e-6)
    return np.clip(c1 * (1 - t[:, :, None]) + c2 * t[:, :, None], 0, 255).astype(np.uint8)


def photometric(image, rng):
    out = image.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-12, 12)) % 180
    hsv[:, :, 1] *= rng.uniform(0.45, 1.75)
    hsv[:, :, 2] *= rng.uniform(0.45, 1.65)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    out = cv2.convertScaleAbs(out, alpha=rng.uniform(0.65, 1.5), beta=rng.uniform(-55, 55))

    if rng.random() < 0.45:
        noise = np.random.default_rng(rng.randint(0, 1_000_000)).normal(0, rng.uniform(4, 20), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.35:
        out = cv2.GaussianBlur(out, (rng.choice([3, 5]), rng.choice([3, 5])), sigmaX=rng.uniform(0.2, 1.2))

    return out


def rotate_scale_on_background(image, rng):
    h, w = image.shape[:2]
    canvas_scale = rng.uniform(1.05, 1.45)
    ch, cw = round(h * canvas_scale), round(w * canvas_scale)
    canvas = random_background(ch, cw, rng)

    scale = rng.uniform(0.55, 1.25)
    angle = rng.uniform(-55, 55)
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += (cw - w) / 2 + rng.uniform(-0.12, 0.12) * cw
    matrix[1, 2] += (ch - h) / 2 + rng.uniform(-0.12, 0.12) * ch

    warped = cv2.warpAffine(image, matrix, (cw, ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    mask = cv2.warpAffine(np.full((h, w), 255, dtype=np.uint8), matrix, (cw, ch), flags=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
    return np.clip(canvas * (1 - mask[:, :, None]) + warped * mask[:, :, None], 0, 255).astype(np.uint8)


def perspective(image, rng):
    h, w = image.shape[:2]
    margin_x = w * rng.uniform(0.04, 0.18)
    margin_y = h * rng.uniform(0.04, 0.18)
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32(
        [
            [rng.uniform(0, margin_x), rng.uniform(0, margin_y)],
            [w - 1 - rng.uniform(0, margin_x), rng.uniform(0, margin_y)],
            [w - 1 - rng.uniform(0, margin_x), h - 1 - rng.uniform(0, margin_y)],
            [rng.uniform(0, margin_x), h - 1 - rng.uniform(0, margin_y)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    bg = random_background(h, w, rng)
    warped = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((h, w), 255, dtype=np.uint8), matrix, (w, h), flags=cv2.INTER_LINEAR)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.0).astype(np.float32) / 255.0
    return np.clip(bg * (1 - mask[:, :, None]) + warped * mask[:, :, None], 0, 255).astype(np.uint8)


def random_crop_or_pad(image, rng):
    h, w = image.shape[:2]
    if rng.random() < 0.65:
        crop_scale = rng.uniform(0.62, 0.95)
        crop_w = max(32, round(w * crop_scale))
        crop_h = max(32, round(h * crop_scale))
        x0 = rng.randint(0, max(w - crop_w, 0))
        y0 = rng.randint(0, max(h - crop_h, 0))
        image = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
        h, w = image.shape[:2]

    if rng.random() < 0.7:
        pad_x = round(w * rng.uniform(0.05, 0.35))
        pad_y = round(h * rng.uniform(0.05, 0.35))
        bg = random_background(h + pad_y * 2, w + pad_x * 2, rng)
        x = pad_x + rng.randint(-pad_x // 2, pad_x // 2)
        y = pad_y + rng.randint(-pad_y // 2, pad_y // 2)
        bg[y : y + h, x : x + w] = image
        image = bg

    return image


def augment(image, rng):
    out = image.copy()
    if rng.random() < 0.8:
        out = rotate_scale_on_background(out, rng)
    if rng.random() < 0.7:
        out = perspective(out, rng)
    out = random_crop_or_pad(out, rng)
    out = photometric(out, rng)

    # Keep output sizes manageable for labelImg and YOLO training.
    out = resize_long_side(out, 1600)
    return out


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    src_dir = Path(args.src)
    out_dir = Path(args.out)

    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS and "_geo" not in p.stem)
    if not image_paths:
        raise RuntimeError(f"no source images found: {src_dir}")

    (out_dir / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (out_dir / "predefined_classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")

    written = 0
    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read: {image_path}")
        image = resize_long_side(image, args.max_side)

        for idx in range(args.per_image):
            out = augment(image, rng)
            out_name = f"{image_path.stem}_geo{idx:02d}.jpg"
            cv2.imwrite(str(out_dir / out_name), out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            written += 1

    print(f"Generated {written} relabeling candidates")
    print(f"  sources   : {len(image_paths)}")
    print(f"  per image : {args.per_image}")
    print(f"  out       : {out_dir}")
    print("No labels were copied. Relabel these images manually.")


if __name__ == "__main__":
    main()
