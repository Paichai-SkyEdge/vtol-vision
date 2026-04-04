#!/usr/bin/env python3

import argparse
import statistics
import time
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference with OpenCV DNN")
    parser.add_argument(
        "--model",
        default="runs/detect/mannequin_detect/yolo11n_v1_gpu/weights/best.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image-dir",
        default="datasets/merged/val/images",
        help="Directory of validation images",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size")
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cpu",
        help="OpenCV DNN backend to benchmark",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional image count limit")
    return parser.parse_args()


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2

    resized = cv2.resize(image, (new_w, new_h))
    canvas = cv2.copyMakeBorder(
        resized,
        pad_y,
        size - new_h - pad_y,
        pad_x,
        size - new_w - pad_x,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return canvas


def configure_net(net, backend):
    if backend == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_dir = Path(args.image_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path.resolve()}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir.resolve()}")

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError("no images found to benchmark")

    net = cv2.dnn.readNet(str(model_path))
    configure_net(net, args.backend)

    timings_ms = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        letterboxed = letterbox(image, args.imgsz)
        blob = cv2.dnn.blobFromImage(
            letterboxed,
            scalefactor=1.0 / 255.0,
            size=(args.imgsz, args.imgsz),
            swapRB=True,
            crop=False,
        )

        start = time.perf_counter()
        net.setInput(blob)
        _ = net.forward(net.getUnconnectedOutLayersNames())
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    if not timings_ms:
        raise RuntimeError("no valid images were benchmarked")

    mean_ms = statistics.mean(timings_ms)
    median_ms = statistics.median(timings_ms)
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0

    print(f"backend={args.backend}")
    print(f"images={len(timings_ms)}")
    print(f"mean_ms={mean_ms:.3f}")
    print(f"median_ms={median_ms:.3f}")
    print(f"min_ms={min(timings_ms):.3f}")
    print(f"max_ms={max(timings_ms):.3f}")
    print(f"fps={fps:.2f}")


if __name__ == "__main__":
    main()
