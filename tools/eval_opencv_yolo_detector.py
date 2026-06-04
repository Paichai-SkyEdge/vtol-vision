#!/usr/bin/env python3

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


LETTERBOX_VALUE = 114


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int


@dataclass
class Detection:
    class_id: int
    score: float
    x: int
    y: int
    w: int
    h: int


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the current OpenCV DNN YOLO path against YOLO-format labels."
    )
    parser.add_argument(
        "--model",
        default="weights/mannequin_yolo11n/best.onnx",
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--image-dir",
        default="datasets/merged/val/images",
        help="Directory containing validation images",
    )
    parser.add_argument(
        "--label-dir",
        default="datasets/merged/val/labels",
        help="Directory containing YOLO txt labels",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--nms", type=float, default=0.45, help="NMS threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for a match")
    parser.add_argument(
        "--backend",
        choices=["cpu", "cuda"],
        default="cpu",
        help="OpenCV DNN backend to use",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional image limit")
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
        value=(LETTERBOX_VALUE, LETTERBOX_VALUE, LETTERBOX_VALUE),
    )
    return canvas, LetterboxMeta(scale=scale, pad_x=pad_x, pad_y=pad_y)


def configure_net(net, backend):
    if backend == "cuda":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


def clip_rect(x, y, w, h, frame_w, frame_h):
    x = max(0, x)
    y = max(0, y)
    w = max(0, min(w, frame_w - x))
    h = max(0, min(h, frame_h - y))
    return x, y, w, h


def normalize_output(output):
    if output.ndim == 3:
        raw = output[0]
        if raw.shape[0] < raw.shape[1]:
            return raw.T
        return raw
    if output.ndim == 2:
        return output
    raise RuntimeError(f"unsupported output rank: {output.ndim}")


def parse_detections(rows, meta, original_shape, conf_threshold, nms_threshold):
    frame_h, frame_w = original_shape[:2]
    boxes = []
    scores = []
    class_ids = []

    if rows.shape[1] == 5 or rows.shape[1] > 6:
        for row in rows:
            if rows.shape[1] == 5:
                class_id = 0
                confidence = float(row[4])
            else:
                objectness = float(row[4])
                if objectness <= 0.0:
                    continue
                class_slice = row[5:]
                if class_slice.size == 0:
                    continue
                class_id = int(class_slice.argmax())
                class_score = float(class_slice[class_id])
                confidence = objectness * class_score
            if confidence < conf_threshold:
                continue

            cx, cy, w, h = map(float, row[:4])
            x0 = (cx - (w * 0.5) - meta.pad_x) / meta.scale
            y0 = (cy - (h * 0.5) - meta.pad_y) / meta.scale
            ww = w / meta.scale
            hh = h / meta.scale
            x, y, ww_i, hh_i = clip_rect(
                int(round(x0)),
                int(round(y0)),
                int(round(ww)),
                int(round(hh)),
                frame_w,
                frame_h,
            )
            if ww_i <= 0 or hh_i <= 0:
                continue
            boxes.append([x, y, ww_i, hh_i])
            scores.append(confidence)
            class_ids.append(class_id)
    else:
        for row in rows:
            confidence = float(row[4])
            if confidence < conf_threshold:
                continue

            class_id = int(round(float(row[5]))) if rows.shape[1] >= 6 else 0
            x1, y1, x2, y2 = map(float, row[:4])
            if x2 <= 1.5 and y2 <= 1.5:
                x1 *= 640.0
                y1 *= 640.0
                x2 *= 640.0
                y2 *= 640.0
            x1 = (x1 - meta.pad_x) / meta.scale
            y1 = (y1 - meta.pad_y) / meta.scale
            x2 = (x2 - meta.pad_x) / meta.scale
            y2 = (y2 - meta.pad_y) / meta.scale
            x, y, ww_i, hh_i = clip_rect(
                int(round(x1)),
                int(round(y1)),
                int(round(max(0.0, x2 - x1))),
                int(round(max(0.0, y2 - y1))),
                frame_w,
                frame_h,
            )
            if ww_i <= 0 or hh_i <= 0:
                continue
            boxes.append([x, y, ww_i, hh_i])
            scores.append(confidence)
            class_ids.append(class_id)

    kept = cv2.dnn.NMSBoxes(boxes, scores, conf_threshold, nms_threshold)
    detections = []
    if len(kept) == 0:
        return detections
    for index in kept.flatten():
        x, y, w, h = boxes[index]
        detections.append(
            Detection(
                class_id=class_ids[index],
                score=float(scores[index]),
                x=int(x),
                y=int(y),
                w=int(w),
                h=int(h),
            )
        )
    return detections


def load_yolo_labels(label_path, image_shape):
    frame_h, frame_w = image_shape[:2]
    labels = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(float(parts[0]))
        cx = float(parts[1]) * frame_w
        cy = float(parts[2]) * frame_h
        w = float(parts[3]) * frame_w
        h = float(parts[4]) * frame_h
        x = int(round(cx - (w * 0.5)))
        y = int(round(cy - (h * 0.5)))
        labels.append(
            Detection(
                class_id=class_id,
                score=1.0,
                x=x,
                y=y,
                w=int(round(w)),
                h=int(round(h)),
            )
        )
    return labels


def iou(a, b):
    ax2 = a.x + a.w
    ay2 = a.y + a.h
    bx2 = b.x + b.w
    by2 = b.y + b.h

    inter_x1 = max(a.x, b.x)
    inter_y1 = max(a.y, b.y)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = (a.w * a.h) + (b.w * b.h) - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def match_detections(preds, targets, iou_threshold):
    preds = sorted(preds, key=lambda d: d.score, reverse=True)
    matched_targets = set()
    tp = 0
    fp = 0

    for pred in preds:
        best_idx = -1
        best_iou = 0.0
        for idx, target in enumerate(targets):
            if idx in matched_targets or pred.class_id != target.class_id:
                continue
            overlap = iou(pred, target)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_targets.add(best_idx)
            tp += 1
        else:
            fp += 1

    fn = len(targets) - len(matched_targets)
    return tp, fp, fn


def main():
    args = parse_args()
    model_path = Path(args.model)
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path.resolve()}")
    if not image_dir.exists():
        raise FileNotFoundError(f"image dir not found: {image_dir.resolve()}")
    if not label_dir.exists():
        raise FileNotFoundError(f"label dir not found: {label_dir.resolve()}")

    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    )
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise RuntimeError("no images found")

    net = cv2.dnn.readNet(str(model_path))
    configure_net(net, args.backend)

    preprocess_ms = []
    inference_ms = []
    postprocess_ms = []
    per_image_ms = []
    all_tp = 0
    all_fp = 0
    all_fn = 0
    num_predictions = 0
    num_targets = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        targets = load_yolo_labels(label_path, image.shape)

        start = time.perf_counter()
        input_image, meta = letterbox(image, args.imgsz)
        blob = cv2.dnn.blobFromImage(
            input_image,
            scalefactor=1.0 / 255.0,
            size=(args.imgsz, args.imgsz),
            swapRB=True,
            crop=False,
        )
        t1 = time.perf_counter()

        net.setInput(blob)
        outputs = net.forward(net.getUnconnectedOutLayersNames())
        t2 = time.perf_counter()

        rows = normalize_output(outputs[0])
        preds = parse_detections(rows, meta, image.shape, args.conf, args.nms)
        t3 = time.perf_counter()

        tp, fp, fn = match_detections(preds, targets, args.iou)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        num_predictions += len(preds)
        num_targets += len(targets)

        preprocess_ms.append((t1 - start) * 1000.0)
        inference_ms.append((t2 - t1) * 1000.0)
        postprocess_ms.append((t3 - t2) * 1000.0)
        per_image_ms.append((t3 - start) * 1000.0)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    mean_ms = statistics.mean(per_image_ms)

    print(f"backend={args.backend}")
    print(f"images={len(per_image_ms)}")
    print(f"targets={num_targets}")
    print(f"predictions={num_predictions}")
    print(f"tp={all_tp}")
    print(f"fp={all_fp}")
    print(f"fn={all_fn}")
    print(f"precision@{args.iou:.2f}={precision:.6f}")
    print(f"recall@{args.iou:.2f}={recall:.6f}")
    print(f"f1@{args.iou:.2f}={f1:.6f}")
    print(f"preprocess_ms={statistics.mean(preprocess_ms):.3f}")
    print(f"inference_ms={statistics.mean(inference_ms):.3f}")
    print(f"postprocess_ms={statistics.mean(postprocess_ms):.3f}")
    print(f"total_ms={mean_ms:.3f}")
    print(f"fps={1000.0 / mean_ms:.2f}")


if __name__ == "__main__":
    main()
