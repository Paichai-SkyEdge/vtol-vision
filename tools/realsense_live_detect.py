#!/usr/bin/env python3
"""
Real-time basket/mannequin detection on Intel RealSense D435i color stream.

Usage:
  python3 tools/realsense_live_detect.py
  python3 tools/realsense_live_detect.py --model <path/to/best.pt-or-best.onnx> --conf 0.4

Keys:
  q / ESC  quit
  d        toggle depth overlay
"""

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

MODEL_DEFAULT = Path("weights/basket_mannequin_yolo11n/best.pt")
CLASS_NAMES = ["basket", "mannequin"]
CLASS_COLORS = [(0, 200, 255), (0, 255, 80)]  # basket=cyan, mannequin=green


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODEL_DEFAULT))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--basket-conf", type=float, default=0.15,
                   help="별도 confidence 임계값 for basket (class 0)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-distance", action="store_true",
                   help="disable distance text on detection boxes")
    p.add_argument("--stable-frames", type=int, default=2,
                   help="display only detections matched for this many frames")
    p.add_argument("--track-iou", type=float, default=0.25)
    p.add_argument("--min-area", type=float, default=0.00008,
                   help="minimum bbox area ratio of frame")
    p.add_argument("--max-area", type=float, default=0.92,
                   help="maximum bbox area ratio of frame")
    p.add_argument("--min-distance", type=float, default=0.15)
    p.add_argument("--max-distance", type=float, default=12.0)
    return p.parse_args()


@dataclass
class Detection:
    xyxy: np.ndarray
    cls: int
    conf: float
    dist: float = 0.0


@dataclass
class Track:
    xyxy: np.ndarray
    cls: int
    conf: float
    dist: float
    hits: int = 1
    missed: int = 0


def depth_colormap(depth_frame, width, height):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    return cv2.resize(depth_colormap, (width, height))


def median_box_distance(depth_frame, x1, y1, x2, y2):
    if depth_frame is None:
        return 0.0

    width = depth_frame.get_width()
    height = depth_frame.get_height()
    cx = min(max((x1 + x2) // 2, 0), width - 1)
    cy = min(max((y1 + y2) // 2, 0), height - 1)
    half_w = max(2, min(12, (x2 - x1) // 8))
    half_h = max(2, min(12, (y2 - y1) // 8))

    distances = []
    for yy in range(max(0, cy - half_h), min(height, cy + half_h + 1), 2):
        for xx in range(max(0, cx - half_w), min(width, cx + half_w + 1), 2):
            dist = depth_frame.get_distance(xx, yy)
            if 0.1 < dist < 20.0:
                distances.append(dist)

    if not distances:
        return depth_frame.get_distance(cx, cy)
    return float(np.median(distances))


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def candidate_threshold(cls, distance, args):
    base = args.basket_conf if cls == 0 else args.conf
    if distance and distance > 3.0:
        return max(0.08, base - 0.07)
    if distance and distance < 0.8:
        return base + 0.08
    return base


def collect_detections(results, depth_frame, frame_shape, args):
    h, w = frame_shape[:2]
    frame_area = float(w * h)
    detections = []
    boxes = results[0].boxes
    if not len(boxes):
        return detections

    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(float)
        x1, y1, x2, y2 = xyxy
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
        if area_ratio < args.min_area or area_ratio > args.max_area:
            continue

        dist = median_box_distance(depth_frame, int(x1), int(y1), int(x2), int(y2))
        if dist > 0 and not (args.min_distance <= dist <= args.max_distance):
            continue
        if conf < candidate_threshold(cls, dist, args):
            continue

        detections.append(Detection(xyxy=xyxy, cls=cls, conf=conf, dist=dist))
    return detections


def update_tracks(tracks, detections, args):
    for track in tracks:
        track.missed += 1

    used = set()
    for det in sorted(detections, key=lambda d: d.conf, reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, track in enumerate(tracks):
            if idx in used or track.cls != det.cls:
                continue
            iou = box_iou(track.xyxy, det.xyxy)
            if iou > best_iou:
                best_idx = idx
                best_iou = iou

        if best_idx is not None and best_iou >= args.track_iou:
            track = tracks[best_idx]
            alpha = 0.65
            track.xyxy = alpha * track.xyxy + (1.0 - alpha) * det.xyxy
            track.conf = max(det.conf, 0.7 * track.conf + 0.3 * det.conf)
            if det.dist > 0:
                track.dist = det.dist if track.dist <= 0 else 0.7 * track.dist + 0.3 * det.dist
            track.hits = min(track.hits + 1, 30)
            track.missed = 0
            used.add(best_idx)
        else:
            tracks.append(Track(det.xyxy.copy(), det.cls, det.conf, det.dist))

    return [track for track in tracks if track.missed <= 4]


def visible_tracks(tracks, stable_frames):
    return [track for track in tracks if track.hits >= stable_frames and track.missed <= 1]


def draw_detections(frame, detections, show_distance):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        cls = det.cls
        conf = det.conf
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (255, 255, 255)
        label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls} {conf:.2f}"

        if show_distance and det.dist > 0:
            label += f"  {det.dist:.2f}m"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


def draw_hud(frame, fps_buf, num_det, show_depth):
    fps = sum(fps_buf) / len(fps_buf) if fps_buf else 0.0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Det: {num_det}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    mode = "depth ON" if show_depth else "depth OFF"
    cv2.putText(frame, f"[d] {mode}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, "[q/ESC] quit", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path.resolve()}")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path), task="detect")
    model(np.zeros((args.height, args.width, 3), dtype=np.uint8), verbose=False)  # warm-up
    print("Model ready.")
    print(f"conf: mannequin>={args.conf:.2f}  basket>={args.basket_conf:.2f}")
    print(
        "filters: "
        f"stable_frames={args.stable_frames} area={args.min_area:.5f}~{args.max_area:.2f} "
        f"distance={args.min_distance:.2f}~{args.max_distance:.1f}m"
    )

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    align = rs.align(rs.stream.color)

    print("Streaming — press 'q' or ESC to quit, 'd' to toggle depth overlay.")

    fps_buf = deque(maxlen=30)
    show_depth = False
    tracks = []
    t_prev = time.perf_counter()

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            results = model(
                color_image,
                imgsz=args.imgsz,
                conf=min(args.conf, args.basket_conf, 0.12),  # 후보는 넓게 받고 아래서 안정화
                iou=0.55,
                max_det=20,
                verbose=False,
            )

            detections = collect_detections(results, depth_frame, color_image.shape, args)
            tracks = update_tracks(tracks, detections, args)
            shown = visible_tracks(tracks, args.stable_frames)

            draw_detections(color_image, shown, not args.no_distance)

            t_now = time.perf_counter()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            draw_hud(color_image, fps_buf, len(shown), show_depth)

            if show_depth and depth_frame:
                dmap = depth_colormap(depth_frame, args.width // 3, args.height // 3)
                color_image[args.height - dmap.shape[0]:, args.width - dmap.shape[1]:] = dmap

            cv2.imshow("basket/mannequin detector  [D435i]", color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
