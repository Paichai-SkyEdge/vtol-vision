#!/usr/bin/env python3
"""
Real-time basket/mannequin detection on Intel RealSense D435i color stream.

Usage:
  python3 tools/realsense_live_detect.py
  python3 tools/realsense_live_detect.py --model <path/to/best.onnx> --conf 0.4

Keys:
  q / ESC  quit
  d        toggle depth overlay
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

ONNX_DEFAULT = Path(
    "runs/detect/runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt"
)
CLASS_NAMES = ["basket", "mannequin"]
CLASS_COLORS = [(0, 200, 255), (0, 255, 80)]  # basket=cyan, mannequin=green


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(ONNX_DEFAULT))
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--basket-conf", type=float, default=0.15,
                   help="별도 confidence 임계값 for basket (class 0)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    return p.parse_args()


def depth_colormap(depth_frame, width, height):
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
    )
    return cv2.resize(depth_colormap, (width, height))


def draw_detections(frame, results, depth_frame, depth_scale, show_depth):
    h, w = frame.shape[:2]
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (255, 255, 255)
        label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls} {conf:.2f}"

        # depth at box center
        if show_depth and depth_frame is not None:
            cx = min(max((x1 + x2) // 2, 0), depth_frame.get_width() - 1)
            cy = min(max((y1 + y2) // 2, 0), depth_frame.get_height() - 1)
            dist = depth_frame.get_distance(cx, cy)
            if dist > 0:
                label += f"  {dist:.2f}m"

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
                conf=args.basket_conf,  # 낮은 쪽 기준으로 뽑고 아래서 필터
                verbose=False,
            )

            # 클래스별 confidence 필터: basket은 basket_conf, 나머지는 conf
            boxes = results[0].boxes
            if len(boxes):
                keep = [
                    i for i, (cls, c) in enumerate(
                        zip(boxes.cls.tolist(), boxes.conf.tolist())
                    )
                    if (int(cls) == 0 and c >= args.basket_conf)
                    or (int(cls) != 0 and c >= args.conf)
                ]
                results[0].boxes = boxes[keep]

            draw_detections(color_image, results, depth_frame if show_depth else None,
                            depth_scale, show_depth)

            t_now = time.perf_counter()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            draw_hud(color_image, fps_buf, len(results[0].boxes), show_depth)

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
