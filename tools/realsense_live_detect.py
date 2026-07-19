#!/usr/bin/env python3
"""
Real-time basket/mannequin detection on Intel RealSense D435i color stream.

Usage:
  python3 tools/realsense_live_detect.py
  python3 tools/realsense_live_detect.py --model <path/to/best.pt-or-best.onnx> --conf 0.4

Keys:
  q / ESC  quit
  d        toggle depth overlay
  t        toggle tiled inference (3x2 grid)
  h        toggle motion compensation
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

MODEL_DEFAULT = Path("runs/detect/runs/skyedge/yolo11s_realsense_v1/weights/best.pt")
CLASS_NAMES = ["basket", "mannequin"]
CLASS_COLORS = [(0, 200, 255), (0, 255, 80)]  # basket=cyan, mannequin=green


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(MODEL_DEFAULT))
    p.add_argument("--compare-model", default="",
                   help="optional second model; press m to switch models live")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--basket-conf", type=float, default=0.15,
                   help="별도 confidence 임계값 for basket (class 0)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0", help="Ultralytics inference device, e.g. 0 or cpu")
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--depth-width", type=int, default=848)
    p.add_argument("--depth-height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-distance", action="store_true")
    p.add_argument("--color-only", action="store_true")
    p.add_argument("--stable-frames", type=int, default=2)
    p.add_argument("--track-iou", type=float, default=0.25)
    p.add_argument("--min-area", type=float, default=0.00008)
    p.add_argument("--max-area", type=float, default=0.92)
    p.add_argument("--min-distance", type=float, default=0.15)
    p.add_argument("--max-distance", type=float, default=12.0)
    p.add_argument("--tiled", action="store_true")
    p.add_argument("--tile-overlap", type=float, default=0.15)
    p.add_argument("--tile-cols", type=int, default=3)
    p.add_argument("--tile-rows", type=int, default=2)
    p.add_argument("--motion-compensation", action="store_true")
    p.add_argument("--floor-start", type=float, default=0.35)
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
    streak: int = 1
    missed: int = 0
    confidence_decay: float = 1.0


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


def collect_detections(
    results, depth_frame, frame_shape, args, offset=(0, 0), enforce_threshold=True,
):
    h, w = frame_shape[:2]
    frame_area = float(w * h)
    detections = []
    boxes = results[0].boxes
    if not len(boxes):
        return detections
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy().astype(float)
        xyxy[[0, 2]] += offset[0]
        xyxy[[1, 3]] += offset[1]
        x1, y1, x2, y2 = xyxy
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / frame_area
        if area_ratio < args.min_area or area_ratio > args.max_area:
            continue
        dist = median_box_distance(depth_frame, int(x1), int(y1), int(x2), int(y2))
        if dist > 0 and not (args.min_distance <= dist <= args.max_distance):
            continue
        if enforce_threshold and conf < candidate_threshold(cls, dist, args):
            continue
        detections.append(Detection(xyxy=xyxy, cls=cls, conf=conf, dist=dist))
    return detections


def class_aware_nms(detections, iou_threshold=0.5):
    kept = []
    for detection in sorted(detections, key=lambda item: item.conf, reverse=True):
        if any(
            detection.cls == previous.cls and
            box_iou(detection.xyxy, previous.xyxy) >= iou_threshold
            for previous in kept
        ):
            continue
        kept.append(detection)
    return kept


def filter_candidate_thresholds(detections, args):
    return [
        detection for detection in detections
        if detection.conf >= candidate_threshold(detection.cls, detection.dist, args)
    ]


def has_multiscale_support(tile_detection, global_detections):
    tx1, ty1, tx2, ty2 = tile_detection.xyxy
    tcx = (tx1 + tx2) * 0.5
    tcy = (ty1 + ty2) * 0.5
    tdiag = max(tx2 - tx1, ty2 - ty1)
    for global_detection in global_detections:
        if global_detection.cls != tile_detection.cls:
            continue
        gx1, gy1, gx2, gy2 = global_detection.xyxy
        gcx = (gx1 + gx2) * 0.5
        gcy = (gy1 + gy2) * 0.5
        center_dist = float(np.hypot(tcx - gcx, tcy - gcy))
        if center_dist <= tdiag * 0.6:
            return True
    return False


def tile_has_depth_content(tile, depth_frame, args, min_valid_ratio=0.05):
    if depth_frame is None:
        return True
    x1, y1, x2, y2 = tile
    depth = np.asanyarray(depth_frame.get_data())
    dw, dh = depth_frame.get_width(), depth_frame.get_height()
    if (y2 - y1) != dh or (x2 - x1) != dw:
        scale_x = dw / (x2 - x1)
        scale_y = dh / (y2 - y1)
        dx1 = int(x1 * scale_x)
        dy1 = int(y1 * scale_y)
        dx2 = int(x2 * scale_x)
        dy2 = int(y2 * scale_y)
        tile_depth = depth[dy1:dy2, dx1:dx2]
    else:
        tile_depth = depth[y1:y2, x1:x2]
    if tile_depth.size == 0:
        return False
    valid = (tile_depth > 0) & (tile_depth < args.max_distance * 1000)
    return np.count_nonzero(valid) / tile_depth.size >= min_valid_ratio


def make_tiles(frame_shape, overlap_ratio, cols=3, rows=2):
    height, width = frame_shape[:2]
    tile_w = width // cols
    tile_h = height // rows
    overlap_w = int(round(tile_w * overlap_ratio * 0.5))
    overlap_h = int(round(tile_h * overlap_ratio * 0.5))
    tiles = []
    for row in range(rows):
        for col in range(cols):
            x1 = max(0, col * tile_w - overlap_w)
            y1 = max(0, row * tile_h - overlap_h)
            x2 = min(width, (col + 1) * tile_w + overlap_w)
            y2 = min(height, (row + 1) * tile_h + overlap_h)
            tiles.append((x1, y1, x2, y2))
    return tiles


def estimate_floor_homography(previous_gray, current_gray, floor_start):
    if previous_gray is None:
        return None, 0, 0
    height, width = previous_gray.shape
    mask = np.zeros_like(previous_gray)
    mask[int(height * floor_start):, :] = 255
    previous_points = cv2.goodFeaturesToTrack(
        previous_gray, maxCorners=350, qualityLevel=0.01,
        minDistance=8, blockSize=7, mask=mask,
    )
    if previous_points is None or len(previous_points) < 20:
        return None, 0, 0
    current_points, status, _ = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous_points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if current_points is None or status is None:
        return None, 0, len(previous_points)
    valid = status.reshape(-1) == 1
    source = previous_points.reshape(-1, 2)[valid]
    destination = current_points.reshape(-1, 2)[valid]
    if len(source) < 16:
        return None, 0, len(source)
    homography, inlier_mask = cv2.findHomography(
        source, destination, cv2.RANSAC, ransacReprojThreshold=3.0,
    )
    if homography is None or inlier_mask is None:
        return None, 0, len(source)
    inliers = int(inlier_mask.sum())
    if inliers < 12 or inliers / len(source) < 0.45:
        return None, inliers, len(source)
    corners = np.array(
        [[[0.0, 0.0]], [[width - 1.0, 0.0]], [[width - 1.0, height - 1.0]], [[0.0, height - 1.0]]],
        dtype=np.float32,
    )
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    if not np.isfinite(warped).all() or np.max(np.abs(warped)) > max(width, height) * 4:
        return None, inliers, len(source)
    return homography, inliers, len(source)


def warp_tracks(tracks, homography, frame_shape):
    if homography is None:
        return tracks
    height, width = frame_shape[:2]
    warped_tracks = []
    for track in tracks:
        x1, y1, x2, y2 = track.xyxy
        corners = np.array(
            [[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32
        )
        warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
        nx1 = float(np.clip(warped[:, 0].min(), 0, width - 1))
        ny1 = float(np.clip(warped[:, 1].min(), 0, height - 1))
        nx2 = float(np.clip(warped[:, 0].max(), 0, width - 1))
        ny2 = float(np.clip(warped[:, 1].max(), 0, height - 1))
        if nx2 - nx1 >= 2 and ny2 - ny1 >= 2:
            track.xyxy = np.array([nx1, ny1, nx2, ny2], dtype=float)
            warped_tracks.append(track)
    return warped_tracks


def update_tracks(tracks, detections, args):
    for track in tracks:
        track.missed += 1
        track.confidence_decay = max(0.3, track.confidence_decay - 0.2)
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
            alpha = 0.55
            track.xyxy = alpha * track.xyxy + (1.0 - alpha) * det.xyxy
            track.conf = max(det.conf, 0.7 * track.conf + 0.3 * det.conf)
            if det.dist > 0:
                track.dist = det.dist if track.dist <= 0 else 0.7 * track.dist + 0.3 * det.dist
            track.hits = min(track.hits + 1, 30)
            track.streak += 1
            track.missed = 0
            track.confidence_decay = 1.0
            used.add(best_idx)
        else:
            tracks.append(Track(det.xyxy.copy(), det.cls, det.conf, det.dist))
    return [track for track in tracks if track.missed <= 4]


def visible_tracks(tracks, stable_frames):
    result = []
    for track in tracks:
        if track.streak >= stable_frames and track.missed <= 2:
            result.append(track)
    return result


def draw_detections(frame, detections, show_distance):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.xyxy)
        cls = det.cls
        conf = det.conf
        color = CLASS_COLORS[cls] if cls < len(CLASS_COLORS) else (255, 255, 255)
        stale = getattr(det, "missed", 0) > 0
        decay = getattr(det, "confidence_decay", 1.0)
        if stale:
            color = tuple(int(c * decay) for c in color)
        thickness = 1 if stale else 2
        label = f"{CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else cls} {conf:.2f}"
        if show_distance and det.dist > 0:
            label += f"  {det.dist:.2f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )
    return frame


def draw_hud(
    frame, fps_buf, num_det, show_depth, model_label, compare_enabled,
    tiled_enabled, motion_enabled, motion_inliers,
):
    fps = sum(fps_buf) / len(fps_buf) if fps_buf else 0.0
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Det: {num_det}", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Model: {model_label}", (10, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 120), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[d] depth {'ON' if show_depth else 'OFF'}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"[t] tiled {'ON' if tiled_enabled else 'OFF'} ({args.tile_cols}x{args.tile_rows})", (10, 122),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(
        frame, f"[h] motion {'ON' if motion_enabled else 'OFF'} ({motion_inliers} inliers)",
        (10, 144), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
    )
    controls = "[m] switch model  " if compare_enabled else ""
    cv2.putText(frame, controls + "[q/ESC] quit", (10, 166),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)


def main():
    global args
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"model not found: {model_path.resolve()}")

    print(f"Loading model: {model_path}")
    model_paths = [model_path]
    if args.compare_model:
        compare_path = Path(args.compare_model)
        if not compare_path.exists():
            raise FileNotFoundError(f"compare model not found: {compare_path.resolve()}")
        model_paths.append(compare_path)

    models = []
    for path in model_paths:
        loaded = YOLO(str(path), task="detect")
        model_names = [loaded.names[index] for index in sorted(loaded.names)]
        if model_names != CLASS_NAMES:
            raise ValueError(
                f"expected custom classes {CLASS_NAMES}, got {model_names} for {path}. "
                "Train the model on the basket/mannequin dataset before GUI debugging."
            )
        loaded(
            np.zeros((args.height, args.width, 3), dtype=np.uint8),
            device=args.device,
            verbose=False,
        )
        models.append((path.stem, loaded))
    active_model = 0
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
    if not args.color_only:
        config.enable_stream(
            rs.stream.depth, args.depth_width, args.depth_height, rs.format.z16, args.fps
        )
    profile = pipeline.start(config)

    align = None if args.color_only else rs.align(rs.stream.color)
    depth_scale = (
        0.001 if args.color_only else profile.get_device().first_depth_sensor().get_depth_scale()
    )

    print("Streaming — press 'q' or ESC to quit, 'd' for depth, 't' for tiled, 'h' for motion.")

    fps_buf = deque(maxlen=30)
    show_depth = False
    tracks = []
    t_prev = time.perf_counter()
    consecutive_timeouts = 0
    tiled_enabled = args.tiled
    motion_enabled = args.motion_compensation
    previous_gray = None
    motion_inliers = 0

    try:
        while True:
            try:
                frames = pipeline.wait_for_frames(timeout_ms=5000)
                consecutive_timeouts = 0
            except RuntimeError as exc:
                if "Frame didn't arrive" not in str(exc):
                    raise
                consecutive_timeouts += 1
                print(f"Frame timeout ({consecutive_timeouts}/10); retrying...")
                if consecutive_timeouts >= 10:
                    raise RuntimeError(
                        "RealSense produced no frames after 10 retries; check USB bandwidth/cable"
                    ) from exc
                continue
            aligned = align.process(frames) if align is not None else frames
            color_frame = aligned.get_color_frame()
            depth_frame = None if args.color_only else aligned.get_depth_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            current_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            if motion_enabled:
                homography, motion_inliers, _ = estimate_floor_homography(
                    previous_gray, current_gray, args.floor_start
                )
                tracks = warp_tracks(tracks, homography, color_image.shape)
            else:
                motion_inliers = 0
            previous_gray = current_gray

            model_label, model = models[active_model]
            tiles = []
            if tiled_enabled:
                tiles = make_tiles(color_image.shape, args.tile_overlap, cols=args.tile_cols, rows=args.tile_rows)
                global_results = model(
                    color_image, imgsz=args.imgsz, conf=0.05, iou=0.55,
                    max_det=50, device=args.device, verbose=False,
                )
                global_candidates = collect_detections(
                    global_results, depth_frame, color_image.shape, args,
                    enforce_threshold=False,
                )
                detections = filter_candidate_thresholds(global_candidates, args)
                for x1, y1, x2, y2 in tiles:
                    if not tile_has_depth_content((x1, y1, x2, y2), depth_frame, args):
                        continue
                    tile_results = model(
                        color_image[y1:y2, x1:x2], imgsz=args.imgsz,
                        conf=min(args.conf, args.basket_conf, 0.12), iou=0.55,
                        max_det=20, device=args.device, verbose=False,
                    )
                    tile_detections = collect_detections(
                        tile_results, depth_frame, color_image.shape, args, offset=(x1, y1)
                    )
                    detections.extend(
                        d for d in tile_detections
                        if has_multiscale_support(d, global_candidates)
                    )
            else:
                results = model(
                    color_image, imgsz=args.imgsz,
                    conf=min(args.conf, args.basket_conf, 0.12), iou=0.55,
                    max_det=20, device=args.device, verbose=False,
                )
                detections = collect_detections(
                    results, depth_frame, color_image.shape, args
                )

            detections = class_aware_nms(detections)
            best_per_class = {}
            for d in detections:
                if d.cls not in best_per_class or d.conf > best_per_class[d.cls].conf:
                    best_per_class[d.cls] = d
            detections = list(best_per_class.values())

            tracks = update_tracks(tracks, detections, args)
            required_streak = max(args.stable_frames, 2) if tiled_enabled else args.stable_frames
            shown = visible_tracks(tracks, required_streak)
            best_per_class_show = {}
            for t in shown:
                if t.cls not in best_per_class_show or t.conf > best_per_class_show[t.cls].conf:
                    best_per_class_show[t.cls] = t
            shown = list(best_per_class_show.values())

            if tiled_enabled:
                for x1, y1, x2, y2 in tiles:
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (255, 160, 40), 1)

            draw_detections(color_image, shown, not args.no_distance)

            t_now = time.perf_counter()
            fps_buf.append(1.0 / max(t_now - t_prev, 1e-6))
            t_prev = t_now

            draw_hud(
                color_image, fps_buf, len(shown), show_depth,
                model_label, len(models) > 1,
                tiled_enabled, motion_enabled, motion_inliers,
            )

            if show_depth and depth_frame:
                dmap = depth_colormap(depth_frame, args.width // 3, args.height // 3)
                color_image[args.height - dmap.shape[0]:, args.width - dmap.shape[1]:] = dmap

            cv2.imshow("basket/mannequin detector  [RealSense D435i]", color_image)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("d"):
                show_depth = not show_depth if not args.color_only else False
            if key == ord("m") and len(models) > 1:
                active_model = (active_model + 1) % len(models)
                tracks.clear()
                fps_buf.clear()
                print(f"Active model: {models[active_model][0]}")
            if key == ord("t"):
                tiled_enabled = not tiled_enabled
                tracks.clear()
                fps_buf.clear()
                print(f"Tiled inference: {'ON' if tiled_enabled else 'OFF'}")
            if key == ord("h"):
                motion_enabled = not motion_enabled
                tracks.clear()
                previous_gray = None
                motion_inliers = 0
                print(f"Motion compensation: {'ON' if motion_enabled else 'OFF'}")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
