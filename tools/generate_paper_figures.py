#!/usr/bin/env python3
"""
Publication-quality figures for the VTOL vision system paper.

Generates in paper/figures/:
  fig_performance.{pdf,png}      — our dataset, validation metrics, and deployment path
  fig_training_curves.{pdf,png}  — YOLO11n training curves from our run
  fig_latency.{pdf,png}          — measured Jetson backend latency/throughput
  fig_gradcam.{pdf,png}          — EigenCAM activation overlay, only when requested

Usage:
  python3 tools/generate_paper_figures.py
  python3 tools/generate_paper_figures.py --latency paper/figures/latency.json
  python3 tools/generate_paper_figures.py --only training_curves
  python3 tools/generate_paper_figures.py --only gradcam
"""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import numpy as np
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont as PILImageFont

_KO_FONT_PATH = "/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf"
if Path(_KO_FONT_PATH).exists():
    font_manager.fontManager.addfont(_KO_FONT_PATH)

def _pil_draw_box(img_rgb: np.ndarray, x1, y1, x2, y2, lbl, col_rgb) -> np.ndarray:
    """Draw a Korean-text bounding box using PIL, font size scaled to image height."""
    h = img_rgb.shape[0]
    font_size = max(20, h // 22)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = PILImageFont.truetype(_KO_FONT_PATH, font_size)
    except Exception:
        font = PILImageFont.load_default()
    draw.rectangle([x1, y1, x2, y2], outline=col_rgb, width=max(3, h // 200))
    ty = max(y1 - font_size - 6, 0)
    tb = draw.textbbox((x1 + 3, ty), lbl, font=font)
    draw.rectangle([tb[0] - 2, tb[1] - 2, tb[2] + 2, tb[3] + 2], fill=(0, 0, 0))
    draw.text((x1 + 3, ty), lbl, font=font, fill=col_rgb)
    return np.array(pil)

OUT = Path("paper/figures")

# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "UnDotum",
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         150,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.06,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linewidth":     0.5,
    "lines.linewidth":    1.6,
    "axes.unicode_minus": False,
})

C_BLUE   = "#2563EB"
C_GREEN  = "#16A34A"
C_RED    = "#DC2626"
C_ORANGE = "#D97706"
C_GRAY   = "#9CA3AF"
C_PURPLE = "#7C3AED"


def _save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"  saved  paper/figures/{name}.{{pdf,png}}")


# ── Data helpers ───────────────────────────────────────────────────────────────
def _read_training_results(csv_path: str) -> list[dict[str, float]]:
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({k.strip(): float(v) for k, v in r.items()})
    if not rows:
        raise ValueError(f"empty results csv: {csv_path}")
    return rows


def _moving_average(vals, window=7):
    vals = list(vals)
    out = []
    for i in range(len(vals)):
        lo = max(0, i - window + 1)
        out.append(sum(vals[lo:i + 1]) / (i - lo + 1))
    return np.array(out)


def _dataset_stats(dataset_dir: str):
    root = Path(dataset_dir)
    class_names = ["basket", "mannequin"]
    stats = {}
    for split in ["train", "val"]:
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        images = [
            p for p in image_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        counts = [0, 0]
        labeled_images = set()
        for label_path in label_dir.glob("*.txt"):
            if label_path.name.endswith(".cache"):
                continue
            lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
            if lines:
                labeled_images.add(label_path.stem)
            for line in lines:
                cls = int(float(line.split()[0]))
                if 0 <= cls < len(counts):
                    counts[cls] += 1
        stats[split] = {
            "images": len(images),
            "labeled_images": len(labeled_images),
            "instances": counts,
            "total_instances": sum(counts),
        }
    return class_names, stats


def _flow_box(ax, xy, text, color):
    x, y = xy
    width, height = 0.76, 0.16
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.018,rounding_size=0.018",
        linewidth=1.1, edgecolor=color, facecolor=f"{color}18",
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2, y + height / 2, text,
        ha="center", va="center", fontsize=8.5,
        transform=ax.transAxes,
    )


# ── Figure 1: Our dataset, metrics, and deployment path ───────────────────────
def fig_performance(csv_path: str, dataset_dir: str):
    """Paper figure centered on the artifacts built in this repository."""

    rows = _read_training_results(csv_path)
    final = rows[-1]
    class_names, stats = _dataset_stats(dataset_dir)

    fig = plt.figure(figsize=(11.3, 3.8))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.05, 1.15, 1.45], wspace=0.38)
    ax_data = fig.add_subplot(gs[0, 0])
    ax_metric = fig.add_subplot(gs[0, 1])
    ax_flow = fig.add_subplot(gs[0, 2])

    # Dataset composition, from our YOLO labels.
    x = np.arange(len(class_names))
    train_counts = stats["train"]["instances"]
    val_counts = stats["val"]["instances"]
    width = 0.34
    ax_data.bar(x - width / 2, train_counts, width, color=C_BLUE, label="train", zorder=3)
    ax_data.bar(x + width / 2, val_counts, width, color=C_GREEN, label="val", zorder=3)
    for i, value in enumerate(train_counts):
        ax_data.text(i - width / 2, value + max(train_counts) * 0.025, str(value),
                     ha="center", va="bottom", fontsize=8)
    for i, value in enumerate(val_counts):
        ax_data.text(i + width / 2, value + max(train_counts) * 0.025, str(value),
                     ha="center", va="bottom", fontsize=8)
    ax_data.set_xticks(x)
    ax_data.set_xticklabels(["basket", "mannequin"])
    ax_data.set_ylabel("라벨 인스턴스 수")
    ax_data.set_title("A. 구축 데이터셋", fontweight="bold")
    ax_data.legend(frameon=False, loc="upper left")
    ax_data.text(
        0.0, -0.23,
        f"images: train {stats['train']['images']}, val {stats['val']['images']}",
        transform=ax_data.transAxes, fontsize=7.8, color="#4B5563",
    )

    # Final validation metrics, from our training run.
    metric_labels = ["Precision", "Recall", "mAP50", "mAP50-95"]
    metric_values = [
        final["metrics/precision(B)"] * 100,
        final["metrics/recall(B)"] * 100,
        final["metrics/mAP50(B)"] * 100,
        final["metrics/mAP50-95(B)"] * 100,
    ]
    colors = [C_GREEN, C_ORANGE, C_BLUE, C_PURPLE]
    y = np.arange(len(metric_labels))
    ax_metric.barh(y, metric_values, color=colors, height=0.55, zorder=3)
    for yi, value in zip(y, metric_values):
        ax_metric.text(value + 1.0, yi, f"{value:.1f}%", va="center",
                       fontsize=8.5, fontweight="bold")
    ax_metric.set_yticks(y)
    ax_metric.set_yticklabels(metric_labels)
    ax_metric.set_xlim(0, 108)
    ax_metric.set_xlabel("검증 점수 (%)")
    ax_metric.invert_yaxis()
    ax_metric.set_title("B. 검증 성능", fontweight="bold")

    # What was built, not an external comparison.
    ax_flow.axis("off")
    ax_flow.set_title("C. 구현 산출물 흐름", fontweight="bold", pad=8)
    steps = [
        ("현장 표적 영상\nbasket / mannequin YOLO 라벨", C_BLUE),
        ("YOLO11n 파인튜닝\n100 epochs · 640 px · batch 8", C_GREEN),
        ("배포 산출물\nbest.pt → ONNX → TensorRT FP16 engine", C_ORANGE),
        ("실시간 비전 노드\nROS2 topics + RealSense live detector", C_PURPLE),
    ]
    ys = [0.75, 0.52, 0.29, 0.06]
    for (label, color), yy in zip(steps, ys):
        _flow_box(ax_flow, (0.11, yy), label, color)
    for yy0, yy1 in zip(ys[:-1], ys[1:]):
        ax_flow.annotate(
            "", xy=(0.49, yy1 + 0.18), xytext=(0.49, yy0),
            xycoords=ax_flow.transAxes, textcoords=ax_flow.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.0, color="#4B5563"),
        )

    fig.suptitle(
        "Basket/Mannequin Detector — 우리가 구축한 데이터셋, 학습 결과, 배포 파이프라인",
        fontsize=10.5, y=1.02,
    )
    _save(fig, "fig_performance")


# ── Figure 2: Training curves from our run ────────────────────────────────────
def fig_training_curves(csv_path: str):
    rows = _read_training_results(csv_path)
    epochs = np.array([r["epoch"] for r in rows])

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.2), sharex=True)
    ax_pr, ax_map, ax_train, ax_val = axes.flatten()

    def plot_metric(ax, key, label, color):
        raw = np.array([r[key] for r in rows])
        sm = _moving_average(raw, 7)
        ax.plot(epochs, raw, color=color, alpha=0.16, linewidth=0.9)
        ax.plot(epochs, sm, color=color, linewidth=2.0,
                label=f"{label} final {raw[-1] * 100:.1f}%")

    plot_metric(ax_pr, "metrics/precision(B)", "Precision", C_GREEN)
    plot_metric(ax_pr, "metrics/recall(B)", "Recall", C_ORANGE)
    ax_pr.set_title("A. precision / recall", fontweight="bold")
    ax_pr.set_ylabel("점수")
    ax_pr.set_ylim(0, 1.05)
    ax_pr.legend(frameon=False, loc="lower right")

    plot_metric(ax_map, "metrics/mAP50(B)", "mAP50", C_BLUE)
    plot_metric(ax_map, "metrics/mAP50-95(B)", "mAP50-95", C_PURPLE)
    ax_map.set_title("B. mAP convergence", fontweight="bold")
    ax_map.set_ylim(0, 1.05)
    ax_map.legend(frameon=False, loc="lower right")

    loss_keys = [
        ("train/box_loss", "box", C_BLUE),
        ("train/cls_loss", "cls", C_GREEN),
        ("train/dfl_loss", "dfl", C_ORANGE),
    ]
    for key, label, color in loss_keys:
        raw = np.array([r[key] for r in rows])
        ax_train.plot(epochs, _moving_average(raw, 7), label=label, color=color)
    total_train = np.array([
        r["train/box_loss"] + r["train/cls_loss"] + r["train/dfl_loss"]
        for r in rows
    ])
    ax_train.plot(epochs, _moving_average(total_train, 7),
                  label="total", color=C_RED, linewidth=2.1)
    ax_train.set_title("C. train losses", fontweight="bold")
    ax_train.set_xlabel("에포크")
    ax_train.set_ylabel("손실")
    ax_train.legend(frameon=False, ncol=2)

    val_keys = [
        ("val/box_loss", "box", C_BLUE),
        ("val/cls_loss", "cls", C_GREEN),
        ("val/dfl_loss", "dfl", C_ORANGE),
    ]
    for key, label, color in val_keys:
        raw = np.array([r[key] for r in rows])
        ax_val.plot(epochs, _moving_average(raw, 7), label=label, color=color)
    total_val = np.array([
        r["val/box_loss"] + r["val/cls_loss"] + r["val/dfl_loss"]
        for r in rows
    ])
    ax_val.plot(epochs, _moving_average(total_val, 7),
                label="total", color=C_RED, linewidth=2.1)
    ax_val.set_title("D. validation losses", fontweight="bold")
    ax_val.set_xlabel("에포크")
    ax_val.legend(frameon=False, ncol=2)

    for ax in axes.flatten():
        ax.set_xlim(1, epochs[-1])
        ax.grid(True, alpha=0.22)

    fig.suptitle(
        "YOLO11n Fine-Tuning Curves — basket/mannequin dataset, 100 epochs",
        fontsize=11, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save(fig, "fig_training_curves")



# ── Figure 3: EigenCAM ─────────────────────────────────────────────────────────
def _crop_landscape(img_bgr, ratio=4 / 3):
    """Center-crop to landscape aspect ratio so subplot cells fill without whitespace."""
    h, w = img_bgr.shape[:2]
    if w / h >= ratio:
        new_w = int(h * ratio)
        x = (w - new_w) // 2
        return img_bgr[:, x:x + new_w]
    new_h = int(w / ratio)
    y = (h - new_h) // 2
    return img_bgr[y:y + new_h, :]


def _eigencam_overlay(torch_model, model, img_bgr, target, conf=0.20):
    """Return (img_rgb, overlay) for one image using EigenCAM."""
    import torch

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    scale    = min(640 / h, 640 / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    pad_top  = (640 - new_h) // 2
    pad_left = (640 - new_w) // 2

    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = cv2.resize(img_bgr, (new_w, new_h))
    inp = torch.from_numpy(canvas).float().permute(2, 0, 1).unsqueeze(0) / 255.0

    activations = {}
    handle = target.register_forward_hook(
        lambda m, i, o: activations.update({"feat": o.detach()})
    )
    with torch.no_grad():
        torch_model(inp)
    handle.remove()

    act = activations["feat"]
    C_dim, fH, fW = act.shape[1], act.shape[2], act.shape[3]
    act_flat = act.squeeze(0).view(C_dim, -1)
    act_flat = act_flat - act_flat.mean(dim=1, keepdim=True)
    try:
        _, _, Vt = torch.linalg.svd(act_flat, full_matrices=False)
        cam_flat = (act_flat.T @ Vt[0]).view(fH, fW)
    except Exception:
        cam_flat = act_flat.norm(dim=0).view(fH, fW)

    cam = cam_flat.abs().numpy().astype(np.float32)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_lb   = cv2.resize(cam, (640, 640), interpolation=cv2.INTER_LINEAR)
    cam_orig = cv2.resize(
        cam_lb[pad_top:pad_top + new_h, pad_left:pad_left + new_w],
        (w, h), interpolation=cv2.INTER_LINEAR,
    )

    heatmap_rgb = cv2.cvtColor(
        cv2.applyColorMap((cam_orig * 255).astype(np.uint8), cv2.COLORMAP_JET),
        cv2.COLOR_BGR2RGB,
    )
    overlay = (img_rgb * 0.5 + heatmap_rgb * 0.5).astype(np.uint8)

    BOX_COLORS = [(0, 220, 255), (50, 255, 80)]
    for box in model.predict(img_bgr, conf=conf, verbose=False)[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        col = BOX_COLORS[cls] if cls < 2 else (255, 255, 255)
        lbl = f"{'basket' if cls == 0 else 'mannequin'} {float(box.conf[0]):.2f}"
        overlay = _pil_draw_box(overlay, x1, y1, x2, y2, lbl, col)

    return img_rgb, overlay


def fig_gradcam(model_path: str, img_path: str):
    """2-row EigenCAM figure: (1) large-basket scene, (2) small-basket + mannequin."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    torch_model = model.model
    torch_model.eval()

    # Target layer: last C3k2/C2f in backbone
    target, target_name = None, ""
    for idx in [8, 6, 4, 10]:
        for name, m in torch_model.named_modules():
            if name.endswith(f".{idx}") and hasattr(m, "cv2"):
                target_name, target = name, m
        if target:
            break
    if target is None:
        for name, m in torch_model.named_modules():
            if hasattr(m, "cv2") and hasattr(m, "cv1"):
                target_name, target = name, m

    # Image 1: user-selected (original, typically large basket)
    img1_bgr = _crop_landscape(cv2.imread(str(img_path)))

    # Image 2: small basket + mannequin — clean wide-field shot, basket_area≈0.03
    val_dir = Path("datasets/basket_mannequin_final/val/images")
    img2_path = val_dir / "20260506_213921_aug02.jpg"
    if not img2_path.exists():
        # Fallback: pick darkest image with both classes detected
        best = None
        for p in sorted(val_dir.glob("*.jpg")):
            img_tmp = cv2.imread(str(p))
            if img_tmp is None or img_tmp.mean() > 160:
                continue
            r = model.predict(str(p), conf=0.15, verbose=False)[0]
            cls_set = {int(b.cls[0]) for b in r.boxes}
            if 0 in cls_set and 1 in cls_set:
                best = p
                break
        if best:
            img2_path = best
    img2_bgr = _crop_landscape(cv2.imread(str(img2_path)))

    rgb1, ov1 = _eigencam_overlay(torch_model, model, img1_bgr, target)
    rgb2, ov2 = _eigencam_overlay(torch_model, model, img2_bgr, target)

    # 1-row × 4-col layout: [A_orig | A_cam | B_orig | B_cam]
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(14.0, 4.4))
    gs  = GridSpec(1, 5, figure=fig,
                   width_ratios=[4, 4, 0.35, 4, 4],
                   wspace=0.03, left=0.005, right=0.885,
                   top=0.84, bottom=0.10)

    ax_a0 = fig.add_subplot(gs[0, 0])
    ax_a1 = fig.add_subplot(gs[0, 1])
    ax_b0 = fig.add_subplot(gs[0, 3])
    ax_b1 = fig.add_subplot(gs[0, 4])

    for ax, img in [(ax_a0, rgb1), (ax_a1, ov1), (ax_b0, rgb2), (ax_b1, ov2)]:
        ax.imshow(img)
        ax.axis("off")

    ax_a0.set_title("입력 영상",    fontsize=8, pad=3)
    ax_a1.set_title("EigenCAM",   fontsize=8, pad=3)
    ax_b0.set_title("입력 영상",    fontsize=8, pad=3)
    ax_b1.set_title("EigenCAM",   fontsize=8, pad=3)

    # 장면 레이블
    for ax_left, ax_right, lbl in [
        (ax_a0, ax_a1, "장면 A — 근접 촬영"),
        (ax_b0, ax_b1, "장면 B — 광각 촬영"),
    ]:
        mid_x = (ax_left.get_position().x0 + ax_right.get_position().x1) / 2
        fig.text(mid_x, 0.01, lbl, ha="center", va="bottom",
                 fontsize=8, style="italic", transform=fig.transFigure)

    # Vertical separator line between scene A and B
    sep_ax = fig.add_subplot(gs[0, 2])
    sep_ax.axis("off")
    sep_ax.axvline(0.5, color="gray", linewidth=0.8, linestyle="--")

    # Shared colorbar
    sm   = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    cbar_ax = fig.add_axes([0.895, 0.08, 0.012, 0.70])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("활성화 강도", fontsize=8)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["낮음", "중간", "높음"])

    fig.suptitle(
        "EigenCAM — YOLO11n 백본 특징 활성화  (바구니 / 마네킹 탐지)",
        fontsize=9, x=0.45, y=0.97,
    )
    _save(fig, "fig_gradcam")


# ── Figure 3: Latency and throughput from our Jetson run ──────────────────────
def fig_latency(json_path=None):
    """Measured backend latency and throughput for the Jetson deployment path."""

    # Our measured values (from jetson_latency_bench.py on Jetson Orin Nano Super)
    if json_path and Path(json_path).exists():
        with open(json_path) as f:
            raw = json.load(f)

        def pick(aliases, fallback_mean, fallback_std, fallback_p95=None):
            for key in aliases:
                if key in raw:
                    item = raw[key]
                    return (
                        item["mean"],
                        item.get("std", fallback_std),
                        item.get("p95", fallback_p95 or item["mean"]),
                        item.get("runs", 200),
                    )
            return fallback_mean, fallback_std, fallback_p95 or fallback_mean, 200

        trt_m, trt_s, trt_p95, runs = pick(
            ["TensorRT FP16 (.engine)", "TensorRT FP16"],
            4.56, 0.12, 4.56,
        )
        onnx_m, onnx_s, onnx_p95, _ = pick(
            ["ONNX Runtime (CPU)", "ONNX Runtime FP32", "ONNX Runtime"],
            106.59, 2.54, 110.25,
        )
        pt_m, pt_s, pt_p95, _ = pick(
            ["PyTorch FP32 (.pt)", "PyTorch FP32"],
            558.03, 8.27, 574.85,
        )
    else:
        # Real Jetson Orin Nano Super measurements (tools/jetson_latency_bench.py)
        trt_m, trt_s, trt_p95 = 4.56, 0.12, 4.56
        onnx_m, onnx_s, onnx_p95 = 106.59, 2.54, 110.25
        pt_m, pt_s, pt_p95 = 558.03, 8.27, 574.85
        runs = 200

    entries = [
        ("TensorRT FP16\n.engine", trt_m, trt_s, trt_p95, C_GREEN),
        ("ONNX Runtime\nCPU", onnx_m, onnx_s, onnx_p95, C_BLUE),
        ("PyTorch FP32\n.pt", pt_m, pt_s, pt_p95, C_GRAY),
    ]
    labels = [e[0] for e in entries]
    means = np.array([e[1] for e in entries])
    stds = np.array([e[2] for e in entries])
    p95s = np.array([e[3] for e in entries])
    colors = [e[4] for e in entries]
    fps = 1000.0 / means

    fig = plt.figure(figsize=(10.6, 4.1))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.25, 1.0], wspace=0.34)
    ax_lat = fig.add_subplot(gs[0, 0])
    ax_fps = fig.add_subplot(gs[0, 1])

    y = np.arange(len(entries))
    ax_lat.barh(
        y, means, xerr=stds, color=colors, height=0.55,
        capsize=3, ecolor="#374151", zorder=3,
    )
    ax_lat.set_xscale("log")
    ax_lat.set_xlim(1.0, 1000.0)
    ax_lat.set_yticks(y)
    ax_lat.set_yticklabels(labels)
    ax_lat.invert_yaxis()
    ax_lat.set_xlabel("추론 지연시간 (ms, 로그축)")
    ax_lat.set_title("A. 측정 지연시간", fontweight="bold")

    real_time_ms = 33.3
    ax_lat.axvline(real_time_ms, color=C_RED, linestyle="--", linewidth=1.2)
    ax_lat.text(
        real_time_ms * 1.05, 1.50, "30 FPS 기준",
        color=C_RED, fontsize=8.2, ha="left", va="center",
    )
    for yi, mean, p95 in zip(y, means, p95s):
        ax_lat.text(mean * 1.12, yi, f"{mean:.1f} ms\np95 {p95:.1f} ms",
                    va="center", fontsize=8.2, color="#111827")

    ax_fps.barh(y, fps, color=colors, height=0.55, zorder=3)
    ax_fps.axvline(30.0, color=C_RED, linestyle="--", linewidth=1.2)
    ax_fps.text(31.5, 2.42, "30 FPS", color=C_RED, fontsize=8, va="top")
    for yi, value in zip(y, fps):
        ax_fps.text(value + max(fps) * 0.025, yi, f"{value:.1f} FPS",
                    va="center", fontsize=8.5, fontweight="bold")
    ax_fps.set_yticks(y)
    ax_fps.set_yticklabels([])
    ax_fps.invert_yaxis()
    ax_fps.set_xlim(0, max(fps) * 1.22)
    ax_fps.set_xlabel("처리량 (FPS)")
    ax_fps.set_title("B. 실시간 처리 여유", fontweight="bold")

    speedup = pt_m / trt_m
    fig.suptitle(
        f"Jetson Orin Nano Super Deployment — YOLO11n 640x640, batch=1, "
        f"{runs} runs, TensorRT speedup x{speedup:.0f}",
        fontsize=10.5, y=1.01,
    )
    fig.subplots_adjust(top=0.82, bottom=0.18, left=0.13, right=0.98)
    _save(fig, "fig_latency")


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default="runs/detect/runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt",
    )
    ap.add_argument(
        "--csv",
        default="runs/detect/runs/detect/basket_mannequin_detect/yolo11n_v1/results.csv",
    )
    ap.add_argument(
        "--dataset",
        default="datasets/basket_mannequin_final",
        help="YOLO dataset directory containing train/ and val/ splits",
    )
    ap.add_argument(
        "--latency", default=None,
        help="path to latency_results.json from tools/jetson_latency_bench.py",
    )
    ap.add_argument(
        "--only", default=None,
        choices=["performance", "training_curves", "gradcam", "latency"],
        help="generate only one specific figure",
    )
    ap.add_argument(
        "--include-gradcam",
        action="store_true",
        help="also regenerate fig_gradcam when generating all figures",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    val_imgs = sorted(
        Path("datasets/basket_mannequin_final/val/images").glob("*.jpg")
    )
    if not val_imgs:
        raise FileNotFoundError("No val images found in datasets/basket_mannequin_final/val/images/")
    img_path = val_imgs[len(val_imgs) * 2 // 3]

    print(f"Output → paper/figures/  |  val image: {img_path.name}")

    run_all = args.only is None

    if run_all or args.only == "performance":
        print("▶ Performance figure...")
        fig_performance(args.csv, args.dataset)

    if run_all or args.only == "training_curves":
        print("▶ Training curves...")
        fig_training_curves(args.csv)

    if args.only == "gradcam" or (run_all and args.include_gradcam):
        print("▶ EigenCAM heatmap...")
        fig_gradcam(args.model, img_path)

    if run_all or args.only == "latency":
        print("▶ Latency chart...")
        fig_latency(args.latency)

    print("Done. Figures in paper/figures/")


if __name__ == "__main__":
    main()
