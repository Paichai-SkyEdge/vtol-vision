#!/usr/bin/env python3
"""Generate a one-page report figure for the mpd_v2 detector.

The qualitative examples come from the training split.  Quantitative metrics
come from the held-out validation results saved by the matching training run.

Outputs:
  paper/figures/fig_mpd_v2_summary.png
  paper/figures/fig_mpd_v2_summary.pdf
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Rectangle
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "mpd_v2/weights/yolo11n_shadow_v1_best.pt"
DEFAULT_DATASET = ROOT / "datasets/skyedge_all_yolo_shadow"
DEFAULT_RESULTS = ROOT / "runs/detect/runs/skyedge/yolo11n_shadow_v1/results.csv"
DEFAULT_OUT = ROOT / "paper/figures/fig_mpd_v2_summary"
CLASS_NAMES = ("basket", "mannequin")
COLORS = {0: "#00A6D6", 1: "#F97316"}
KO_FONT = Path("/usr/share/fonts/truetype/unfonts-core/UnDotum.ttf")

if KO_FONT.exists():
    font_manager.fontManager.addfont(str(KO_FONT))

plt.rcParams.update(
    {
        "font.family": "UnDotum" if KO_FONT.exists() else "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    }
)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    labels: tuple[tuple[int, float, float, float, float], ...]
    role: str


def read_labels(label_path: Path) -> tuple[tuple[int, float, float, float, float], ...]:
    labels = []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            fields = line.split()
            if len(fields) >= 5:
                labels.append((int(float(fields[0])), *(float(x) for x in fields[1:5])))
    return tuple(labels)


def image_for_label(label_path: Path, image_dir: Path) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        path = image_dir / f"{label_path.stem}{suffix}"
        if path.exists():
            return path
    return None


def choose_samples(dataset: Path) -> list[Sample]:
    """Choose deterministic, visually useful class/object-count combinations."""
    image_dir = dataset / "train/images"
    candidates = []
    for label_path in sorted((dataset / "train/labels").glob("*.txt")):
        labels = read_labels(label_path)
        image_path = image_for_label(label_path, image_dir)
        if not labels or image_path is None:
            continue
        classes = {x[0] for x in labels}
        mean_area = float(np.mean([x[3] * x[4] for x in labels]))
        candidates.append((image_path, labels, classes, mean_area))

    if not candidates:
        raise RuntimeError(f"no labeled training samples found under {dataset}")

    specs = [
        ("basket 단일 표적", lambda x: x[2] == {0} and len(x[1]) == 1),
        ("mannequin 단일 표적", lambda x: x[2] == {1} and len(x[1]) == 1),
        ("두 클래스 동시 탐지", lambda x: x[2] == {0, 1}),
        ("다중 표적 장면", lambda x: len(x[1]) >= 3),
    ]
    chosen: list[Sample] = []
    used: set[Path] = set()
    for role, predicate in specs:
        matches = [x for x in candidates if predicate(x) and x[0] not in used]
        if not matches:
            matches = [x for x in candidates if x[0] not in used]
        # Prefer a medium object scale so boxes and scene context are both visible.
        match = min(matches, key=lambda x: (abs(x[3] - 0.08), x[0].name))
        chosen.append(Sample(match[0], match[1], role))
        used.add(match[0])
    return chosen


def best_epoch_metrics(results_csv: Path) -> tuple[dict[str, float], list[dict[str, float]]]:
    with results_csv.open() as f:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(f)]
    if not rows:
        raise RuntimeError(f"empty training results: {results_csv}")
    best = max(rows, key=lambda row: row["metrics/mAP50-95(B)"])
    return best, rows


def dataset_counts(dataset: Path) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for split in ("train", "valid", "test"):
        image_dir = dataset / split / "images"
        label_dir = dataset / split / "labels"
        images = [p for p in image_dir.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        instances = [0, 0]
        for label_path in label_dir.glob("*.txt"):
            for label in read_labels(label_path):
                if 0 <= label[0] < 2:
                    instances[label[0]] += 1
        stats[split] = {"images": len(images), "instances": instances}
    return stats


def yolo_xywh_to_xyxy(label, width: int, height: int):
    _, cx, cy, bw, bh = label
    return ((cx - bw / 2) * width, (cy - bh / 2) * height, bw * width, bh * height)


def draw_sample(ax, sample: Sample, result, index: int):
    bgr = cv2.imread(str(sample.image_path))
    if bgr is None:
        raise RuntimeError(f"failed to read {sample.image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    ax.imshow(rgb)

    # Ground truth: dashed and slightly thinner. Prediction: solid with confidence.
    for label in sample.labels:
        cls = label[0]
        x, y, w, h = yolo_xywh_to_xyxy(label, width, height)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=COLORS[cls],
                               linewidth=1.35, linestyle=(0, (3, 2))))

    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        color = COLORS.get(cls, "#FFFFFF")
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                               edgecolor=color, linewidth=2.2))
        ax.text(x1, max(0, y1 - 3), f"{CLASS_NAMES[cls]} {conf:.2f}", color="white",
                fontsize=7.4, va="bottom", ha="left",
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.92, pad=1.6))

    ax.set_title(f"{chr(65 + index)}. {sample.role}", loc="left", fontweight="bold", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#D1D5DB")
        spine.set_linewidth(0.8)


def metric_card(ax, label: str, value: float, color: str, subtitle: str = ""):
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.84, transform=ax.transAxes,
                           facecolor="#F8FAFC", edgecolor="#DDE3EA", linewidth=0.8))
    ax.text(0.5, 0.68, label, ha="center", va="center", color="#475569",
            fontsize=8.5, transform=ax.transAxes)
    ax.text(0.5, 0.43, f"{value * 100:.1f}%", ha="center", va="center", color=color,
            fontsize=18, fontweight="bold", transform=ax.transAxes)
    if subtitle:
        ax.text(0.5, 0.19, subtitle, ha="center", va="center", color="#64748B",
                fontsize=6.8, transform=ax.transAxes)


def make_figure(model_path: Path, dataset: Path, results_csv: Path, output: Path, conf: float):
    samples = choose_samples(dataset)
    model = YOLO(str(model_path))
    predictions = model.predict([str(x.image_path) for x in samples], imgsz=640, conf=conf,
                                iou=0.7, device="cpu", verbose=False)
    best, rows = best_epoch_metrics(results_csv)
    stats = dataset_counts(dataset)

    precision = best["metrics/precision(B)"]
    recall = best["metrics/recall(B)"]
    f1 = 2 * precision * recall / (precision + recall)
    map50 = best["metrics/mAP50(B)"]
    map5095 = best["metrics/mAP50-95(B)"]

    fig = plt.figure(figsize=(12.0, 8.0), facecolor="white")
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.12, 1.63, 1.0], hspace=0.21)

    title_ax = fig.add_subplot(outer[0])
    title_ax.axis("off")
    title_ax.text(0.0, 0.74, "MPD-V2 OBJECT DETECTOR", fontsize=8.5, fontweight="bold",
                  color="#2563EB", transform=title_ax.transAxes)
    title_ax.text(0.0, 0.18, "Basket / Mannequin 탐지 정성·정량 결과", fontsize=19,
                  fontweight="bold", color="#111827", transform=title_ax.transAxes)
    title_ax.text(1.0, 0.2, "YOLO11n · 640 px · confidence ≥ 0.25", ha="right",
                  color="#64748B", fontsize=8.5, transform=title_ax.transAxes)

    samples_gs = GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], wspace=0.08, hspace=0.18)
    for i, (sample, prediction) in enumerate(zip(samples, predictions)):
        draw_sample(fig.add_subplot(samples_gs[i // 2, i % 2]), sample, prediction, i)

    bottom = GridSpecFromSubplotSpec(2, 6, subplot_spec=outer[2], height_ratios=[0.62, 1.0],
                                    hspace=0.28, wspace=0.18)
    metric_card(fig.add_subplot(bottom[0, 0]), "Precision", precision, "#0F766E")
    metric_card(fig.add_subplot(bottom[0, 1]), "Recall", recall, "#C2410C")
    metric_card(fig.add_subplot(bottom[0, 2]), "F1 score", f1, "#7C3AED")
    metric_card(fig.add_subplot(bottom[0, 3]), "mAP@50", map50, "#2563EB")
    metric_card(fig.add_subplot(bottom[0, 4]), "mAP@50–95", map5095, "#1D4ED8")
    info_ax = fig.add_subplot(bottom[0, 5])
    info_ax.axis("off")
    info_ax.add_patch(Rectangle((0.02, 0.08), 0.96, 0.84, transform=info_ax.transAxes,
                                facecolor="#111827", edgecolor="#111827"))
    info_ax.text(0.5, 0.66, "Best validation", color="#CBD5E1", ha="center", fontsize=7.5,
                 transform=info_ax.transAxes)
    info_ax.text(0.5, 0.39, f"epoch {int(best['epoch'])}", color="white", ha="center",
                 fontsize=14, fontweight="bold", transform=info_ax.transAxes)
    info_ax.text(0.5, 0.18, "held-out split", color="#94A3B8", ha="center", fontsize=6.8,
                 transform=info_ax.transAxes)

    # Compact learning curves.
    curve_ax = fig.add_subplot(bottom[1, :3])
    epochs = [int(x["epoch"]) for x in rows]
    curve_ax.plot(epochs, [x["metrics/mAP50(B)"] * 100 for x in rows], marker="o",
                  color="#2563EB", label="mAP@50")
    curve_ax.plot(epochs, [x["metrics/mAP50-95(B)"] * 100 for x in rows], marker="o",
                  color="#7C3AED", label="mAP@50–95")
    curve_ax.set_title("E. Fine-tuning validation 성능", loc="left", fontweight="bold")
    curve_ax.set_xlabel("Epoch")
    curve_ax.set_ylabel("Score (%)")
    curve_ax.set_xticks(epochs)
    curve_ax.grid(True, alpha=0.2, linewidth=0.6)
    curve_ax.legend(frameon=False, ncol=2, loc="lower right", fontsize=7.5)

    # Dataset composition by image and object instances.
    data_ax = fig.add_subplot(bottom[1, 3:5])
    splits = ("train", "valid", "test")
    x = np.arange(3)
    baskets = [stats[s]["instances"][0] for s in splits]
    mannequins = [stats[s]["instances"][1] for s in splits]
    data_ax.bar(x, baskets, color=COLORS[0], label="basket")
    data_ax.bar(x, mannequins, bottom=baskets, color=COLORS[1], label="mannequin")
    data_ax.set_xticks(x, [f"{s}\n(n={stats[s]['images']})" for s in splits])
    data_ax.set_ylabel("Instances")
    data_ax.set_title("F. 데이터셋 구성", loc="left", fontweight="bold")
    data_ax.grid(axis="y", alpha=0.2, linewidth=0.6)
    data_ax.legend(frameon=False, fontsize=7.5, ncol=2)

    note_ax = fig.add_subplot(bottom[1, 5])
    note_ax.axis("off")
    note_ax.text(0.0, 0.96, "표기 안내", fontweight="bold", fontsize=9, va="top",
                 transform=note_ax.transAxes)
    note_ax.text(0.0, 0.72, "실선  모델 예측\n점선  Ground truth", fontsize=8, linespacing=1.55,
                 color="#334155", transform=note_ax.transAxes)
    note_ax.plot([0.0, 0.22], [0.40, 0.40], color=COLORS[0], lw=4,
                 transform=note_ax.transAxes, clip_on=False)
    note_ax.text(0.28, 0.40, "basket", va="center", fontsize=8, transform=note_ax.transAxes)
    note_ax.plot([0.0, 0.22], [0.22, 0.22], color=COLORS[1], lw=4,
                 transform=note_ax.transAxes, clip_on=False)
    note_ax.text(0.28, 0.22, "mannequin", va="center", fontsize=8, transform=note_ax.transAxes)

    fig.text(0.5, 0.012,
             "정성 예시는 training split, 정량 지표는 held-out validation split 기준 · "
             f"model: {model_path.name}", ha="center", fontsize=7.2, color="#64748B")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    print(f"saved: {output.with_suffix('.png')}")
    print(f"saved: {output.with_suffix('.pdf')}")
    print("samples:")
    for sample in samples:
        print(f"  {sample.role}: {sample.image_path.relative_to(ROOT)}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT,
                        help="output path without extension")
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.model, args.dataset, args.results, args.output, args.conf)
