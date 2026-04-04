"""
Train a single-class mannequin detector with YOLO11n.

Usage:
  python3 tools/train_mannequin_yolo.py --mode smoke
  python3 tools/train_mannequin_yolo.py --mode full
"""

import argparse
import os
from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.data import utils as data_utils
from ultralytics.utils import checks


DATASET_YAML = Path("./datasets/merged/dataset.yaml")
MODEL = "yolo11n.pt"
PROJECT_NAME = "mannequin_detect"
PROJECT_DIR = Path("runs/detect") / PROJECT_NAME
RUN_NAME = "yolo11n_v1"

FULL_EPOCHS = 100
SMOKE_EPOCHS = 5
IMG_SIZE = 640
BATCH = 16
WORKERS = 4


def parse_args():
    parser = argparse.ArgumentParser(description="Train mannequin detector with YOLO11n")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--device", default=None, help="Training device, e.g. 0 or cpu")
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional checkpoint or weight file to initialize training from.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Optional run name override.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the selected run from its last checkpoint if available.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Explicit resumable checkpoint path, e.g. runs/.../weights/epoch60.pt",
    )
    return parser.parse_args()


def metric_value(results, key: str) -> str:
    value = results.results_dict.get(key)
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def validate_inputs():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"dataset yaml not found: {DATASET_YAML.resolve()}")

    image_label_pairs = [
        ("datasets/merged/train/images", "datasets/merged/train/labels"),
        ("datasets/merged/val/images", "datasets/merged/val/labels"),
    ]
    for image_dir, label_dir in image_label_pairs:
        image_count = len([p for p in Path(image_dir).iterdir() if p.is_file()])
        label_count = len(list(Path(label_dir).glob("*.txt")))
        if image_count != label_count:
            raise RuntimeError(
                f"image/label count mismatch: {image_dir}={image_count}, {label_dir}={label_count}"
            )

    if not Path(MODEL).exists():
        raise FileNotFoundError(f"base model not found: {Path(MODEL).resolve()}")


def disable_optional_font_check():
    # Avoid importing matplotlib during dataset validation in environments
    # where the system matplotlib build is incompatible with the active NumPy.
    checks.check_font = lambda *args, **kwargs: None
    data_utils.check_font = checks.check_font


def configure_ultralytics_dirs():
    config_home = Path(".ultralytics_config").resolve()
    config_home.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(config_home)


def ensure_resumable_checkpoint(checkpoint_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"checkpoint is not resumable: {checkpoint_path}")

    required_keys = ["epoch", "optimizer", "train_args"]
    missing = [key for key in required_keys if key not in checkpoint or checkpoint[key] is None]
    if missing:
        raise RuntimeError(
            "resume checkpoint is missing training state "
            f"({', '.join(missing)}): {checkpoint_path}. "
            "Use --weights for fine-tuning from a stripped checkpoint like last.pt."
        )


def main():
    args = parse_args()
    configure_ultralytics_dirs()
    validate_inputs()
    disable_optional_font_check()

    epochs = args.epochs if args.epochs is not None else (
        SMOKE_EPOCHS if args.mode == "smoke" else FULL_EPOCHS
    )
    run_name = args.name or (f"{RUN_NAME}_{args.mode}" if args.mode == "smoke" else RUN_NAME)
    last_checkpoint = PROJECT_DIR / run_name / "weights" / "last.pt"
    resume_checkpoint = Path(args.resume_checkpoint) if args.resume_checkpoint else last_checkpoint

    print("=" * 50)
    print("YOLO11n training start")
    print(f"  mode   : {args.mode}")
    print(f"  model  : {MODEL}")
    print(f"  data   : {DATASET_YAML.resolve()}")
    print(f"  epochs : {epochs}")
    print(f"  batch  : {args.batch}")
    print(f"  device : {args.device or 'auto'}")
    print(f"  resume : {args.resume}")
    if args.resume:
        print(f"  ckpt   : {resume_checkpoint}")
    print("=" * 50)

    if args.resume:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_checkpoint.resolve()}")
        ensure_resumable_checkpoint(resume_checkpoint)
        model = YOLO(str(resume_checkpoint))
        results = model.train(resume=True, device=args.device)
    else:
        model_source = args.weights or MODEL
        model = YOLO(model_source)
        results = model.train(
            data=str(DATASET_YAML),
            epochs=epochs,
            imgsz=IMG_SIZE,
            batch=args.batch,
            workers=args.workers,
            project=PROJECT_NAME,
            name=run_name,
            exist_ok=True,
            degrees=45,
            fliplr=0.5,
            flipud=0.3,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            scale=0.5,
            mosaic=1.0,
            mixup=0.1,
            patience=20,
            save_period=10,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            plots=False,
            device=args.device,
        )

    print("\n" + "=" * 50)
    print("Training complete")
    print("=" * 50)
    print(f"  run path : {PROJECT_NAME}/{run_name}")
    print(f"  best pt  : {PROJECT_NAME}/{run_name}/weights/best.pt")
    print(f"  mAP@50   : {metric_value(results, 'metrics/mAP50(B)')}")
    print(f"  precision: {metric_value(results, 'metrics/precision(B)')}")
    print(f"  recall   : {metric_value(results, 'metrics/recall(B)')}")


if __name__ == "__main__":
    main()
