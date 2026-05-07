#!/usr/bin/env python3
"""
Train a two-class basket/mannequin detector with Ultralytics YOLO.

Usage:
  python3 tools/prepare_basket_mannequin_dataset.py
  python3 tools/train_basket_mannequin_yolo.py --mode smoke
  python3 tools/train_basket_mannequin_yolo.py --mode full
"""

import argparse
import os
from pathlib import Path

from ultralytics import YOLO
from ultralytics.data import utils as data_utils
from ultralytics.utils import checks


DATASET_YAML = Path("datasets/basket_mannequin_final/dataset.yaml")
MODEL = "yolo11n.pt"
PROJECT_NAME = "basket_mannequin_detect"
PROJECT_DIR = Path("runs/detect") / PROJECT_NAME
RUN_NAME = "yolo11n_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Train basket/mannequin detector")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default=None, help="Training device, e.g. 0 or cpu")
    parser.add_argument("--weights", default=None, help="Initial .pt weights")
    parser.add_argument("--name", default=None, help="Run name override")
    return parser.parse_args()


def configure_ultralytics_dirs():
    config_home = Path(".ultralytics_config").resolve()
    config_home.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CONFIG_HOME"] = str(config_home)


def disable_optional_font_check():
    checks.check_font = lambda *args, **kwargs: None
    data_utils.check_font = checks.check_font


def validate_inputs():
    if not DATASET_YAML.exists():
        raise FileNotFoundError(f"dataset yaml not found: {DATASET_YAML.resolve()}")


def main():
    args = parse_args()
    configure_ultralytics_dirs()
    disable_optional_font_check()
    validate_inputs()

    epochs = args.epochs if args.epochs is not None else (5 if args.mode == "smoke" else 100)
    run_name = args.name or (f"{RUN_NAME}_{args.mode}" if args.mode == "smoke" else RUN_NAME)
    model_source = args.weights or MODEL

    print("=" * 50)
    print("YOLO training start")
    print(f"  data   : {DATASET_YAML.resolve()}")
    print(f"  model  : {model_source}")
    print(f"  run    : {PROJECT_DIR / run_name}")
    print(f"  epochs : {epochs}")
    print(f"  device : {args.device or 'auto'}")
    print("=" * 50)

    model = YOLO(model_source)
    results = model.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=640,
        batch=args.batch,
        workers=args.workers,
        project=str(PROJECT_DIR),
        name=run_name,
        exist_ok=True,
        degrees=25,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        scale=0.5,
        mosaic=1.0,
        patience=20,
        plots=False,
        device=args.device,
    )

    print("\nTraining complete")
    print(f"  best: {PROJECT_DIR / run_name / 'weights' / 'best.pt'}")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")


if __name__ == "__main__":
    main()
