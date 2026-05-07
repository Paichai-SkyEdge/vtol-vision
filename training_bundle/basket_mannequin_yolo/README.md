# Basket + Mannequin YOLO Training Bundle

This folder is self-contained. Copy it to a GPU machine or Jetson and train from here.

## Dataset

- Classes: `0 basket`, `1 mannequin`
- Dataset YAML: `dataset.yaml`
- ROS class map after export: `config/class_map.yaml`

## Train

```bash
cd basket_mannequin_yolo
DEVICE=0 BATCH=8 EPOCHS=100 scripts/train.sh
```

For a quick check:

```bash
DEVICE=cpu BATCH=4 EPOCHS=1 RUN_NAME=smoke scripts/train.sh
```

The best checkpoint is written to:

```txt
runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt
```

## Export

Create TensorRT engine on the Jetson or target GPU machine:

```bash
scripts/export_engine.sh
```

Create ONNX:

```bash
scripts/export_onnx.sh
```

Override the model path if needed:

```bash
MODEL_PATH=runs/detect/basket_mannequin_detect/yolo11n_v1/weights/best.pt scripts/export_engine.sh
```

TensorRT `.engine` files are hardware/JetPack specific. Export them on the machine that will run inference.
