# figj ROS2 C++ Runtime Bundle

드론 팀 전달용 최소 번들입니다. 학습 데이터셋, Python GUI, 학습 로그는 제외했고 ROS2 C++ 런타임에 필요한 것만 넣었습니다.

## Directory

```text
figj/
  vtol_vision_cpp/
    CMakeLists.txt
    package.xml
    src/
    include/
    msg/
    launch/
    config/
    weights/basket_mannequin_yolo11n_best.pt
    scripts/export_engine.sh
    scripts/run_vision_node.sh
```

## Classes

```text
0 basket
1 mannequin
```

`skyedge_box`는 `basket`, `skyedge_person`은 `mannequin`으로 통합해서 학습했습니다.

## Model

권장 weight:

```text
vtol_vision_cpp/weights/basket_mannequin_yolo11n_best.pt
```

검증 결과:

```text
all        mAP50 0.854 / mAP50-95 0.553
basket     mAP50 0.932 / mAP50-95 0.604
mannequin  mAP50 0.776 / mAP50-95 0.501
```

## On Drone / Jetson

TensorRT `.engine`은 GPU/JetPack/TensorRT 버전에 묶이므로 대상 드론에서 직접 생성해야 합니다.

```bash
cd <ros2_ws>/src
cp -r /path/to/figj/vtol_vision_cpp ./vtol_vision
cd <ros2_ws>
colcon build --packages-select vtol_vision
source install/setup.bash
```

Export TensorRT engine on the target:

```bash
cd <ros2_ws>/src/vtol_vision
./scripts/export_engine.sh
```

Run:

```bash
source <ros2_ws>/install/setup.bash
ros2 launch vtol_vision vision.launch.py \
  camera_uri:=0 \
  trt_engine_path:=<ros2_ws>/src/vtol_vision/weights/basket_mannequin_yolo11n_best.engine
```

Or:

```bash
cd <ros2_ws>/src/vtol_vision
./scripts/run_vision_node.sh
```

## ROS Topics

```text
/vision/objects      vtol_vision/msg/VisionDetections
/vision/aruco        vtol_vision/msg/VisionDetections
/vision/debug_image  sensor_msgs/msg/Image, only if enable_debug_image=true
```

## Main Parameters

```text
camera_uri          camera index or GStreamer pipeline
trt_engine_path     absolute path to .engine
class_map_yaml      config/basket_mannequin_class_map.yaml
conf_thr            detection confidence threshold
nms_thr             NMS IoU threshold
yolo_input_size     model input size, 640 for exported engine
yolo_period_ms      minimum YOLO loop period
```
