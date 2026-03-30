# vtol_vision

Jetson Nano + ROS2 Humble용 C++ 비전 모듈입니다.

## 기능
- ArUco(`DICT_4X4_50`) 인식, ID 화이트리스트 지원
- ArUco 결과: `ID + 6DoF Pose + reprojection_error + confidence`
- YOLO 비동기 추론 파이프라인
- `class_map.yaml` 기반 클래스 이름 매핑(코드 수정 없이 교체 가능)
- 디버그 이미지 토픽(옵션)

## 토픽/메시지
- `/vision/aruco` : `vtol_vision/msg/VisionDetections`
- `/vision/objects` : `vtol_vision/msg/VisionDetections`
- `/vision/debug_image` : `sensor_msgs/msg/Image` (옵션)

메시지 파일
- `msg/ArucoDetection.msg`
- `msg/ObjectDetection.msg`
- `msg/VisionDetections.msg`

## 빠른 실행
```bash
source /opt/ros/humble/setup.bash
cd /home/dev/vtol-vision
colcon build --packages-select vtol_vision
source install/setup.bash
ros2 launch vtol_vision vision.launch.py \
  camera_uri:=0 \
  trt_engine_path:=/path/to/model.onnx \
  class_map_yaml:=/home/dev/vtol-vision/config/class_map.example.yaml
```

기본 파라미터: `config/vision_params.yaml`
