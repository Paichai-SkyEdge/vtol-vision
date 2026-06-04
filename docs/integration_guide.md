# 타 부서 연동 가이드

이 문서는 비전팀이 제공하는 `vtol_vision` 패키지를 다른 부서의 ROS2 시스템에서 구독하고 연결하기 위한 한국어 인터페이스 문서입니다.

## 제공 범위

`vtol_vision`은 카메라 프레임을 직접 읽어서 다음 결과를 ROS2 토픽으로 발행합니다.

| 기능 | 출력 |
|---|---|
| ArUco 마커 탐지 | `/vision/aruco` |
| YOLO 객체 탐지 | `/vision/objects` |
| 디버그 이미지 | `/vision/debug_image`, 옵션 |

카메라 입력, TensorRT engine 경로, class map, 임계값은 launch argument 또는 parameter YAML로 설정합니다.

## 연동 전 준비물

| 항목 | 필요 여부 | 비고 |
|---|---|---|
| ROS2 Humble workspace | 필수 | 이 패키지를 같은 workspace에 포함하거나 binary package로 설치 |
| `vtol_vision` 메시지 타입 | 필수 | `VisionDetections`, `ArucoDetection`, `ObjectDetection` |
| TensorRT `.engine` | YOLO 사용 시 필수 | Jetson/대상 GPU에서 생성 |
| 카메라 보정 YAML | ArUco pose 사용 시 권장 | 없으면 마커 ID만 신뢰, pose 품질 저하 |
| class map YAML | 객체 class 이름 커스텀 시 권장 | 기본값은 코드 내 fallback 사용 |

## 실행 계약

기본 실행:

```bash
ros2 launch vtol_vision vision.launch.py \
  trt_engine_path:=/absolute/path/to/weights/mannequin_yolo11n/best.engine \
  camera_uri:=0
```

레포에 포함된 모델:

| 경로 | 용도 |
|---|---|
| `weights/mannequin_yolo11n/best.pt` | ROS 노드 기본 단일 mannequin detector export 원본 |
| `weights/mannequin_yolo11n/best.onnx` | 단일 mannequin detector ONNX 검증용 |
| `weights/basket_mannequin_yolo11n/best.pt` | RealSense/학습 검증용 basket+mannequin detector |
| `weights/basket_mannequin_yolo11n/best.onnx` | basket+mannequin detector ONNX 검증용 |

TensorRT engine 생성 예시:

```bash
cd ~/ros2_ws/src/vtol_vision
yolo export model=weights/mannequin_yolo11n/best.pt format=engine device=0 imgsz=640 half=True
```

파라미터 파일을 따로 넘기는 방식:

```bash
ros2 launch vtol_vision vision.launch.py \
  params_file:=/absolute/path/to/vision_params.yaml \
  trt_engine_path:=/absolute/path/to/weights/mannequin_yolo11n/best.engine \
  class_map_yaml:=/absolute/path/to/class_map.yaml
```

CSI 카메라 예시:

```bash
ros2 launch vtol_vision vision.launch.py \
  trt_engine_path:=/absolute/path/to/weights/mannequin_yolo11n/best.engine \
  "camera_uri:=nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"
```

## 발행 토픽

| 토픽 | 타입 | 발행 조건 | 설명 |
|---|---|---|---|
| `/vision/aruco` | `vtol_vision/msg/VisionDetections` | 카메라 루프마다 | ArUco 탐지 결과. 탐지가 없어도 빈 배열로 발행 |
| `/vision/objects` | `vtol_vision/msg/VisionDetections` | YOLO ready 상태에서 새 프레임마다 | 객체 탐지 결과 |
| `/vision/debug_image` | `sensor_msgs/msg/Image` | `enable_debug_image=true`일 때 | ArUco overlay가 그려진 BGR8 이미지 |

`/vision/objects` 발행 주기는 `yolo_period_ms`와 실제 추론 시간에 의해 결정됩니다. 기본값은 50ms입니다.

## 메시지 구조

### VisionDetections

```txt
std_msgs/Header header
ArucoDetection[] aruco_detections
ObjectDetection[] object_detections
float32 pipeline_latency_ms
```

| 필드 | 의미 |
|---|---|
| `header.stamp` | 카메라 프레임 캡처 시각 |
| `header.frame_id` | 기본값 `camera`, `frame_id` 파라미터로 변경 |
| `aruco_detections` | ArUco 결과 배열. `/vision/objects`에서는 보통 비어 있음 |
| `object_detections` | YOLO 결과 배열. `/vision/aruco`에서는 보통 비어 있음 |
| `pipeline_latency_ms` | 캡처 시각부터 발행 직전까지 걸린 시간 |

### ArucoDetection

```txt
std_msgs/Header header
int32 marker_id
geometry_msgs/Pose pose
float32 reprojection_error
float32 confidence
```

| 필드 | 의미 |
|---|---|
| `marker_id` | 감지된 ArUco ID |
| `pose.position` | 카메라 좌표계 기준 마커 위치, 단위 m |
| `pose.orientation` | 카메라 좌표계 기준 마커 자세 quaternion |
| `reprojection_error` | 평균 reprojection error, px. 보정 없음이면 `-1` |
| `confidence` | `1 / (1 + reprojection_error)`. 보정 없음이면 `0` |

ArUco dictionary는 `DICT_5X5_50`입니다. `aruco_allowed_ids`가 비어 있으면 모든 ID를 허용합니다.

### ObjectDetection

```txt
std_msgs/Header header
int32 class_id
string class_name
float32 score
sensor_msgs/RegionOfInterest bbox
```

| 필드 | 의미 |
|---|---|
| `class_id` | YOLO class index |
| `class_name` | class map으로 해석된 이름 |
| `score` | 객체 confidence |
| `bbox.x_offset` | 원본 프레임 기준 좌상단 x 픽셀 |
| `bbox.y_offset` | 원본 프레임 기준 좌상단 y 픽셀 |
| `bbox.width` | bounding box width, px |
| `bbox.height` | bounding box height, px |
| `bbox.do_rectify` | 현재 항상 `false` |

좌표는 letterbox 보정 후 원본 프레임 크기로 복원된 픽셀 좌표입니다.

## 주요 파라미터

| 파라미터 | 기본값 | 연동 영향 |
|---|---|---|
| `camera_uri` | `"0"` | 카메라 index 또는 GStreamer pipeline |
| `camera_info_yaml` | `""` | ArUco pose/reprojection 계산용 보정 파일 |
| `frame_id` | `"camera"` | 모든 detection header frame |
| `frame_width` / `frame_height` | `640` / `480` | 카메라 캡처 요청 해상도 |
| `camera_fps` | `30` | 카메라 캡처 요청 FPS |
| `marker_size_m` | `0.15` | ArUco pose scale |
| `aruco_allowed_ids` | `[]` | 허용 마커 ID 목록 |
| `aruco_min_perimeter_rate` | `0.02` | 작은 마커 탐지 허용 정도 |
| `trt_engine_path` | `""` | YOLO TensorRT engine 절대경로 |
| `class_map_yaml` | `class_map.example.yaml` | class id to name 매핑 |
| `conf_thr` | `0.25` | YOLO confidence threshold |
| `nms_thr` | `0.45` | YOLO NMS IoU threshold |
| `yolo_input_size` | `640` | YOLO 입력 크기 |
| `yolo_period_ms` | `50` | YOLO 루프 최소 주기 |
| `queue_size` | `10` | publisher queue depth |
| `enable_debug_image` | `false` | 디버그 이미지 발행 여부 |
| `enable_yolo_debug_log` | `true` | YOLO 요약 로그 출력 |
| `undistort_image` | `true` | 보정 파일이 있을 때 undistort 적용 |

## 구독 예시

### Python

```python
import rclpy
from rclpy.node import Node
from vtol_vision.msg import VisionDetections


class VisionConsumer(Node):
    def __init__(self):
        super().__init__("vision_consumer")
        self.create_subscription(
            VisionDetections,
            "/vision/objects",
            self.on_objects,
            10,
        )

    def on_objects(self, msg):
        for det in msg.object_detections:
            self.get_logger().info(
                f"{det.class_name} score={det.score:.2f} "
                f"bbox=({det.bbox.x_offset},{det.bbox.y_offset},"
                f"{det.bbox.width},{det.bbox.height})"
            )


def main():
    rclpy.init()
    rclpy.spin(VisionConsumer())
    rclpy.shutdown()


if __name__ == "__main__":
    main()
```

### C++

```cpp
#include "vtol_vision/msg/vision_detections.hpp"

#include <rclcpp/rclcpp.hpp>

class VisionConsumer : public rclcpp::Node
{
public:
  VisionConsumer() : Node("vision_consumer")
  {
    sub_ = create_subscription<vtol_vision::msg::VisionDetections>(
      "/vision/objects",
      10,
      [this](const vtol_vision::msg::VisionDetections::SharedPtr msg) {
        for (const auto & det : msg->object_detections) {
          RCLCPP_INFO(
            get_logger(),
            "%s score=%.2f bbox=(%u,%u,%u,%u)",
            det.class_name.c_str(),
            det.score,
            det.bbox.x_offset,
            det.bbox.y_offset,
            det.bbox.width,
            det.bbox.height);
        }
      });
  }

private:
  rclcpp::Subscription<vtol_vision::msg::VisionDetections>::SharedPtr sub_;
};
```

## 연동 확인 절차

1. 비전 노드를 실행합니다.
2. 토픽 목록을 확인합니다.

```bash
ros2 topic list | grep /vision
```

3. 메시지 타입을 확인합니다.

```bash
ros2 topic info /vision/objects
ros2 interface show vtol_vision/msg/VisionDetections
```

4. 발행 주기와 payload를 확인합니다.

```bash
ros2 topic hz /vision/objects
ros2 topic echo /vision/objects --once
```

5. 타 부서 노드에서 `header.frame_id`, `header.stamp`, `bbox` 좌표계를 그대로 사용해 후속 로직에 연결합니다.

## 연동 시 주의사항

| 항목 | 주의 |
|---|---|
| 시간 동기화 | `header.stamp`는 비전 노드가 캡처한 시각입니다. 다른 센서와 fusion하려면 장비 시간 동기화를 맞추세요. |
| 좌표계 | ArUco pose는 카메라 optical frame 기준입니다. 기체 좌표계 변환은 소비 부서에서 TF 또는 별도 변환으로 처리합니다. |
| 빈 배열 | 탐지가 없으면 배열이 비어 있을 수 있습니다. 빈 배열을 정상 상태로 처리하세요. |
| YOLO disabled | engine이 없거나 로드 실패하면 `/vision/objects`가 발행되지 않을 수 있습니다. |
| debug image | 비행/운영 중에는 대역폭을 위해 기본 `false`를 권장합니다. |
| 메시지 변경 | `msg/*.msg` 변경은 연동부 breaking change입니다. 변경 시 버전/문서/소비 코드 업데이트가 필요합니다. |

## 인수인계 체크리스트

타 부서에 넘길 때 아래 항목을 같이 전달하세요.

| 항목 | 전달 예시 |
|---|---|
| 레포/브랜치 | `<repo-url>`, `main` 또는 릴리스 태그 |
| 실행 장비 | Jetson Orin Nano Super, JetPack 6.x |
| engine 경로 | `/home/<user>/ros2_ws/src/vtol_vision/weights/mannequin_yolo11n/best.engine` |
| class map | `config/class_map.example.yaml` 또는 프로젝트별 YAML |
| 카메라 입력 | USB index 또는 CSI GStreamer pipeline |
| 카메라 보정 | `camera_info_yaml` 경로 |
| 구독 토픽 | `/vision/objects`, `/vision/aruco` |
| 기대 FPS/latency | `ros2 topic hz`, `pipeline_latency_ms` 기준 |
| 담당 연락점 | 비전 담당자/모델 담당자/장비 담당자 |
