# vtol_vision

ROS2 Humble 기반 C++ 비전 모듈. VTOL 기체의 Jetson Orin Nano Super에 탑재하여 ArUco 마커 추적과 YOLO 객체 탐지를 실시간으로 수행합니다.

---

## 기능

| 기능 | 설명 |
|---|---|
| ArUco 탐지 | `DICT_5X5_50`, ID 화이트리스트 필터링, 6-DoF 포즈 추정 |
| YOLO 추론 | TensorRT `.engine` 직접 로드, 비동기 추론 루프 |
| 클래스 매핑 | `class_map.yaml`로 코드 수정 없이 클래스 이름 교체 |
| 디버그 이미지 | `/vision/debug_image` 토픽 (옵션) |

---

## 아키텍처

```
카메라
  │
  ▼
CaptureLoop (thread)
  ├─ ArUco 탐지 → /vision/aruco
  ├─ 프레임 공유 버퍼 (mutex)
  └─ 디버그 이미지 → /vision/debug_image (옵션)

YoloLoop (thread)
  ├─ 공유 버퍼에서 최신 프레임 가져오기
  ├─ Letterbox → FP32 NCHW 변환
  ├─ TensorRT 추론 (GPU)
  ├─ 후처리 + NMS
  └─ /vision/objects
```

---

## 토픽

| 토픽 | 타입 | 설명 |
|---|---|---|
| `/vision/aruco` | `vtol_vision/VisionDetections` | ArUco 탐지 결과 |
| `/vision/objects` | `vtol_vision/VisionDetections` | YOLO 객체 탐지 결과 |
| `/vision/debug_image` | `sensor_msgs/Image` | 디버그 오버레이 (옵션) |

**메시지 구조**
- `VisionDetections` — 헤더 + `ArucoDetection[]` + `ObjectDetection[]` + `pipeline_latency_ms`
- `ArucoDetection` — `marker_id`, `pose` (6-DoF), `reprojection_error`, `confidence`
- `ObjectDetection` — `class_id`, `class_name`, `score`, `bbox`

---

## 모델 산출물

| 파일 | 설명 |
|---|---|
| `runs/.../best.pt` | YOLO11n PyTorch 체크포인트 |
| `runs/.../best.onnx` | ONNX export (검증/벤치 용도) |
| `runs/.../best.engine` | TensorRT engine — **Jetson에서 직접 생성** |

단일 클래스 (`mannequin`), 입력 640×640, 출력 텐서 `[1, 5, 8400]` (cx, cy, w, h, score).

> `.engine` 파일은 GPU 아키텍처에 종속됩니다. 반드시 Jetson 위에서 export해야 합니다.
> 자세한 절차는 [`docs/jetson_deploy.md`](docs/jetson_deploy.md) 참조.

---

## 검증 결과 (OpenCV CPU 기준)

- 검증 데이터: `datasets/merged/val` 26장
- IoU 0.50 기준: Precision `0.9615` / Recall `0.9615` / F1 `0.9615`
- 평균 레이턴시: preprocess `2.7ms` / inference `59ms` / postprocess `2.3ms` (CPU 기준)
- **Jetson TRT FP16 기준 inference 목표: 5~10ms**

---

## 빌드 및 실행 (개발 PC)

> 개발 PC에는 TensorRT가 없으므로 빌드가 되지 않습니다.  
> 전체 빌드 및 실행은 Jetson에서 수행하세요 → [`docs/jetson_deploy.md`](docs/jetson_deploy.md)

---

## 빌드 및 실행 (Jetson)

```bash
# 1. engine 생성 (최초 1회, 5~15분 소요)
yolo export model=best.pt format=engine device=0 imgsz=640 half=True

# 2. 빌드
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vtol_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# 3. 실행
ros2 launch vtol_vision vision.launch.py \
    trt_engine_path:=/absolute/path/to/best.engine
```

CSI 카메라 사용 시:
```bash
ros2 launch vtol_vision vision.launch.py \
    trt_engine_path:=/absolute/path/to/best.engine \
    "camera_uri:=nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"
```

---

## 주요 파라미터

`config/vision_params.yaml`에서 설정합니다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `camera_uri` | `"0"` | 카메라 인덱스 또는 GStreamer 파이프라인 |
| `trt_engine_path` | `""` | `.engine` 파일 절대경로 |
| `conf_thr` | `0.25` | YOLO 신뢰도 임계값 |
| `nms_thr` | `0.45` | NMS IoU 임계값 |
| `yolo_input_size` | `640` | 모델 입력 해상도 |
| `yolo_period_ms` | `50` | 추론 루프 주기 (ms) |
| `marker_size_m` | `0.15` | ArUco 마커 물리 크기 (m) — 포즈 추정 정확도에 직접 영향 |
| `aruco_allowed_ids` | `[]` | 허용 ArUco ID 목록 (비어 있으면 전체 허용) |
| `aruco_min_perimeter_rate` | `0.02` | 탐지 허용 최소 마커 둘레 비율. 낮을수록 원거리 소형 마커 허용 (오탐 주의) |
| `enable_debug_image` | `false` | 디버그 이미지 토픽 발행 여부 |

---

## 프로젝트 구조

```
vtol_vision/
├── include/vtol_vision/
│   ├── vision_node.hpp       # ROS2 노드
│   └── yolo_detector.hpp     # TensorRT detector 인터페이스
├── src/
│   ├── main.cpp
│   ├── vision_node.cpp       # ArUco, 카메라, 퍼블리시
│   └── yolo_detector.cpp     # TRT 엔진 로드/추론/후처리
├── msg/                      # 커스텀 ROS2 메시지
├── config/
│   ├── vision_params.yaml    # 실행 파라미터
│   └── class_map.example.yaml
├── launch/vision.launch.py
├── docs/jetson_deploy.md     # Jetson 배포 전체 절차
└── tools/                    # 학습/평가/벤치 스크립트
```

---

## 의존성

| 패키지 | 비고 |
|---|---|
| ROS2 Humble | |
| OpenCV 4.x | aruco, dnn, calib3d |
| TensorRT 10.x | JetPack 6에 포함 (`libnvinfer-dev`) |
| CUDA 12.x | JetPack 6에 포함 |
| cv_bridge | `ros-humble-cv-bridge` |
