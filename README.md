# vtol_vision

ROS2 Humble 기반 C++ 비전 모듈. VTOL 기체의 Jetson Orin Nano Super에 탑재하여 ArUco 마커 추적과 YOLO 객체 탐지를 실시간으로 수행합니다.

---

## 처음 받는 사람을 위한 문서

| 상황 | 먼저 볼 문서 |
|---|---|
| 레포를 처음 받아 개발 환경을 잡는 경우 | [`docs/development_guide.md`](docs/development_guide.md) |
| Jetson에 배포하고 실행하는 경우 | [`docs/jetson_deploy.md`](docs/jetson_deploy.md) |
| 다른 부서에서 ROS2 토픽을 구독해 연결하는 경우 | [`docs/integration_guide.md`](docs/integration_guide.md) |
| 레포 구조와 데이터/산출물 기준을 확인하는 경우 | [`docs/repository_layout.md`](docs/repository_layout.md) |
| 학습/평가/장비 보조 스크립트를 찾는 경우 | [`tools/README.md`](tools/README.md) |

빠른 개발 시작(Jetson 기준):

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <repo-url> vtol_vision
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vtol_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
```

빠른 연동 확인:

```bash
ros2 topic info /vision/objects
ros2 interface show vtol_vision/msg/VisionDetections
```

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
| `/vision/aruco` | `vtol_vision/msg/VisionDetections` | ArUco 탐지 결과 |
| `/vision/objects` | `vtol_vision/msg/VisionDetections` | YOLO 객체 탐지 결과 |
| `/vision/debug_image` | `sensor_msgs/msg/Image` | 디버그 오버레이 (옵션) |

**메시지 구조**
- `VisionDetections` — 헤더 + `ArucoDetection[]` + `ObjectDetection[]` + `pipeline_latency_ms`
- `ArucoDetection` — `marker_id`, `pose` (6-DoF), `reprojection_error`, `confidence`
- `ObjectDetection` — `class_id`, `class_name`, `score`, `bbox`

---

## 모델 산출물

| 파일 | 설명 |
|---|---|
| `weights/basket_mannequin_yolo11n/best.pt` | basket/mannequin 2-class YOLO11n PyTorch 체크포인트 |
| `weights/basket_mannequin_yolo11n/best.onnx` | basket/mannequin 2-class ONNX export |
| `weights/mannequin_yolo11n/best.pt` | mannequin 단일 클래스 YOLO11n PyTorch 체크포인트 |
| `weights/mannequin_yolo11n/best.onnx` | mannequin 단일 클래스 ONNX export |
| `*.engine` | TensorRT engine — **Jetson에서 직접 생성, 커밋하지 않음** |

단일 클래스 (`mannequin`), 입력 640×640, 출력 텐서 `[1, 5, 8400]` (cx, cy, w, h, score).

> `.engine` 파일은 GPU 아키텍처에 종속됩니다. 반드시 Jetson 위에서 export해야 합니다.
> 자세한 절차는 [`docs/jetson_deploy.md`](docs/jetson_deploy.md) 참조.

사전 학습 모델은 `weights/` 아래에 같이 커밋되어 있습니다. `.engine` 파일만 Jetson에서 따로 생성하세요.

---

## 데이터와 산출물 관리

레포에는 현재 학습 재현을 위한 데이터 스냅샷과 번들이 함께 들어 있습니다. 새로 생성되는 대용량 데이터와 학습 산출물은 기본적으로 `.gitignore`에 막아두었습니다.

| 경로 | 용도 | 관리 기준 |
|---|---|---|
| `datasets/` | YOLO 학습/검증 데이터셋 | 메타데이터는 보존, 새 이미지/라벨은 필요할 때만 명시적으로 추가 |
| `images/` | 원본/증강/재라벨 후보 이미지 작업 공간 | 실험용 작업 공간, 새 이미지 산출물은 기본 ignore |
| `runs/` | Ultralytics 학습/평가 결과 | 로컬 산출물, 인수인계 모델은 `weights/`에서 관리 |
| `training_bundle/` | 외부 GPU/Jetson으로 복사 가능한 학습 번들 | 스크립트와 설정은 보존, 생성 아카이브와 산출물은 ignore |
| `paper/` | 보고서/논문 자료 | LaTeX 빌드 산출물은 ignore |

전체 구조와 운영 기준은 [`docs/repository_layout.md`](docs/repository_layout.md)를 참고하세요. 개발 인수인계는 [`docs/development_guide.md`](docs/development_guide.md), 타 부서 연동은 [`docs/integration_guide.md`](docs/integration_guide.md), 도구별 용도는 [`tools/README.md`](tools/README.md)에 정리되어 있습니다.

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
# 0. engine 생성 (최초 1회, 5~15분 소요)
cd ~/ros2_ws/src/vtol_vision
yolo export model=weights/mannequin_yolo11n/best.pt format=engine device=0 imgsz=640 half=True

# 1. 빌드
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vtol_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash

# 2. 실행
ros2 launch vtol_vision vision.launch.py \
    trt_engine_path:=/absolute/path/to/weights/mannequin_yolo11n/best.engine
```

CSI 카메라 사용 시:
```bash
ros2 launch vtol_vision vision.launch.py \
    trt_engine_path:=/absolute/path/to/weights/mannequin_yolo11n/best.engine \
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
├── CMakeLists.txt
├── package.xml
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
├── docs/
│   ├── README.md             # 역할별 문서 안내
│   ├── development_guide.md  # 신규 개발자 온보딩
│   ├── integration_guide.md  # 타 부서 ROS2 연동 계약
│   ├── jetson_deploy.md      # Jetson 배포 전체 절차
│   └── repository_layout.md  # 레포 구조와 산출물 관리 기준
├── tools/                    # 학습/평가/벤치/장비 스크립트
├── datasets/                 # 학습 데이터 스냅샷 및 메타데이터
├── images/                   # 원본/증강/재라벨 후보 작업 공간
├── training_bundle/          # 외부 학습용 self-contained 번들
├── runs/                     # 로컬 학습/평가 산출물 (ignore)
└── paper/                    # 보고서/논문 자료
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
