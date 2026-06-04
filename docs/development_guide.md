# 개발 인수인계 가이드

이 문서는 `vtol_vision`을 처음 받은 개발자가 바로 개발 환경을 잡고, 코드를 수정하고, Jetson에서 검증할 수 있도록 정리한 한국어 가이드입니다.

## 대상

| 대상 | 이 문서를 보면 되는 경우 |
|---|---|
| 신규 개발자 | 레포를 처음 clone 받아 빌드/실행/수정 흐름을 잡아야 할 때 |
| 모델/데이터 담당자 | YOLO 학습 데이터, 학습 결과, 모델 export 흐름을 확인해야 할 때 |
| Jetson 담당자 | Jetson Orin Nano Super에서 카메라/엔진/ROS2 노드를 올려야 할 때 |

타 부서에서 이 패키지를 구독하거나 시스템에 연결하는 방법은 [`integration_guide.md`](integration_guide.md)를 보세요.

## 한눈에 보는 구조

```txt
vtol-vision/
├── src/, include/             # ROS2 C++ 비전 노드
├── msg/                       # 타 패키지가 구독할 메시지 정의
├── launch/                    # vision_node 실행 진입점
├── config/                    # 런타임 파라미터와 class map
├── tools/                     # 데이터 준비, 학습, 평가, 장비 보조 스크립트
├── docs/                      # 개발/배포/연동 문서
├── datasets/, images/         # 데이터 작업 공간
├── training_bundle/           # 외부 GPU/Jetson 학습 번들
├── runs/                      # 로컬 학습/평가 산출물, 기본 ignore
└── weights/                   # clone 직후 사용할 인수인계 모델
```

상세 구조와 산출물 관리 기준은 [`repository_layout.md`](repository_layout.md)에 있습니다.

## 개발 환경 준비

### 1. 레포 받기

ROS2 workspace 안에 패키지 이름을 `vtol_vision`으로 두는 것을 권장합니다.

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone <repo-url> vtol_vision
cd ~/ros2_ws
```

### 2. Jetson 기준 의존성 설치

대상 실행 환경은 Jetson Orin Nano Super, JetPack 6.x, ROS2 Humble, TensorRT 10.x입니다.

```bash
sudo apt update
sudo apt install -y \
  ros-humble-cv-bridge \
  ros-humble-vision-msgs \
  libopencv-dev \
  libnvinfer-dev \
  libnvinfer-plugin-dev
```

더 자세한 Jetson 세팅은 [`jetson_deploy.md`](jetson_deploy.md)를 기준으로 진행합니다.

### 3. 개발 PC에서의 주의점

이 패키지는 TensorRT/CUDA 헤더와 라이브러리를 링크합니다. 일반 개발 PC에 TensorRT가 없으면 전체 `colcon build`가 실패할 수 있습니다.

개발 PC에서는 다음 작업을 우선 수행하세요.

```bash
# 문서/스크립트 수정 확인
git diff --check

# Python 도구 문법 확인이 필요할 때
python3 -m py_compile tools/realsense_live_detect.py
```

C++ 빌드와 실제 카메라/YOLO 검증은 Jetson에서 수행하는 것을 기본 원칙으로 둡니다.

## 빌드

Jetson에서 실행합니다.

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vtol_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

성공 시 `vision_node` 실행 파일과 메시지 타입이 workspace에 설치됩니다.

## 모델 준비

배포용 `.pt`/`.onnx` 모델은 `weights/` 아래에 커밋되어 있습니다. `.engine`만 Jetson에서 직접 생성해야 합니다. 다른 GPU에서 만든 `.engine`은 Jetson에서 동작하지 않을 수 있습니다.

```bash
cd ~/ros2_ws/src/vtol_vision
yolo export model=weights/mannequin_yolo11n/best.pt format=engine device=0 imgsz=640 half=True
```

운영 시에는 launch argument로 engine 절대경로를 넘깁니다.

```bash
ros2 launch vtol_vision vision.launch.py \
  trt_engine_path:=/home/<user>/ros2_ws/src/vtol_vision/weights/mannequin_yolo11n/best.engine
```

## 실행

USB 카메라:

```bash
ros2 launch vtol_vision vision.launch.py \
  trt_engine_path:=/home/<user>/ros2_ws/src/vtol_vision/weights/mannequin_yolo11n/best.engine \
  camera_uri:=0
```

CSI 카메라:

```bash
ros2 launch vtol_vision vision.launch.py \
  trt_engine_path:=/home/<user>/ros2_ws/src/vtol_vision/weights/mannequin_yolo11n/best.engine \
  "camera_uri:=nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"
```

동작 확인:

```bash
ros2 topic echo /vision/objects
ros2 topic echo /vision/aruco
ros2 topic hz /vision/objects
```

## 주요 개발 포인트

| 변경하고 싶은 것 | 주로 볼 파일 |
|---|---|
| 토픽 발행, 카메라 입력, ArUco 탐지 | `src/vision_node.cpp`, `include/vtol_vision/vision_node.hpp` |
| TensorRT YOLO 전처리/후처리 | `src/yolo_detector.cpp`, `include/vtol_vision/yolo_detector.hpp` |
| 메시지 필드 추가/변경 | `msg/*.msg`, `CMakeLists.txt` |
| 실행 파라미터 | `config/vision_params.yaml`, `launch/vision.launch.py` |
| 모델 class 이름 | `config/class_map.example.yaml` 또는 별도 class map YAML |
| 학습/평가 스크립트 | `tools/README.md` |

메시지 필드를 바꾸면 타 부서 연동 코드가 깨질 수 있습니다. 변경 전에 [`integration_guide.md`](integration_guide.md)의 인터페이스 계약을 같이 업데이트하세요.

## 데이터와 학습

기본 학습 흐름:

```bash
python3 tools/prepare_basket_mannequin_final_dataset.py
python3 tools/train_basket_mannequin_yolo.py --mode smoke --device cpu
python3 tools/train_basket_mannequin_yolo.py --mode full --device 0
```

외부 GPU/Jetson으로 넘길 학습 번들 생성:

```bash
tools/create_basket_mannequin_training_bundle.sh
```

학습 결과는 `runs/` 아래에 생성됩니다. 새 `runs/`, 새 `.pt`, 새 `.onnx`, `.engine`은 기본적으로 커밋하지 않습니다. 단, 타 부서 인수인계용 모델은 `weights/` 아래의 명시된 파일만 커밋합니다.

## 커밋 전 체크리스트

```bash
git status --short
git diff --check
```

코드 변경 시 Jetson에서 가능한 범위까지 확인합니다.

```bash
colcon build --packages-select vtol_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
ros2 launch vtol_vision vision.launch.py \
  trt_engine_path:=/absolute/path/to/weights/mannequin_yolo11n/best.engine
```

문서만 변경했다면 `git diff --check` 통과 여부를 최소 기준으로 둡니다.

## 자주 생기는 문제

| 증상 | 확인할 것 |
|---|---|
| `Failed to open camera_uri` | 카메라 번호, GStreamer pipeline, 권한, 다른 프로세스 점유 여부 |
| `YOLO is disabled` | `trt_engine_path`가 비어 있거나 파일이 없거나 engine 로드 실패 |
| `failed to deserialize engine` | `.engine`을 Jetson이 아닌 다른 GPU에서 만든 경우 |
| ArUco pose가 0 또는 confidence 0 | `camera_info_yaml`이 비어 있거나 보정 파일 파싱 실패 |
| `/vision/debug_image`가 안 보임 | `enable_debug_image: true` 설정 필요 |
