# Jetson Orin Nano Super 배포 가이드

대상 환경: **Jetson Orin Nano Super / JetPack 6.x / TensorRT 10.x / ROS2 Humble**

---

## 0. 전제 조건 확인 (Jetson에서 실행)

```bash
# JetPack 버전 확인
cat /etc/nv_tegra_release

# TensorRT 확인 (JetPack 6에 기본 포함)
dpkg -l | grep libnvinfer-dev
python3 -c "import tensorrt; print(tensorrt.__version__)"

# CUDA 확인
nvcc --version

# Python / Ultralytics 확인
python3 -c "from ultralytics import YOLO; print('OK')"
```

**필요 버전:**
| 패키지 | 최소 버전 |
|---|---|
| JetPack | 6.0 |
| TensorRT | 10.x |
| CUDA | 12.x |
| ROS2 | Humble |
| Ultralytics | 8.x |

---

## 1. TensorRT .engine 파일 생성

> **중요:** `.engine` 파일은 반드시 **이 Jetson 위에서** 생성해야 합니다.  
> 다른 GPU(개발 PC 등)에서 만든 engine은 GPU 아키텍처가 달라 동작하지 않습니다.

사전 학습 모델 Release:
- <https://github.com/Paichai-SkyEdge/vtol-vision/releases/tag/mannequin-model-v1>
- 포함 자산: `best.pt`, `best.onnx`

```bash
# best.pt 다운로드
mkdir -p ~/vtol-vision/weights
cd ~/vtol-vision/weights
wget https://github.com/Paichai-SkyEdge/vtol-vision/releases/download/mannequin-model-v1/best.pt

# Jetson에서 engine 생성
yolo export \
    model=best.pt \
    format=engine \
    device=0 \
    imgsz=640 \
    half=True        # FP16 — Orin Ampere GPU 성능 최대화
```

생성 완료 후 `best.engine` 파일이 같은 디렉토리에 생성됩니다.  
생성 시간: 약 5~15분 (첫 실행 시 최적화 포함)

---

## 2. ROS2 Humble 설치 (미설치 시)

```bash
# ROS2 Humble (Ubuntu 22.04 기준)
sudo apt install software-properties-common
sudo add-apt-repository universe
sudo apt update && sudo apt install curl -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
    http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update
sudo apt install ros-humble-desktop ros-dev-tools
```

---

## 3. 의존 패키지 설치

```bash
sudo apt install -y \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    libopencv-dev \
    libnvinfer-dev \
    libnvinfer-plugin-dev
```

> TensorRT 헤더(`libnvinfer-dev`)는 JetPack 6에서  
> `/usr/include/aarch64-linux-gnu/NvInfer.h` 에 설치됩니다.  
> CMakeLists.txt의 `find_path(TENSORRT_INCLUDE_DIR ...)`가 이 경로를 자동으로 감지합니다.

---

## 4. 패키지 빌드

```bash
# ROS2 workspace 설정
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 소스 복사 또는 클론
git clone <repo-url> vtol_vision
# 또는: scp -r user@dev-pc:~/vtol-vision/ ~/ros2_ws/src/vtol_vision/

# 빌드
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select vtol_vision --cmake-args -DCMAKE_BUILD_TYPE=Release
```

빌드 성공 시 출력:
```
TensorRT: /usr/lib/aarch64-linux-gnu/libnvinfer.so  include: /usr/include/aarch64-linux-gnu
```

---

## 5. 파라미터 설정

`config/vision_params.yaml`의 Jetson 환경에 맞게 수정:

```yaml
vision_node:
  ros__parameters:
    # --- 카메라 ---
    # USB 웹캠
    camera_uri: "0"
    # CSI 카메라 (IMX219 등) 사용 시 아래 주석 해제
    # camera_uri: "nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"

    frame_width: 640
    frame_height: 480
    camera_fps: 30

    # --- 모델 경로 (Jetson에서 생성한 engine) ---
    trt_engine_path: "/home/<user>/vtol-vision/weights/best.engine"

    # --- 검출 임계값 (비행 전 지상에서 튜닝 권장) ---
    conf_thr: 0.25
    nms_thr: 0.45

    # --- 추론 주기 (ms) ---
    # Orin Nano Super: 20ms(50fps) 목표, 실측 후 조정
    yolo_period_ms: 33

    enable_debug_image: false      # 비행 시 false 권장 (대역폭 절약)
    enable_yolo_debug_log: true
```

---

## 6. 실행

```bash
source ~/ros2_ws/install/setup.bash

# 기본 실행 (params.yaml 경로는 패키지 share에서 로드)
ros2 launch vtol_vision vision.launch.py \
    trt_engine_path:=/home/<user>/vtol-vision/weights/best.engine

# CSI 카메라 사용 시
ros2 launch vtol_vision vision.launch.py \
    trt_engine_path:=/home/<user>/vtol-vision/weights/best.engine \
    "camera_uri:=nvarguscamerasrc ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! appsink"
```

---

## 7. 동작 확인

```bash
# 검출 결과 토픽
ros2 topic echo /vision/objects
ros2 topic echo /vision/aruco

# 추론 레이턴시 확인
ros2 topic hz /vision/objects

# 디버그 이미지 확인 (enable_debug_image: true 설정 후)
ros2 run rqt_image_view rqt_image_view /vision/debug_image
```

정상 동작 로그:
```
[vision_node]: TensorRT engine loaded: .../best.engine  output=[1,5,8400]  conf=0.25  nms=0.45
[vision_node]: YOLO ready: model=...
[vision_node]: vision_node started.
```

---

## 8. 성능 기대치 (Jetson Orin Nano Super)

| 설정 | 추론 시간 (단독) |
|---|---|
| FP16, input=640, batch=1 | ~5~10ms |
| 전체 파이프라인 (캡처+추론+publish) | ~15~30ms |

- `yolo_period_ms: 33` 으로 설정하면 약 30fps 추론
- 비행 중 CPU/GPU 부하는 `jtop`으로 모니터링:  
  ```bash
  sudo apt install python3-jtop && sudo jtop
  ```

---

## 9. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `engine file does not exist` | 경로 오타 | `trt_engine_path` 절대경로 확인 |
| `failed to deserialize engine` | 다른 GPU에서 만든 engine | Jetson에서 재생성 |
| `unexpected engine output shape` | YOLO export 설정 문제 | `yolo export ... imgsz=640 half=True` 재실행 |
| `cudaMalloc failed` | GPU 메모리 부족 | 다른 GPU 프로세스 종료 후 재시도 |
| 카메라 열기 실패 | device 번호 | `ls /dev/video*` 로 확인 |
| CSI 카메라 안 열림 | GStreamer 파이프라인 | `gst-launch-1.0 nvarguscamerasrc ! fakesink` 로 먼저 테스트 |
