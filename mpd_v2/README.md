# Realsense Live Detect — C++ Standalone

RealSense D435i / 시뮬레이션 실시간 YOLO 객체 탐지 (C++, ONNX-DNN / TensorRT, OpenCV GUI)

Python 버전 `tools/realsense_live_detect.py`를 C++로 포팅한 self-contained 모듈입니다.  
다른 팀원의 프로젝트에 바로 복사해서 사용 가능. **Jetson + RealSense** 모드와 **PC/Gazebo 시뮬레이션** 모드를 모두 지원합니다.

---

## 요구사항

| 모드 | 필요 패키지 |
|---|---|
| **Sim (PC/Gazebo)** | OpenCV 4.x (`libopencv-dev`), CMake 3.16+ |
| **Jetson + RealSense** | 위 + CUDA 12.x, TensorRT 10.x, `librealsense2-dev` |

---

## 빠른 시작

### 시뮬레이션 모드 (PC/Gazebo — 팀원들이 가장 많이 쓸 모드)

```bash
# 1. ONNX 모델 export (최초 1회)
./scripts/export_onnx.sh

# 2. 빌드 & 실행 (OpenCV DNN 기반, CUDA 불필요)
./scripts/run_sim.sh

# 웹캠/비디오 파일 사용
./scripts/run_sim.sh --camera 0        # 웹캠 인덱스
./scripts/run_sim.sh --video test.mp4   # 비디오 파일
```

### Jetson + RealSense 모드

```bash
# 1. TensorRT 엔진 export
./scripts/export_engine.sh

# 2. 빌드 & 실행
./scripts/run.sh --tiled --motion
```

---

## 커맨드라인 옵션

| 옵션 | 기본값 | 모드 |
|---|---|---|
| `--model PATH` | `.onnx` 또는 `.engine` | 공통 |
| `--sim` | — | sim 모드 활성화 |
| `--camera N` | 0 | sim 모드 웹캠 인덱스 |
| `--video PATH` | — | sim 모드 비디오 입력 |
| `--conf F` | 0.25 | 공통 |
| `--basket-conf F` | 0.15 | 공통 |
| `--imgsz N` | 640 | 공통 |
| `--tiled` | off | 공통 |
| `--tile-cols N --tile-rows N` | 3×2 | 공통 |
| `--motion` | off | 공통 |
| `--width W --height H` | 848×480 | 공통 |

---

## 키 조작

| 키 | 기능 |
|---|---|
| `q` / `ESC` | 종료 |
| `d` | depth colormap (RealSense only) |
| `t` | tiled 추론 토글 |
| `h` | motion compensation 토글 |

---

## 구조

```
mpd_v2/
├── CMakeLists.txt                  # WITH_TENSORRT/WITH_REALSENSE 옵션
├── config/basket_mannequin_class_map.yaml
├── include/vtol_vision/
│   └── yolo_detector.hpp           # DNN(ONNX) + TRT dual backend
├── src/
│   ├── yolo_detector.cpp           # 자동 백엔드 선택 (.onnx→DNN, .engine→TRT)
│   └── realsense_live_detect.cpp   # RealSense / VideoCapture dual camera
├── scripts/
│   ├── export_onnx.sh              # .pt → .onnx (시뮬레이션용)
│   ├── export_engine.sh            # .pt → .engine (Jetson용)
│   ├── run_sim.sh                  # 시뮬레이션 빌드+실행
│   └── run.sh                      # Jetson 빌드+실행
└── weights/
    ├── yolo11n_shadow_v1_best.pt
    └── yolo11n_shadow_v1_best.onnx
```

## 다른 프로젝트에 통합 (Gazebo 시뮬레이션)

```bash
# 복사
cp -r mpd_v2/ /path/to/gazebo_ws/src/

# ONNX export + 빌드
cd /path/to/gazebo_ws/src/mpd_v2
./scripts/export_onnx.sh
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_REALSENSE=OFF -DWITH_TENSORRT=OFF
make -j$(nproc)

# 실행 (웹캠 또는 Gazebo 카메라 토픽을 virtual camera로)
./realsense_live_detect --sim --model ../weights/yolo11n_shadow_v1_best.onnx
```
