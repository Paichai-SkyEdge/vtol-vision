# Models

이 디렉터리는 다른 개발자와 연동 부서가 clone 직후 바로 사용할 수 있도록 커밋해 둔 모델 파일을 담습니다.

| 경로 | 용도 |
|---|---|
| `mannequin_yolo11n/best.pt` | ROS `vision_node` TensorRT export 원본, mannequin 단일 클래스 |
| `mannequin_yolo11n/best.onnx` | mannequin 단일 클래스 ONNX 검증/벤치마크용 |
| `basket_mannequin_yolo11n/best.pt` | `tools/realsense_live_detect.py` 기본 모델, basket+mannequin 2-class |
| `basket_mannequin_yolo11n/best.onnx` | basket+mannequin 2-class ONNX 검증/벤치마크용 |

TensorRT `.engine`은 Jetson/GPU/JetPack 버전에 묶인 산출물이므로 커밋하지 않습니다. 대상 Jetson에서 아래처럼 생성합니다.

```bash
cd ~/ros2_ws/src/vtol_vision
yolo export model=weights/mannequin_yolo11n/best.pt format=engine device=0 imgsz=640 half=True
```
