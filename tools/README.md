# Tools

이 디렉터리는 데이터셋 준비, 학습, 평가, Jetson 배포 보조 스크립트를 모아둔 작업 공간입니다. 별도 설명이 없으면 레포 루트에서 실행하는 것을 기준으로 합니다.

## Dataset Preparation

| 스크립트 | 용도 |
|---|---|
| `prepare_basket_mannequin_dataset.py` | basket/mannequin 학습 데이터셋 기본 구조 생성 |
| `prepare_basket_mannequin_final_dataset.py` | 최종 학습용 train/val 데이터셋 생성 |
| `prepare_basket_mannequin_relabel_dataset.sh` | 재라벨 후보 데이터셋 준비 |
| `merge_mannequin_dataset.py` | mannequin 데이터셋 병합 |
| `create_labeled_flat_dataset.sh` | 라벨이 있는 샘플만 flat 디렉터리로 정리 |
| `augment_basket_mannequin_images.py` | basket/mannequin 이미지 증강 |
| `augment_basket_mannequin_images.sh` | 증강 스크립트 실행 래퍼 |
| `generate_relabel_candidates.py` | 재라벨 후보 생성 |
| `generate_relabel_candidates.sh` | 재라벨 후보 생성 래퍼 |
| `prepare_shadow_augmented_dataset.py` | YOLO train에만 객체 경계 교차 그림자를 추가하고 validation/test는 고정한 비교 실험 데이터셋 생성 |
| `prepare_shadow_stress_test.py` | 학습과 다른 seed로 test split에 그림자를 적용해 강건성을 비교하는 평가셋 생성 |
| `prepare_mild_photometric_dataset.py` | 원본을 보존하고 train 일부에만 제한된 명도·채도 변형을 추가하며 별도 평가셋 생성 |

## Training And Export

| 스크립트 | 용도 |
|---|---|
| `train_basket_mannequin_yolo.py` | 2-class basket/mannequin YOLO 학습 |
| `train_basket_mannequin_yolo.sh` | 학습 실행 래퍼 |
| `train_mannequin_yolo.py` | mannequin 단일 클래스 YOLO 학습 |
| `create_basket_mannequin_training_bundle.sh` | 외부 GPU/Jetson 학습용 self-contained 번들 생성 |
| `export_tensorrt_engine.sh` | YOLO 모델을 TensorRT engine으로 export |
| `monitor_training.sh` | 학습 로그/상태 모니터링 |

### Shadow Robustness Experiment

기존 validation/test를 변경하지 않고 train 이미지 40%에 객체 인식형 그림자 변형을 한 장씩 추가합니다. 기본 모드는 원본 파일을 hard link하여 디스크 중복을 줄입니다.

```bash
python3 tools/prepare_shadow_augmented_dataset.py --overwrite
yolo detect train \
  model=figj/vtol_vision_cpp/weights/basket_mannequin_yolo11n_best.pt \
  data=datasets/skyedge_all_yolo_shadow/data.yaml \
  epochs=5 imgsz=640 batch=4 device=0 \
  optimizer=AdamW lr0=0.0002 lrf=0.1 warmup_epochs=0 \
  project=runs/skyedge name=yolo11n_shadow_v1
```

비교 시 baseline과 shadow 모델 모두 동일한 `datasets/skyedge_all_yolo/data.yaml`의 test split으로 최종 평가합니다. 그림자 강건성 비교셋은 다음처럼 별도로 생성합니다.

```bash
python3 tools/prepare_shadow_stress_test.py --overwrite
```

## Evaluation And Benchmark

| 스크립트 | 용도 |
|---|---|
| `eval_opencv_yolo_detector.py` | OpenCV DNN 기반 YOLO 평가 |
| `benchmark_onnx_opencv.py` | ONNX + OpenCV 추론 벤치마크 |
| `benchmark_tensorrt_engine.sh` | TensorRT engine 벤치마크 |
| `jetson_latency_bench.py` | Jetson 환경 레이턴시 측정 |
| `generate_paper_figures.py` | 보고서/논문용 figure 생성 |

## Device And Labeling Helpers

| 스크립트 | 용도 |
|---|---|
| `realsense_live_detect.py` | Intel RealSense D435i 실시간 basket/mannequin 탐지 |

YOLO11s RealSense GUI 디버깅:

```bash
python3 tools/realsense_live_detect.py \
  --model runs/detect/runs/skyedge/yolo11s_realsense_v1/weights/best.pt \
  --device 0
```

YOLO11n과 YOLO11s를 동일 카메라에서 비교하려면 `--compare-model`을 지정하고 GUI에서 `m` 키로 전환합니다.
| `labelimg_basket_mannequin.sh` | LabelImg 실행 보조 |
| `labelimg_basket_mannequin_relabel.sh` | 재라벨 작업용 LabelImg 실행 보조 |
| `jetson_connect.sh` | Jetson 접속 보조 |
| `jetson_provision.sh` | Jetson 초기 세팅 보조 |

## Output Policy

스크립트가 생성하는 데이터와 학습 결과는 주로 `datasets/`, `images/`, `runs/`, `training_bundle/` 아래에 생깁니다. 새 대용량 이미지, 라벨, weight, engine, archive 파일은 `.gitignore`에 의해 기본적으로 커밋되지 않습니다.
