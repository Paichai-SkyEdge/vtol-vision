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

## Training And Export

| 스크립트 | 용도 |
|---|---|
| `train_basket_mannequin_yolo.py` | 2-class basket/mannequin YOLO 학습 |
| `train_basket_mannequin_yolo.sh` | 학습 실행 래퍼 |
| `train_mannequin_yolo.py` | mannequin 단일 클래스 YOLO 학습 |
| `create_basket_mannequin_training_bundle.sh` | 외부 GPU/Jetson 학습용 self-contained 번들 생성 |
| `export_tensorrt_engine.sh` | YOLO 모델을 TensorRT engine으로 export |
| `monitor_training.sh` | 학습 로그/상태 모니터링 |

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
| `labelimg_basket_mannequin.sh` | LabelImg 실행 보조 |
| `labelimg_basket_mannequin_relabel.sh` | 재라벨 작업용 LabelImg 실행 보조 |
| `jetson_connect.sh` | Jetson 접속 보조 |
| `jetson_provision.sh` | Jetson 초기 세팅 보조 |

## Output Policy

스크립트가 생성하는 데이터와 학습 결과는 주로 `datasets/`, `images/`, `runs/`, `training_bundle/` 아래에 생깁니다. 새 대용량 이미지, 라벨, weight, engine, archive 파일은 `.gitignore`에 의해 기본적으로 커밋되지 않습니다.
