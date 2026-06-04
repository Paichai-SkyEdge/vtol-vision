# TODO

작성일: 2026-06-04

이 문서는 현재 `vtol_vision` 개발 상태와 다음 작업을 한눈에 이어받기 위한 작업 목록입니다.

## 현재 개발 상태

- ROS2 Humble 기반 `vtol_vision` C++ 노드 구조가 잡혀 있음.
- 카메라 입력, ArUco 탐지, YOLO TensorRT 추론 루프가 분리되어 동작하도록 구현되어 있음.
- 발행 토픽:
  - `/vision/aruco`: ArUco 탐지 결과
  - `/vision/objects`: YOLO 객체 탐지 결과
  - `/vision/debug_image`: 디버그 이미지, 옵션
- 메시지 타입은 `msg/ArucoDetection.msg`, `msg/ObjectDetection.msg`, `msg/VisionDetections.msg`로 정의되어 있음.
- Jetson 배포 문서, 신규 개발자 문서, 타 부서 연동 문서가 한국어로 정리되어 있음.
- 인수인계용 모델 파일이 `weights/` 아래에 포함되어 있음.
  - `weights/mannequin_yolo11n/best.pt`
  - `weights/mannequin_yolo11n/best.onnx`
  - `weights/basket_mannequin_yolo11n/best.pt`
  - `weights/basket_mannequin_yolo11n/best.onnx`
- TensorRT `.engine`은 Jetson에서 직접 생성해야 하며 커밋하지 않는 정책으로 정리되어 있음.
- RealSense D435i 실시간 탐지 스크립트가 `tools/realsense_live_detect.py`에 있음.

## 최우선 작업

### 1. 버티포트 인식 기능 추가

목표: 비전 시스템이 버티포트를 YOLO 객체로 인식할 수 있게 만든다.

- [ ] 버티포트 class 이름 확정
  - 권장 class name: `vertiport`
  - 기존 class와 함께 쓸 경우 class 순서 예시: `basket`, `mannequin`, `vertiport`
- [ ] 버티포트 사진 촬영
  - 실제 운용 카메라와 최대한 같은 시야각/해상도/설치 높이에서 촬영
  - 정면, 사선, 원거리, 근거리, 밝은 환경, 어두운 환경, 그림자, 부분 가림 상황 포함
  - 착륙 접근 상황처럼 화면 가장자리/작게 보이는 케이스 포함
- [ ] 버티포트 bounding box 라벨링
  - YOLO 형식으로 박스 라벨 작성
  - 박스는 버티포트 전체 외곽을 기준으로 일관되게 그림
  - 애매한 샘플은 별도 폴더에 분리하고 라벨 기준을 먼저 확정
- [ ] 버티포트 데이터셋 생성
  - `datasets/` 아래에 신규 데이터셋 구성
  - train/val 분리
  - `classes.txt`, `dataset.yaml` 작성
- [ ] YOLO 학습 스크립트가 3-class 데이터셋을 처리하도록 정리
  - 기존 basket/mannequin 학습 스크립트 재사용 가능 여부 확인
  - class map YAML도 3-class 기준으로 추가
- [ ] 버티포트 포함 모델 학습
  - smoke 학습으로 데이터/라벨 파이프라인 먼저 확인
  - full 학습 후 `best.pt`, `best.onnx` 생성
- [ ] Jetson에서 TensorRT `.engine` 생성
  - `yolo export model=<best.pt> format=engine device=0 imgsz=640 half=True`
- [ ] ROS 노드에서 새 class map 적용
  - `/vision/objects`에 `class_name: vertiport`가 정상 발행되는지 확인
- [ ] 실제 카메라로 실시간 테스트
  - 원거리/근거리 탐지율 확인
  - 오탐 케이스 기록
  - `conf_thr`, `nms_thr`, `yolo_period_ms` 튜닝

완료 기준:

- [ ] `/vision/objects`에서 버티포트가 `vertiport` class로 발행됨.
- [ ] 실사용 거리/고도에서 버티포트 탐지 성공률이 팀 기준을 만족함.
- [ ] 새 모델 파일과 class map이 `weights/`, `config/` 기준으로 정리됨.
- [ ] 타 부서 연동 문서에 `vertiport` class 추가 사항이 반영됨.

### 2. 마네킹/바구니 데이터 재촬영 및 정확도 개선

목표: 기존 mannequin, basket 모델의 정확도를 높이고 오탐/미탐을 줄인다.

- [ ] 현재 모델의 실패 케이스 수집
  - 마네킹 미탐
  - 바구니 미탐
  - 배경/물체 오탐
  - 원거리에서 작은 객체 누락
  - 조명/그림자/흔들림 상황
- [ ] 마네킹 사진 재촬영
  - 다양한 거리, 각도, 배경, 조명 조건 포함
  - 실제 운용 카메라 장착 위치와 비슷하게 촬영
  - 부분 가림, 화면 가장자리, 작은 크기 케이스 포함
- [ ] 바구니 사진 재촬영
  - 색상/형태/각도 차이가 있는 바구니 포함
  - 지면/배경과 색이 비슷한 어려운 케이스 포함
  - 원거리 소형 객체 케이스 포함
- [ ] 재촬영 이미지 라벨링
  - YOLO bbox 기준을 문서화
  - 라벨 누락/중복/잘린 박스 검수
- [ ] 기존 데이터셋과 병합
  - 중복 이미지 제거
  - train/val 분포 확인
  - class imbalance 확인
- [ ] 2-class 또는 3-class 통합 모델 재학습
  - 버티포트 작업과 합쳐서 최종 class 구성을 결정
  - 기존 모델 대비 Precision/Recall/F1 비교
- [ ] RealSense 실시간 탐지 스크립트로 현장 검증
  - `tools/realsense_live_detect.py`
  - basket confidence와 mannequin confidence 별도 튜닝 필요 여부 확인

완료 기준:

- [ ] mannequin, basket 검증 지표가 기존 모델보다 개선됨.
- [ ] 실제 카메라 실시간 화면에서 주요 미탐/오탐 케이스가 줄어듦.
- [ ] 새 모델이 `weights/`에 정리되고 README/연동 문서가 업데이트됨.

## 다음 개발 작업

### 모델/데이터 파이프라인

- [ ] 최종 class 구성을 확정한다.
  - 후보 A: `basket`, `mannequin`, `vertiport`
  - 후보 B: ROS용 `mannequin`, 별도 RealSense용 `basket/mannequin/vertiport`
- [ ] 신규 데이터셋 생성 스크립트를 3-class 기준으로 정리한다.
- [ ] 라벨링 기준 문서를 추가한다.
  - bbox를 어디까지 잡는지
  - 잘린 객체 처리
  - 흐릿한 객체 처리
  - 너무 작은 객체 처리
- [ ] 학습 결과 비교 표를 남긴다.
  - dataset version
  - model file
  - Precision
  - Recall
  - F1
  - mAP50
  - Jetson latency
- [ ] 최종 모델 파일 naming rule을 정한다.
  - 예: `weights/field_objects_yolo11n_v1/best.pt`

### ROS2 패키지

- [ ] `config/class_map.example.yaml`을 최종 class 구성에 맞게 갱신한다.
- [ ] `config/vision_params.yaml` 기본값이 clone 직후 안전한 값인지 확인한다.
- [ ] `/vision/objects` 소비 부서에서 필요한 추가 필드가 있는지 확인한다.
  - 예: class별 confidence threshold
  - 예: bbox 중심점
  - 예: normalized bbox
- [ ] 메시지 변경이 필요하면 `msg/*.msg`와 연동 문서를 함께 업데이트한다.
- [ ] Jetson에서 `colcon build`와 launch 실행을 다시 검증한다.

### Jetson/현장 테스트

- [ ] Jetson에서 `.engine` 생성 절차를 실제로 재검증한다.
- [ ] USB 카메라, CSI 카메라, RealSense 각각 실행 경로를 확인한다.
- [ ] `ros2 topic hz /vision/objects`로 실제 발행 주기를 기록한다.
- [ ] `pipeline_latency_ms` 기준 지연 시간을 기록한다.
- [ ] 비행/현장 조건에서 debug image를 끄고 안정성을 확인한다.

### 문서/인수인계

- [ ] `README.md` 모델 설명을 최종 모델 기준으로 갱신한다.
- [ ] `docs/integration_guide.md`에 `vertiport` class와 최종 class map을 반영한다.
- [ ] `docs/development_guide.md`에 신규 학습 절차를 반영한다.
- [ ] `tools/README.md`에 버티포트 데이터 준비/학습 스크립트를 추가한다.
- [ ] 타 부서 전달 체크리스트에 최종 모델 경로, engine 생성 경로, class map 경로를 명시한다.

## 데이터 수집 체크리스트

촬영할 때 최소한 아래 조건을 채운다.

- [ ] 각 class별 충분한 원본 이미지 확보
  - `basket`
  - `mannequin`
  - `vertiport`
- [ ] 거리별 샘플
  - 근거리
  - 중거리
  - 원거리
- [ ] 각도별 샘플
  - 정면
  - 좌/우 사선
  - 상단에서 내려다보는 시점
  - 화면 가장자리
- [ ] 환경별 샘플
  - 실내
  - 실외
  - 햇빛
  - 그림자
  - 흐린 날
  - 저조도
- [ ] 실패 케이스 별도 저장
  - 오탐
  - 미탐
  - 흔들림
  - blur
  - 부분 가림

## 우선순위

1. 버티포트 사진 촬영 및 bbox 라벨링
2. 마네킹/바구니 재촬영 및 라벨 검수
3. 3-class 데이터셋 구성
4. smoke 학습으로 파이프라인 확인
5. full 학습 및 지표 비교
6. Jetson TensorRT export
7. ROS2 실시간 검증
8. 타 부서 연동 문서 업데이트
