# Repository Layout

이 문서는 `vtol_vision` 레포에서 코드, 데이터, 학습 산출물을 어디에 두는지 정리합니다.

## Core ROS2 Package

| 경로 | 설명 |
|---|---|
| `CMakeLists.txt`, `package.xml` | ROS2 Humble 패키지 빌드 메타데이터 |
| `include/vtol_vision/` | C++ public headers |
| `src/` | ROS2 노드, 카메라/ArUco 파이프라인, TensorRT YOLO 구현 |
| `msg/` | `VisionDetections`, `ArucoDetection`, `ObjectDetection` 메시지 |
| `launch/` | 실행 launch 파일 |
| `config/` | 런타임 파라미터, 클래스 매핑, 데이터셋 설정 |
| `test/` | C++ 단위 테스트 |

## Documentation

| 경로 | 설명 |
|---|---|
| `README.md` | 프로젝트 개요, 빌드/실행, 운영 기준 |
| `docs/README.md` | 역할별 문서 안내 |
| `docs/development_guide.md` | 신규 개발자 개발 환경/빌드/검증 인수인계 |
| `docs/integration_guide.md` | 타 부서 ROS2 토픽/메시지/파라미터 연동 계약 |
| `docs/jetson_deploy.md` | Jetson Orin Nano Super 배포 절차 |
| `docs/repository_layout.md` | 레포 구조와 산출물 관리 기준 |
| `tools/README.md` | 학습/평가/장비 스크립트 목록 |

## Data And Artifacts

| 경로 | 설명 | 기본 정책 |
|---|---|---|
| `datasets/` | YOLO 데이터셋 스냅샷과 데이터셋 메타데이터 | 새 이미지/라벨은 ignore |
| `images/` | 원본, 증강, 재라벨 후보 이미지 작업 공간 | 새 이미지/XML 산출물은 ignore |
| `runs/` | Ultralytics 학습/평가 결과 | 전체 ignore |
| `weights/` | clone 직후 사용할 handoff 모델 보관 위치 | 명시된 `best.pt`/`best.onnx`만 추적, 새 모델은 ignore |
| `training_bundle/` | GPU 장비로 옮길 self-contained 학습 번들 | 현재 추적된 번들은 보존, 새 번들은 기본 ignore |
| `paper/` | 보고서/논문 자료 | LaTeX 빌드 산출물은 ignore |

현재 레포에는 학습 재현을 위해 이미 추적 중인 데이터 스냅샷이 남아 있습니다. `.gitignore`는 새로 생기는 대용량 산출물을 막는 기준이며, 이미 추적 중인 파일을 자동으로 제거하지는 않습니다.

대용량 데이터를 별도 저장소나 Release 자산으로 완전히 분리하려면, 별도 커밋에서 다음 범위를 검토하세요.

```bash
git rm --cached -r datasets images training_bundle
```

이 작업은 히스토리와 clone 크기에 영향이 크므로 코드 변경과 분리해서 진행하는 것이 좋습니다.

## Suggested Workflow

1. 데이터 전처리와 증강은 `tools/` 스크립트로 수행합니다.
2. 결과 데이터셋은 `datasets/` 아래에 생성하되, 새 대용량 파일은 기본적으로 커밋하지 않습니다.
3. 학습은 `tools/train_basket_mannequin_yolo.py` 또는 `training_bundle/.../scripts/train.sh`로 실행합니다.
4. 학습 결과는 `runs/` 아래에 남기고, 인수인계할 `best.pt`/`best.onnx`만 `weights/`의 명시된 모델 디렉터리로 복사해 커밋합니다.
5. TensorRT `.engine`은 Jetson에서 다시 export합니다. `.engine`은 GPU/JetPack 환경에 묶이므로 커밋하지 않습니다.
