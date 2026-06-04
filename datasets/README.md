# Datasets

YOLO 학습과 검증에 쓰는 데이터셋 작업 공간입니다.

현재 레포에는 basket/mannequin 실험을 재현하기 위한 스냅샷이 일부 추적되어 있습니다. 앞으로 새로 생성되는 이미지와 라벨은 기본적으로 `.gitignore`에 의해 제외되며, 공유가 필요한 데이터셋은 별도 아카이브나 Release 자산으로 관리하는 것을 권장합니다.

일반적인 YOLO 데이터셋 구조:

```txt
dataset_name/
├── dataset.yaml
├── classes.txt
├── train/
│   ├── images/
│   └── labels/
└── val/
    ├── images/
    └── labels/
```

주요 데이터셋:

| 경로 | 설명 |
|---|---|
| `basket_mannequin/` | basket/mannequin 학습용 중간 데이터셋 |
| `basket_mannequin_final/` | 최종 train/val 분할 데이터셋 |
| `basket_mannequin_labeled_flat/` | 라벨이 있는 샘플만 모은 flat 데이터셋 |
| `merged/` | 병합/평가용 데이터셋 |
