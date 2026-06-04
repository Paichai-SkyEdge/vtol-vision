# Training Bundle

외부 GPU 머신이나 Jetson으로 복사해서 학습할 수 있는 self-contained 번들을 두는 위치입니다.

현재 `basket_mannequin_yolo/` 번들은 dataset, class map, 학습/export 스크립트를 함께 담고 있습니다. 새로 생성되는 대용량 이미지, weight, archive 파일은 기본적으로 커밋하지 않습니다.

번들 생성:

```bash
tools/create_basket_mannequin_training_bundle.sh
```

생성된 `.tar.gz` 아카이브와 학습 결과는 `.gitignore`에 의해 제외됩니다.
