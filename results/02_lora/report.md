# 02_lora

| partition | scene | branch | auroc | ap | f1 | tpr | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | R01 | S | 0.650 | 0.489 | 0.210 | 0.133 | 0.078 |
| development | R02 | S | 0.810 | 0.671 | 0.043 | 0.023 | 0.008 |
| development | R03 | S | 0.730 | 0.769 | 0.471 | 0.311 | 0.008 |
| development | R04 | S | 0.731 | 0.595 | 0.000 | 0.000 | 0.005 |

![development_curves](figures/development_curves.png)

![development_performance](figures/development_performance.png)

![lora_vs_frozen](figures/lora_vs_frozen.png)

![normal_validation_loss](figures/normal_validation_loss.png)

![score_example](figures/score_example.png)

상세: [장면별 지표](metrics_by_scene.csv), [영상별 지표](metrics_by_video.csv), [실행 설정](config.json), [manifest](run_manifest.json), `logs/`, `scores/`.

### 채택 판단

**LoRA를 채택하지 않고 frozen DINOv2로 모듈 실험을 진행합니다.** 개발셋 macro AUROC는 frozen **73.13%**, LoRA seed 평균 **73.01%**입니다. 차이는 **-0.12%p**, 영상 단위 paired bootstrap 95% 구간은 **[-0.46, +0.16]%p**입니다.

사전에 정한 두 조건(평균 +1%p 이상, 95% 하한 > 0)을 모두 만족할 때만 LoRA를 채택합니다. 정상 validation 손실 감소 자체를 이상 탐지 향상으로 해석하지 않습니다. 장면별로 유리한 백본을 따로 고르지 않았으며 최종 평가 라벨은 채택 판단에 사용하지 않았습니다.

[채택 근거 JSON](backbone_decision.json) · [frozen 대조표](frozen_comparison.csv) · [학습 요약](training_summary.csv)

Seed별 macro AUROC는 **seed 42: 73.18%, seed 43: 72.95%, seed 44: 72.89%**입니다. [Seed별 성능표](performance_by_seed.csv)에 AP와 frozen 대비 차이도 보존합니다. Bootstrap 구간은 관측된 세 seed 평균을 대상으로 영상 표집의 불확실성을 나타냅니다.

### 학습 및 불확실성

12개 fit 모두 정상 training/validation 영상만 사용했습니다. 원래 DINOv2 가중치의 학습 전후 SHA256 일치와 adapter 체크포인트 SHA256을 각 training 로그에 남겼습니다. 신뢰구간은 장면·라벨 유형 내 영상을 재표집하고 같은 표집을 모든 seed와 두 백본에 적용한 10,000회 결과입니다. 프레임이나 seed를 독립 표본으로 세지 않습니다. 원본 녹화 그룹을 알 수 없고 일부 strata의 영상 수가 작아 이 구간을 광범위한 일반화의 보장으로 해석할 수 없습니다.

전체 LoRA 학습 기록 합계 237.9분, 관측 peak allocated VRAM 7.13GiB입니다. 평가 추출 비용은 별도의 영상 로그에 있습니다.
