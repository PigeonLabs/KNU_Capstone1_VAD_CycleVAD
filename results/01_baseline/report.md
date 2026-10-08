# 01_baseline

| partition | scene | branch | auroc | ap | f1 | tpr | fpr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| development | R01 | S | 0.657 | 0.491 | 0.183 | 0.111 | 0.059 |
| development | R02 | S | 0.808 | 0.665 | 0.044 | 0.023 | 0.008 |
| development | R03 | S | 0.730 | 0.769 | 0.466 | 0.307 | 0.009 |
| development | R04 | S | 0.731 | 0.594 | 0.000 | 0.000 | 0.005 |

![development_curves](figures/development_curves.png)

![development_performance](figures/development_performance.png)

![score_example](figures/score_example.png)

상세: [장면별 지표](metrics_by_scene.csv), [영상별 지표](metrics_by_video.csv), [실행 설정](config.json), [manifest](run_manifest.json), `logs/`, `scores/`.

### 해석

개발셋 26개 영상, 12,954프레임에서 **macro AUROC 73.13%**, **macro AP 62.96%**입니다. 장면별 AUROC는 R01 65.67%, R02 80.81%, R03 72.99%, R04 73.05%입니다. 표의 S는 결합 실험을 위한 정상 reference 보정 점수이며, 원시 공식 점수와의 AUROC 차이는 [대조표](raw_vs_calibrated_auroc.csv)에 별도로 보존합니다.

- 정상 q99 임계값에서 R04 TPR은 **0%**입니다. AUROC 73.05%라는 순위 분리 성능이 해당 경보 임계값의 높은 recall을 보장하지 않습니다. 임계값을 개발 라벨에 맞춰 낮추지 않습니다.
- R01 FPR은 **5.95%**로 정상 threshold 분할의 목표 1%보다 큽니다. 정상 분할에서 정한 q99가 새 영상에서 1% FPR을 보장하지 않는 사례입니다.
- PCA rank는 R01/R02/R03/R04 = **281/323/383/359**이고, 모든 장면에서 목표 설명분산 99%를 달성했습니다.
- frozen 특징·PCA는 결정적이어서 한 번 측정한 결과를 이후 3개 LoRA seed와 paired 비교합니다. 같은 수치를 독립 반복으로 세지 않습니다.
- 최종 평가셋은 이 단계에 사용하지 않았습니다. 결과는 LoRA 채택 판단에 쓰일 개발셋 baseline입니다.

### 검증과 비용

압축 프레임 CSV에서 별도 sklearn 지표 계산과 영상별 이벤트 순회로 모든 장면 수치를 재검증했습니다. [검증 로그](metric_verification.json)를 제공합니다. 최종 집계에서 Pandas `gt` 이름과 메서드가 충돌한 오류를 수정했으며, 저장된 점수에서 집계를 복구했습니다. 재추출은 하지 않았고 수정 이력은 recovery 로그에 남겼습니다.

추출·점수 계산 기록의 합은 403.0초, 관측 peak allocated VRAM은 2.39GiB입니다. 시간은 파일 읽기와 전처리를 포함하고 cache 압축·전체 보고서 생성은 제외합니다. 실행 중 짧은 LoRA GPU 검증이 병행됐으므로 엄격히 격리된 추론 속도 벤치마크가 아닙니다.
