# 소스 출처와 구현 대응

## SubspaceAD

공식 commit은 configs/provenance.json에 고정합니다. vendor/subspacead의 extractor/PCA/scoring/common은 해당 commit의 원본이며 Apache-2.0 라이선스를 포함합니다.

새 추출기는 공식 ImageProcessor, 336px square resize, hidden_states[-4,-5] mean을 사용합니다. 정상 PCA는 모든 fit patch의 Chan 중심화 통계를 누적합니다. 공식 two-pass mean/covariance와 같은 통계이며 단위 및 실제 GPU 표본 대조로 검증합니다. 특징과 동일 FP32로 PCA를 변환해 원본 재구성 순서, post_process_map, mtop1p를 따릅니다. 마스킹·specular filter·whitening·test fitting은 사용하지 않습니다.

results/00_prepare/gpu_parity.json은 실제 가중치 3프레임 대조입니다. 작은 표본 검증을 전체 IPAD 성능 검증으로 해석하지 않습니다.

## 노트북

사용자 CycleVAD_Colab.ipynb의 ZIP payload에서 src/cycle_vad와 회귀 테스트를 복원했습니다. notebook/payload SHA256은 provenance에 있습니다. 이 패키지는 참고 구현이며 새 실험의 entry point는 ipad_experiment입니다.

| 요소 | 참고 구현 | 새 실험 |
|---|---|---|
| DINOv2 | Base/336px | 동일, 고정 revision |
| 전처리 | Letterbox | 공식 square resize로 통일 |
| 지역 기술자 | 후반 2층, 6×6, 64차원 | 동일 정의, projection seed42 |
| 주기 | DTW + causal Bayesian tracker | 동일 수학적 정의 |
| 조건부 평균 | Fourier 회귀 | 동일 후보, 정상 validation 선택 |
| 잔차 | 공유 PCA 하나 | 최대rank64/설명분산95% |
| 외형 보조 | pooled PCA + LocalMemory | 주 비교에서 제외, 공식 S 사용 |
| 보정 | reference tail calibrator | 동일, 독립 threshold 분할 |

S용 중간층과 C/P용 후반층은 같은 forward에서 얻지만 다른 특징입니다. C₀는 C와 동일한 특징과 rank를 써서 조건부 평균의 효과를 분리합니다. P alignment는 외형 적합도도 포함하므로 P를 순수 시간 순서 정보로 주장하지 않습니다.

## 라벨 불일치

R02 testing 12/13/14는 인덱스가 0부터 연속이지만 프레임/라벨 개수가 806/805, 609/608, 497/498입니다. 단순 파일명 누락으로 설명되지 않습니다. 특정 프레임을 제거·삽입할 공식 근거를 확보하지 못했으므로 세 영상 전체를 모든 평가에서 제외합니다. 원본 파일은 수정하지 않습니다.
