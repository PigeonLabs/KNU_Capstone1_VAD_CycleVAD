# 실험 규약 — 2026-10-09

## 목표와 순서

Frozen baseline → 정상 영상 LoRA 채택 판단 → 선택 백본의 C/P ablation. 각 단계의 검증 후 코드·로그·그림·README를 main에 게시한다. 성능 향상을 완료 조건으로 삼지 않는다.

## 데이터

R01–R04. 정상 training 영상은 fit/validation/reference/threshold 55/15/15/15%, seed42 고정. 최대 나머지 방식, 동률은 앞 분할 우선. 유효 testing은 장면별 normal/mixed/anomaly strata 안에서 약40% development/60% final, 개수×0.4 반올림, strata당 영상이 2개 이상이면 양쪽에 최소1개. label은 층화·평가용이며 모델 적합에 사용하지 않는다. R02 test12/13/14 제외. 실제 cycle 경계·원본 녹화 그룹 미확인. 모든 평가 stride1, 영상마다 tracker reset. 미래·test길이·영상별min-max 없음. 과거 test 결과 확인 이력이 있는 후속 실험이다.

## Baseline

DINOv2-base/336px/FP32, 공식 layers[-4,-5] mean. 전체 fit patch PCA99%, whitening없음, reconstruction 및 원본 map resize/blur/top1%mean. FP64 통계, TF32 off. Frozen 계산은 결정적이므로 1회 측정값을 paired seed에 공유한다. 실제 GPU 특징·점수를 공식 원본과 대조한다. development만 먼저 보고한다.

## LoRA

장면·seed42/43/44별 attention query/value, r8/alpha16/dropout0. 원래 가중치 frozen. Teacher는 원본 DINOv2. 정상 입력 brightness/contrast 각각0.9–1.1, noise sigma0–0.01(픽셀0–1). 증강 학생 특징을 clean teacher 특징에 맞추고 clean 보존 손실0.1을 추가한다. S용 중간 patch, C/P용 두 후반 patch, final CLS의 MSE를 각 teacher 평균제곱+1e-8로 나누고 동일 가중 평균한다. 기하·합성 결함·cycle 학습목표 없음.

AdamW lr1e-4/wd.01, 최대20epoch/patience5, effective batch32/micro8, epoch별 영상당128 균등구간 표본. validation은 영상당64 고정프레임·고정증강. warmup1epoch후cosine. 정상 validation loss 최소 checkpoint, 동률은 이른 epoch. 개발 AUROC로 학습설정 선택 없음.

채택: development 3-seed 평균 macro AUROC delta≥.01 AND paired 95%CI 하한>0. 장면·라벨 strata 영상단위10,000 bootstrap, seed20261009. 동일 표본을 모든 방법·seed에 적용. 전체장면에 한 번 결정, 미충족시frozen. 실행오류는 미향상으로 세지 않는다.

## Cycle ablation

정상 DTW+causal tracker grid128/latent24. Fourier harmonics{2,4,8}, ridge{.01,.1,1} 정상validation MSE선택. 공유잔차PCA maxrank64/variance95%/balanced4096. S=보정SubspaceAD; C=confidence×max(보정외부잔차,내부Mahalanobis); P=max(보정alignment,innovation,progress). max결합/가중치1. reference tail보정/threshold q99.

주비교 S/S+C/S+P/S+C+P/S+C₀/S+C₀+P. C₀는 동일특징·표본·confidence·실제rank의 상수평균. 진단 C/P단독, C confidence제거, P각성분결합. pooledPCA/LocalMemory제외. LoRA채택시frozen S도 최종보고. 최종결과로 설정 재선택 없음.

## 지표와 게시

주지표 장면frame AUROC/4장면macro. AP/F1/TPR/FPR, event coverage/첫경보지연, 시간·VRAM·rank·설명분산. 단일클래스 영상AUROC NaN, 장면합산에는 포함. 영상경계간 event연결 금지. 주요개선량 paired bootstrap CI.

results/00_prepare,01_baseline,02_lora,03_ablation에 누적. frame index·GT·scalar score 압축CSV로 GPU없이 재계산. raw영상·가중치·특징캐시는 로컬artifacts. 각 단계 완료시 README 표·그림·해석과 함께 main게시, 업로드 후 원격 파일·그림·commit 확인. 실패·하락도 기록한다.
