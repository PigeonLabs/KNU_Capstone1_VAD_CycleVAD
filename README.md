# IPAD Cycle-Aware SubspaceAD

**연속 공정 진행도에 따라 정상 평균을 바꾸고 잔차 subspace를 공유하는 모듈의 유용성**을 IPAD R01–R04에서 검증합니다.

실험 순서는 **Frozen SubspaceAD → 정상 영상 LoRA의 채택 판단 → 선택한 백본에서 C/P ablation**입니다. 각 단계의 코드·로그·결과·그림을 함께 게시합니다. 아직 실행하지 않은 결과나 성능 개선을 주장하지 않습니다.

## 진행 상태

| 단계 | 상태 | 결과 |
| --- | --- | --- |
| 데이터·설정·구현 검증 | 완료 | [상세](results/00_prepare/report.md) |
| Frozen SubspaceAD | 미실행 | — |
| LoRA 학습·채택 판단 | 미실행 | — |
| 진행도 모듈 ablation | 미실행 | — |

## 방법과 평가 규약

- DINOv2-base 336px, 공식 중간층 `[-4,-5]` 평균, 정상 patch PCA 99% 설명분산, 공식 재구성 잔차·확대/블러·상위 1% 집계. 원 논문의 Giant/672px 벤치마크 수치 재현이 아닌 **IPAD 방법 적용 실험**입니다.
- S: SubspaceAD, C: 진행도 조건부 외형 점수, P: 정렬·innovation·진행 오차. 결합은 `max`, 가중치 1로 고정합니다.
- 정상 training을 fit/validation/reference/threshold = 55/15/15/15%로 영상 단위 분리합니다. 유효 testing은 약 40% 개발 / 60% 최종 평가입니다. 프레임 단위 random split을 사용하지 않습니다.
- 주 지표는 장면별 frame AUROC와 네 장면 macro 평균입니다. AP, 정상 q99 임계값의 F1/TPR/FPR와 이벤트 탐지도 보고합니다. 모든 평가 프레임을 stride 1로 관측합니다.
- LoRA 채택 기준: 개발셋 seed 42·43·44 평균 macro AUROC +1%p 이상 **및** 영상 단위 paired bootstrap 95% 구간 하한 > 0. 최종 결과를 보고 백본을 다시 선택하지 않습니다.
- 실제 cycle 경계와 원본 녹화 그룹이 없습니다. `weak_recording_alignment`를 사용하며 cycle 위치 정확도·원본 그룹 독립성을 주장하지 않습니다.
- R02 testing 12·13·14는 프레임/라벨 길이가 불일치하여 공통 제외합니다. 자세한 목록은 [exclusions.csv](results/00_prepare/exclusions.csv)를 참고하세요.
- 과거 테스트 결과 확인 이력이 있는 후속 실험입니다. 최종 분할은 이번 모델 선택에서 제외하지만 완전히 새로운 미관측 데이터로 표현하지 않습니다.

![고정 데이터 분할](results/00_prepare/figures/data_split.png)

설정: [experiment.json](configs/experiment.json) · [분할 manifest](results/00_prepare/splits.json) · [상세 규약](docs/EXPERIMENT_DESIGN.md) · [출처와 대응](docs/SOURCE_MAPPING.md)

## Frozen SubspaceAD

미실행 또는 집계 전입니다.

## LoRA 학습·채택 판단

미실행 또는 집계 전입니다.

## 진행도 모듈 ablation

미실행 또는 집계 전입니다.

## 재현

```bash
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python scripts/download_model.py
python -m ipad_experiment.pipeline prepare --data-root /path/to/IPAD_dataset
python -m pytest -q
python scripts/verify_gpu.py
python -m ipad_experiment.pipeline baseline --data-root /path/to/IPAD_dataset
python -m ipad_experiment.pipeline report
```

후속 단계 CLI는 `train-lora`, `select-backbone`, `ablation`입니다. 각 단계는 구현·검증 후 실행 상태를 갱신합니다. `report`는 저장된 점수·지표에서 그림과 README를 다시 생성하며 GPU가 필요하지 않습니다. [실행/분석 노트북](notebooks/Experiment.ipynb)을 함께 제공합니다.

## 결과 파일 정책

GitHub에는 코드·설정·분할·구조화 로그·지표·압축 프레임 점수 CSV·시각자료를 보관합니다. `results/<run_id>`의 manifest에 실행 코드 commit과 SHA256을 남깁니다. 원본 영상/프레임, 특징 캐시, 가중치와 체크포인트는 로컬 `artifacts/`에 보존합니다. 압축 CSV만으로 지표를 독립 재계산할 수 있습니다.

## 출처

- [SubspaceAD 논문](https://arxiv.org/abs/2602.23013), [공식 구현](https://github.com/CLendering/SubspaceAD). 고정 commit 및 라이선스는 [provenance](configs/provenance.json), [Apache-2.0](vendor/SUBSPACEAD_LICENSE)에 기록합니다.
- [IPAD 논문](https://arxiv.org/abs/2404.15033).
- 사용자 제공 `CycleVAD_Colab.ipynb`의 내장 구현: 정상 템플릿·causal tracker·Fourier 평균·공유 잔차 PCA·정상 tail 보정. 노트북 SHA256은 provenance에 기록합니다.
