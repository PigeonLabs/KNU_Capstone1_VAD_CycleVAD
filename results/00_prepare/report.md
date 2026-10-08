# 준비 단계 결과

| scene | fit | validation | reference | threshold | development | final |
| --- | --- | --- | --- | --- | --- | --- |
| R01 | 19 | 5 | 5 | 5 | 6 | 9 |
| R02 | 17 | 5 | 4 | 4 | 5 | 7 |
| R03 | 12 | 4 | 3 | 3 | 7 | 10 |
| R04 | 14 | 4 | 4 | 3 | 8 | 11 |

영상 단위 고정 분할입니다. 원본 녹화 그룹·실제 cycle 경계는 미확인입니다.

![Data split](figures/data_split.png)

실제 사전학습 DINOv2 3프레임 공식 구현 대조: **passed**. 특징 최대 오차 0, PCA projector 최대 오차 0, 점수 최대 오차 2.38e-07.
