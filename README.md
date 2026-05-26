# core-lightvision-experiment

xBD 위성 이미지에서 재난 발생 여부를 먼저 탐지하고, 재난이 감지된 경우에만 재난 종류를 분류하는 경량 비전 파이프라인.

## 태스크 정의

```
입력:
  - pre-disaster 위성 이미지
  - post-disaster 위성 이미지

출력:
  1단계: 재난 있음 / 재난 없음
  2단계: no_disaster / earthquake / flood / hurricane / tornado / tsunami / wildfire
```

## 파이프라인 구조

```
pre 이미지 ─┐
            ├─→ |post - pre| diff 이미지 ─→ [DisasterDetector]
post 이미지 ┘                                      │
                                                   ├─ 재난 없음 → no_disaster
                                                   │
                                                   └─ 재난 있음 → [DisasterClassifier]
                                                                  └─ 재난 종류 분류
```

탐지기는 pre/post의 픽셀 차이를 절댓값으로 계산한 diff 이미지를 사용합니다.  
재난이 없는 샘플은 pre 이미지를 post로도 사용하여 diff가 거의 0에 가깝도록 구성합니다.

## 레이블 생성 방식

xBD 파일명 prefix에서 재난 종류를 자동 추출합니다.

```
파일명: socal-fire_00000371       → wildfire
파일명: midwest-flooding_00000181 → flood
파일명: palu-tsunami_00000012     → tsunami
```

`no_disaster` 클래스는 재난 전(pre-disaster) 이미지를 정상 샘플로 사용해 생성합니다.

## 프로젝트 구조

```
core-lightvision-experiment/
├── config/config.py              ← 클래스, 데이터 경로, 학습 설정
├── data/
│   ├── indexer.py                ← xBD 파일명 → 레이블 / 인덱스 생성
│   ├── dataset.py                ← 재난 분류기용 Dataset / DataLoader
│   └── detector_dataset.py       ← diff 이미지 기반 탐지기 Dataset / DataLoader
├── models/
│   ├── backbone.py               ← 6개 백본 팩토리
│   ├── detector.py               ← 재난 있음/없음 이진 탐지기
│   └── classifier.py             ← 재난 종류 분류기
├── losses/losses.py              ← Focal Loss + 클래스 가중치
├── metrics/metrics.py            ← Accuracy, F1, Confusion Matrix
├── training/
│   ├── detector_trainer.py       ← 탐지기 학습 / fine-tuning
│   └── trainer.py                ← 분류기 학습 / fine-tuning
├── evaluation/evaluator.py       ← 평가 유틸리티
├── visualization/visualizer.py   ← 분포, 혼동 행렬, 샘플 예측 시각화
├── experiment.py                 ← 백본별 자동 실험 / 비교
└── main.py                       ← 전체 파이프라인 CLI 진입점
```

## 지원 백본

| 백본 | 파라미터 | 목적 |
|---|---:|---|
| resnet50 | 약 25M | 기본 베이스라인 |
| resnet101 | 약 44M | 성능 상한선 확인 |
| efficientnet_b0 | 약 5M | 경량 고성능 후보 |
| efficientnet_b3 | 약 12M | 성능/크기 균형 후보 |
| mobilenet_v3_large | 약 5M | 모바일 경량화 후보 |
| mobilenet_v3_small | 약 3M | 최경량 1단계 스크리닝 후보 |

탐지기와 분류기 모두 같은 백본 팩토리를 사용하므로, 실험 스크립트에서 백본을 쉽게 교체할 수 있습니다.

## 실행

```bash
# 전체 파이프라인
python main.py

# 인덱스 강제 재생성 포함
python main.py --force-index

# 탐지기만 학습
python main.py --steps index train_detector finetune_detector

# 분류기만 학습
python main.py --steps index train_classifier finetune_classifier

# 저장된 체크포인트로 평가 및 예측 시각화
python main.py --steps eval predict
```

## 백본 실험

```bash
# 지원 백본 목록 확인
python experiment.py --list

# 전체 백본 실험
python experiment.py

# 일부 백본만 실험
python experiment.py --backbones efficientnet_b0 mobilenet_v3_small

# 분류기 백본만 교체
python experiment.py --target classifier

# 저장된 결과 비교표만 출력
python experiment.py --compare-only
```

실험 결과는 `results/experiment_results.json`에 저장되며, 백본별 체크포인트와 로그는 `results/{backbone_name}/` 아래에 분리 저장됩니다.

## 학습 전략

```
탐지기 Phase 1:
  입력: |post - pre| diff 이미지
  출력: 재난 있음 / 없음
  손실: BCEWithLogitsLoss + pos_weight
  기준: validation F1

탐지기 Phase 2:
  backbone unfrozen
  낮은 learning rate로 fine-tuning

분류기 Phase 1:
  입력: post-disaster 이미지
  출력: 7개 클래스
  손실: Focal Loss + 클래스 가중치

분류기 Phase 2:
  backbone unfrozen
  backbone/head 차등 learning rate 적용
```

## 출력 경로

```
checkpoints/
├── detector/                 ← 탐지기 best / best_ft 체크포인트
└── classifier 또는 루트        ← 분류기 best / best_ft 체크포인트

logs/
├── detector/                 ← 탐지기 loss/F1 커브
└── classifier 또는 루트        ← 분류기 학습 로그

predictions/
├── class_distribution.png
├── confusion_matrix.png
└── sample_predictions.png
```

## 예상 활용

```
위성 이미지 pre/post 쌍
    │
    ├─→ [탐지기] 재난 없음 → no_disaster
    │
    └─→ [탐지기] 재난 있음
            │
            └─→ [분류기] flood / wildfire / ...

→ "이 지역은 재난 변화가 감지되었고, 유형은 flood로 추정됨"
→ 이후 건물 피해 세그멘테이션 및 LLM 리포트 생성 단계와 연결
```
