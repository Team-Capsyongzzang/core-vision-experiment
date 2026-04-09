"""
config/config.py
================
xBD 재난 분류기 전체 설정.
"""

from dataclasses import dataclass, field
from typing import List
import os


# ─────────────────────────────────────────────
#  재난 클래스 정의
# ─────────────────────────────────────────────

# xBD 파일명 prefix → 재난 종류 매핑
DISASTER_MAP = {
    "socal-fire":          "wildfire",
    "santa-rosa-wildfire": "wildfire",
    "pinery-bushfire":     "wildfire",
    "woolsey-fire":        "wildfire",
    "midwest-flooding":    "flood",
    "nepal-flooding":      "flood",
    "hurricane-harvey":    "hurricane",
    "hurricane-florence":  "hurricane",
    "hurricane-matthew":   "hurricane",
    "hurricane-michael":   "hurricane",
    "joplin-tornado":      "tornado",
    "moore-tornado":       "tornado",
    "tuscaloosa-tornado":  "tornado",
    "mexico-earthquake":   "earthquake",
    "palu-tsunami":        "tsunami",
}

# 레이블 인덱스
# no_disaster를 맨 앞에 고정 배치 (index=0)
# → 나머지는 알파벳 정렬로 자동 결정
DISASTER_CLASSES = ["no_disaster"] + sorted(set(DISASTER_MAP.values()))
# ['no_disaster', 'earthquake', 'flood', 'hurricane', 'tornado', 'tsunami', 'wildfire']

CLASS_TO_IDX = {cls: i for i, cls in enumerate(DISASTER_CLASSES)}
IDX_TO_CLASS = {i: cls for cls, i in CLASS_TO_IDX.items()}
NUM_CLASSES   = len(DISASTER_CLASSES)

# 시각화용 색상
CLASS_COLORS = {
    "earthquake": "#8B4513",
    "flood":      "#1E90FF",
    "hurricane":  "#9370DB",
    "tornado":    "#FF8C00",
    "tsunami":    "#00CED1",
    "wildfire":   "#FF4500",
    "no_disaster": "#AAAAAA",  # 회색
}


@dataclass
class DataConfig:
    data_root: str = "/home/moa/.cache/kagglehub/datasets/qianlanzz/xbd-dataset/versions/1/xbd"
    index_dir: str = "./xbd_cls_index"

    # 샘플링 비율
    tier1_ratio: float = 0.75
    tier3_ratio: float = 0.70
    test_ratio:  float = 0.40
    train_val_split: float = 0.80   # 분류는 val 비율 줄여도 됨

    # 입력: post 이미지만 사용
    use_post_only: bool = True
    image_size: int = 224            # ResNet50 표준 입력 크기

    batch_size:  int  = 32           # 분류는 세그멘테이션보다 배치 크게 가능
    num_workers: int  = 2
    pin_memory:  bool = True
    seed: int = 42


@dataclass
class ModelConfig:
    backbone: str  = "resnet50"      # resnet50 | resnet101 | efficientnet_b0
    pretrained: bool = True
    num_classes: int = NUM_CLASSES
    dropout: float = 0.3



@dataclass
class DetectorConfig:
    """
    방법 B - 탐지기 설정.
    탐지기는 |post - pre| diff 이미지로 재난 있음/없음을 이진 분류합니다.
    """
    backbone: str   = "resnet50"
    pretrained: bool = True
    dropout: float  = 0.3

    num_epochs: int   = 15
    lr: float         = 1e-4
    weight_decay: float = 1e-4
    backbone_lr_factor: float = 0.1

    finetune: bool       = True
    finetune_epochs: int = 8
    finetune_lr: float   = 2e-5

    scheduler: str   = "cosine"
    grad_clip: float = 1.0

    # 탐지기 전용 체크포인트 경로
    checkpoint_dir: str = "./checkpoints/detector"
    log_dir: str        = "./logs/detector"
    pred_dir: str       = "./predictions"

    # 양성(재난 있음) 임계값 — 이 값 이상이면 "재난 있음"으로 판정
    # 0.5보다 낮게 설정하면 더 민감하게 탐지 (False Negative 줄임)
    threshold: float = 0.4


@dataclass
class TrainConfig:
    num_epochs: int   = 20
    lr: float         = 1e-4
    weight_decay: float = 1e-4

    # Backbone 차등 lr (pretrained 보호)
    backbone_lr_factor: float = 0.1  # backbone lr = lr * factor

    # Fine-tuning
    finetune: bool        = True
    finetune_epochs: int  = 10
    finetune_lr: float    = 2e-5

    # 스케줄러
    scheduler: str        = "cosine"   # cosine | plateau
    warmup_epochs: int    = 2

    # 기타
    grad_clip: float      = 1.0
    checkpoint_dir: str   = "./checkpoints"
    log_dir: str          = "./logs"
    pred_dir: str         = "./predictions"


@dataclass
class Config:
    data:     DataConfig     = field(default_factory=DataConfig)
    model:    ModelConfig    = field(default_factory=ModelConfig)
    train:    TrainConfig    = field(default_factory=TrainConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def make_dirs(self):
        for d in [
            self.data.index_dir,
            self.train.checkpoint_dir,
            self.train.log_dir,
            self.train.pred_dir,
            self.detector.checkpoint_dir,
            self.detector.log_dir,
        ]:
            os.makedirs(d, exist_ok=True)


cfg = Config()
