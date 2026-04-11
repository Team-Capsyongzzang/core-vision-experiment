"""
models/detector.py
==================
재난 탐지기 — backbone.py의 팩토리를 사용하여 6개 백본 지원.

config.py의 cfg.detector.backbone 값으로 백본을 선택합니다.

    cfg.detector.backbone = "resnet50"           # 기본값
    cfg.detector.backbone = "mobilenet_v3_small" # 최경량 (1단계 스크리닝 후보)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.config import cfg
from models.backbone import build_backbone, print_backbone_info


class DisasterDetector(nn.Module):
    """
    재난 유무 이진 분류기.

    Forward 출력: logit (B, 1) — sigmoid 적용 전 raw score
    예측:         sigmoid(logit) >= threshold → 재난 있음
    """

    def __init__(
        self,
        backbone_name: str   | None = None,
        pretrained:    bool         = True,
        dropout:       float | None = None,
    ):
        super().__init__()
        backbone_name = backbone_name or cfg.detector.backbone
        dropout       = dropout       or cfg.detector.dropout

        # ── Backbone ─────────────────────────────────────
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.backbone_name      = backbone_name

        # ── Detection Head (이진 분류) ────────────────────
        # 분류기보다 간단한 head (탐지기는 단순 이진 판단)
        mid_dim = max(feat_dim // 8, 64)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, 1),   # 이진 분류 → 1개 출력
        )

    def forward(self, diff: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        diff : (B, 3, H, W) — |post - pre| diff 이미지

        Returns
        -------
        logit : (B, 1)
        """
        feat = self.backbone(diff)
        return self.head(feat)

    def predict(
        self,
        diff:      torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """
        Returns
        -------
        (B,) bool tensor — True: 재난 있음, False: 없음
        """
        threshold = threshold or cfg.detector.threshold
        with torch.no_grad():
            prob = torch.sigmoid(self.forward(diff)).squeeze(1)
        return prob >= threshold

    def freeze_backbone(self, freeze: bool = True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"✓ Detector backbone {'frozen' if freeze else 'unfrozen'}")

    def get_param_groups(self, base_lr: float) -> list[dict]:
        backbone_params = list(self.backbone.parameters())
        head_params = [
            p for p in self.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        factor = cfg.detector.backbone_lr_factor
        return [
            {"params": backbone_params, "lr": base_lr * factor, "name": "backbone"},
            {"params": head_params,     "lr": base_lr,          "name": "head"},
        ]


def build_detector(
    device:          torch.device | None = None,
    pretrained:      bool                = True,
    freeze_backbone: bool                = True,
) -> DisasterDetector:
    device   = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = DisasterDetector(pretrained=pretrained).to(device)

    if freeze_backbone:
        detector.freeze_backbone(freeze=True)

    n_total     = sum(p.numel() for p in detector.parameters())
    n_trainable = sum(p.numel() for p in detector.parameters() if p.requires_grad)

    print(f"✓ Detector : DisasterDetector")
    print_backbone_info(detector.backbone_name)
    print(f"  Task            : binary (재난 있음/없음)")
    print(f"  Threshold       : {cfg.detector.threshold}")
    print(f"  Total params    : {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}  "
          f"(backbone {'frozen' if freeze_backbone else 'unfrozen'})")
    print(f"  Device          : {device}")
    return detector
