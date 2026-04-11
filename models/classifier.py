"""
models/classifier.py
====================
재난 분류기 — backbone.py의 팩토리를 사용하여 6개 백본 지원.

config.py의 cfg.model.backbone 값으로 백본을 선택합니다.

    cfg.model.backbone = "resnet50"           # 기본값
    cfg.model.backbone = "efficientnet_b0"    # 경량 고성능
    cfg.model.backbone = "mobilenet_v3_small" # 최경량
"""

from __future__ import annotations

import torch
import torch.nn as nn

from config.config import cfg, DISASTER_CLASSES
from models.backbone import build_backbone, print_backbone_info


class DisasterClassifier(nn.Module):
    """
    xBD 재난 종류 분류기.

    backbone.py의 팩토리로 백본을 교체할 수 있습니다.
    head는 feat_dim에 맞게 자동으로 구성됩니다.
    """

    def __init__(
        self,
        backbone_name: str   | None = None,
        num_classes:   int   | None = None,
        pretrained:    bool         = True,
        dropout:       float | None = None,
    ):
        super().__init__()
        backbone_name = backbone_name or cfg.model.backbone
        num_classes   = num_classes   or cfg.model.num_classes
        dropout       = dropout       or cfg.model.dropout

        # ── Backbone ─────────────────────────────────────
        self.backbone, feat_dim = build_backbone(backbone_name, pretrained)
        self.backbone_name      = backbone_name

        # ── Classification Head ───────────────────────────
        # feat_dim은 백본마다 다름 (2048 / 1280 / 1536 / 960 / 576)
        # 중간 레이어는 feat_dim의 1/4로 자동 조정
        mid_dim = max(feat_dim // 4, 128)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid_dim, num_classes),
        )

        self.class_names = DISASTER_CLASSES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat   = self.backbone(x)   # (B, feat_dim, 1, 1)
        logits = self.head(feat)    # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        with torch.no_grad():
            indices = self.forward(x).argmax(dim=1)
        names = [self.class_names[i] for i in indices.cpu().tolist()]
        return indices, names

    def freeze_backbone(self, freeze: bool = True):
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"✓ Classifier backbone {'frozen' if freeze else 'unfrozen'}")

    def get_param_groups(self, base_lr: float) -> list[dict]:
        backbone_params = list(self.backbone.parameters())
        head_params = [
            p for p in self.parameters()
            if not any(p is bp for bp in backbone_params)
        ]
        factor = cfg.train.backbone_lr_factor
        return [
            {"params": backbone_params, "lr": base_lr * factor, "name": "backbone"},
            {"params": head_params,     "lr": base_lr,          "name": "head"},
        ]


def build_model(
    device:          torch.device | None = None,
    pretrained:      bool                = True,
    freeze_backbone: bool                = True,
) -> DisasterClassifier:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DisasterClassifier(pretrained=pretrained).to(device)

    if freeze_backbone:
        model.freeze_backbone(freeze=True)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Classifier : DisasterClassifier")
    print_backbone_info(model.backbone_name)
    print(f"  Classes         : {model.class_names}")
    print(f"  Total params    : {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}  "
          f"(backbone {'frozen' if freeze_backbone else 'unfrozen'})")
    print(f"  Device          : {device}")
    return model
