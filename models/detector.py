"""
models/detector.py
==================
재난 탐지기 — |post - pre| diff 이미지로 재난 있음/없음 이진 분류.

방법 A (DisasterClassifier)와 구조는 거의 동일하지만:
  - 출력이 7개 클래스 → 2개 (있음/없음)
  - 입력이 post 이미지 → diff 이미지 (3ch 동일)
  - sigmoid 기반 임계값으로 최종 판정

탐지기가 "있음"으로 판정한 경우에만 분류기로 전달합니다.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from config.config import cfg


class DisasterDetector(nn.Module):
    """
    재난 유무 이진 분류기.

    Forward 출력: logit (B, 1) — sigmoid 적용 전 raw score
    예측:         sigmoid(logit) > threshold → 재난 있음
    """

    def __init__(
        self,
        pretrained: bool        = True,
        dropout:    float | None = None,
    ):
        super().__init__()
        dropout = dropout or cfg.detector.dropout

        weights  = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet   = tvm.resnet50(weights=weights)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # → (B, 2048, 1, 1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),     # 이진 분류 → 1개 출력
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
        feat  = self.backbone(diff)
        return self.head(feat)

    def predict(
        self,
        diff:      torch.Tensor,
        threshold: float | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        threshold : sigmoid 임계값. None이면 config 값 사용.

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

    print(f"✓ Detector : DisasterDetector (ResNet50, pretrained={pretrained})")
    print(f"  Task            : binary (재난 있음/없음)")
    print(f"  Threshold       : {cfg.detector.threshold}")
    print(f"  Total params    : {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}  "
          f"(backbone {'frozen' if freeze_backbone else 'unfrozen'})")
    print(f"  Device          : {device}")
    return detector
