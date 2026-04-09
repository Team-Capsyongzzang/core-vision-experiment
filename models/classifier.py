"""
models/classifier.py
====================
pretrained ResNet50 기반 재난 분류기.

구조:
    ResNet50 (ImageNet pretrained)
        → GlobalAvgPool (2048-dim)
        → Dropout
        → FC (2048 → 512)
        → ReLU
        → Dropout
        → FC (512 → num_classes)

세그멘테이션 모델과의 차이:
    - 출력이 픽셀 맵이 아닌 이미지당 클래스 확률 벡터
    - 입력이 3ch (post 이미지만)
    - 훨씬 가볍고 빠름 (~25M params)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

from config.config import cfg, DISASTER_CLASSES


class DisasterClassifier(nn.Module):
    """
    xBD 재난 종류 분류기.

    Parameters
    ----------
    num_classes : 분류할 재난 종류 수 (기본값: config에서)
    pretrained  : ImageNet pretrained 가중치 사용 여부
    dropout     : Dropout 비율
    """

    def __init__(
        self,
        num_classes: int   | None = None,
        pretrained:  bool         = True,
        dropout:     float | None = None,
    ):
        super().__init__()
        num_classes = num_classes or cfg.model.num_classes
        dropout     = dropout     or cfg.model.dropout

        # ── Backbone: ResNet50 ────────────────────────────
        weights    = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet     = tvm.resnet50(weights=weights)

        # ResNet50의 마지막 FC 레이어 앞까지 가져옴
        # resnet.fc 는 (2048 → 1000) — 이걸 우리 head로 교체
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        # 출력: (B, 2048, 1, 1)

        # ── Classification Head ───────────────────────────
        self.head = nn.Sequential(
            nn.Flatten(),                          # (B, 2048)
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),           # (B, num_classes)
        )

        # 클래스 이름 저장 (예측 시 사용)
        self.class_names = DISASTER_CLASSES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W)

        Returns
        -------
        logits : (B, num_classes)
        """
        feat   = self.backbone(x)   # (B, 2048, 1, 1)
        logits = self.head(feat)    # (B, num_classes)
        return logits

    def predict(self, x: torch.Tensor) -> tuple[torch.Tensor, list[str]]:
        """
        logits → 예측 클래스 인덱스 + 이름 반환.

        Returns
        -------
        indices : (B,)  — 예측 클래스 인덱스
        names   : list[str] — 예측 클래스 이름
        """
        with torch.no_grad():
            logits = self.forward(x)
            indices = logits.argmax(dim=1)
        names = [self.class_names[i] for i in indices.cpu().tolist()]
        return indices, names

    def freeze_backbone(self, freeze: bool = True):
        """Backbone 가중치 동결/해제."""
        for param in self.backbone.parameters():
            param.requires_grad = not freeze
        print(f"✓ Backbone {'frozen' if freeze else 'unfrozen'}")

    def get_param_groups(self, base_lr: float) -> list[dict]:
        """
        Backbone / Head 차등 lr 파라미터 그룹 반환.
        backbone lr = base_lr * backbone_lr_factor (기본 0.1)
        """
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
    device:    torch.device | None = None,
    pretrained: bool               = True,
    freeze_backbone: bool          = True,
) -> DisasterClassifier:
    """
    DisasterClassifier를 생성합니다.

    Parameters
    ----------
    freeze_backbone : Phase1에서 backbone을 동결할지 여부.
                      True → head만 먼저 학습 (빠른 수렴)
                      False → 처음부터 전체 학습
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DisasterClassifier(pretrained=pretrained).to(device)

    if freeze_backbone:
        model.freeze_backbone(freeze=True)

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"✓ Model : DisasterClassifier (ResNet50, pretrained={pretrained})")
    print(f"  Classes         : {model.class_names}")
    print(f"  Total params    : {n_total:,}")
    print(f"  Trainable params: {n_trainable:,}  (backbone {'frozen' if freeze_backbone else 'unfrozen'})")
    print(f"  Device          : {device}")
    return model
