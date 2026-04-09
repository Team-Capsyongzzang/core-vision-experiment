"""
losses/losses.py
================
재난 분류용 손실 함수.

xBD 클래스 불균형:
  - flood, hurricane, wildfire 샘플이 많음
  - tsunami, earthquake 샘플이 적음
  → Weighted CrossEntropy 또는 Focal Loss로 보정
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg, DISASTER_CLASSES


def compute_class_weights(
    index_path: str,
    device: torch.device,
) -> torch.Tensor:
    """
    학습 인덱스에서 클래스 빈도를 계산하고
    inverse frequency 가중치를 반환합니다.

    적은 클래스(tsunami, earthquake)에 높은 가중치를 부여합니다.
    """
    import json
    from collections import Counter

    with open(index_path) as f:
        data = json.load(f)

    counter   = Counter(item["label"] for item in data)
    n_classes = len(DISASTER_CLASSES)
    counts    = np.array([counter.get(i, 1) for i in range(n_classes)], dtype=np.float32)

    # inverse frequency balancing
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes   # normalize

    print("  Class weights:")
    for i, (cls, w) in enumerate(zip(DISASTER_CLASSES, weights)):
        print(f"    {cls:<15} count={int(counts[i]):>4}  weight={w:.3f}")

    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss for classification.

    어려운 샘플(잘 못 맞추는 클래스)에 더 집중해서 학습합니다.
    특히 tsunami, earthquake처럼 샘플 수 적고 헷갈리는 클래스에 효과적.

    Parameters
    ----------
    alpha   : 클래스별 가중치 텐서 (compute_class_weights로 계산)
    gamma   : focusing parameter (0이면 CE와 동일, 보통 2.0)
    """

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float               = 2.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (B, C)  — raw model output
        targets : (B,)    — class indices
        """
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        pt      = torch.exp(-ce_loss.clamp(min=0))
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


def build_criterion(
    train_index_path: str,
    device: torch.device,
    loss_type: str = "focal",   # "ce" | "focal"
) -> nn.Module:
    """
    손실 함수를 생성합니다.

    Parameters
    ----------
    loss_type : "ce" (CrossEntropy) | "focal" (FocalLoss)
    """
    print("Computing class weights...")
    weights = compute_class_weights(train_index_path, device)

    if loss_type == "focal":
        criterion = FocalLoss(alpha=weights, gamma=2.0)
        print(f"✓ Loss: FocalLoss (gamma=2.0, weighted)")
    else:
        criterion = nn.CrossEntropyLoss(weight=weights)
        print(f"✓ Loss: CrossEntropyLoss (weighted)")

    return criterion
