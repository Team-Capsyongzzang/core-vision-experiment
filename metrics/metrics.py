"""
metrics/metrics.py
==================
분류 태스크 평가 메트릭.

세그멘테이션(IoU, Dice)과 달리 분류는:
  - Accuracy (전체 / 클래스별)
  - F1 Score (macro / per-class)
  - Confusion Matrix
를 주로 사용합니다.
"""

from __future__ import annotations

import numpy as np
import torch
from collections import defaultdict

from config.config import DISASTER_CLASSES


class MetricAggregator:
    """
    배치 단위로 예측값을 누적하고 최종 메트릭을 계산합니다.

    Usage
    -----
    agg = MetricAggregator()
    for batch in loader:
        logits = model(batch['image'])
        agg.update(logits, batch['label'])
    results = agg.compute()
    """

    def __init__(self, num_classes: int = len(DISASTER_CLASSES)):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self._preds   = []
        self._targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Parameters
        ----------
        logits  : (B, C) — raw model output
        targets : (B,)   — ground truth class indices
        """
        preds = logits.argmax(dim=1).cpu().numpy()
        self._preds  .extend(preds.tolist())
        self._targets.extend(targets.cpu().numpy().tolist())

    def compute(self) -> dict:
        """
        누적된 예측값으로 전체 메트릭을 계산합니다.

        Returns
        -------
        dict with keys:
            accuracy         : float  — 전체 정확도
            accuracy_per_cls : array  — 클래스별 정확도
            f1_macro         : float  — macro F1
            f1_per_cls       : array  — 클래스별 F1
            confusion_matrix : array  — (C, C) confusion matrix
            precision_per_cls: array
            recall_per_cls   : array
        """
        preds   = np.array(self._preds)
        targets = np.array(self._targets)
        C       = self.num_classes

        # ── Confusion Matrix ─────────────────────────────
        cm = np.zeros((C, C), dtype=np.int64)
        for t, p in zip(targets, preds):
            cm[t][p] += 1

        # ── Per-class metrics ─────────────────────────────
        precision = np.zeros(C)
        recall    = np.zeros(C)
        f1        = np.zeros(C)
        acc_cls   = np.zeros(C)

        for c in range(C):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            tn = cm.sum() - tp - fp - fn

            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[c]    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1[c]        = (2 * precision[c] * recall[c] / (precision[c] + recall[c])
                            if (precision[c] + recall[c]) > 0 else 0.0)
            acc_cls[c]   = tp / cm[c].sum() if cm[c].sum() > 0 else 0.0

        return {
            "accuracy":          float((preds == targets).mean()),
            "accuracy_per_cls":  acc_cls,
            "f1_macro":          float(f1.mean()),
            "f1_per_cls":        f1,
            "precision_per_cls": precision,
            "recall_per_cls":    recall,
            "confusion_matrix":  cm,
        }


def print_metrics(results: dict, title: str = "Evaluation Results"):
    """메트릭 결과를 표 형식으로 출력합니다."""
    w = 70
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}")
    print(f"  Overall Accuracy : {results['accuracy']:.4f}")
    print(f"  Macro F1 Score   : {results['f1_macro']:.4f}")
    print(f"\n  {'Class':<15} {'Acc':>7} {'Precision':>10} {'Recall':>8} {'F1':>7}")
    print(f"  {'-'*50}")

    for i, cls in enumerate(DISASTER_CLASSES):
        print(f"  {cls:<15}"
              f"  {results['accuracy_per_cls'][i]:.4f}"
              f"  {results['precision_per_cls'][i]:>9.4f}"
              f"  {results['recall_per_cls'][i]:>7.4f}"
              f"  {results['f1_per_cls'][i]:>6.4f}")

    print(f"\n  Confusion Matrix (rows=GT, cols=Pred):")
    print(f"  {'':15}", end="")
    for cls in DISASTER_CLASSES:
        print(f"  {cls[:6]:>6}", end="")
    print()

    for i, cls in enumerate(DISASTER_CLASSES):
        print(f"  {cls:<15}", end="")
        for j in range(len(DISASTER_CLASSES)):
            val = results["confusion_matrix"][i, j]
            # 대각선(정답)은 강조
            marker = "★" if i == j else " "
            print(f"  {val:>5}{marker}", end="")
        print()

    print(f"{'='*w}\n")
