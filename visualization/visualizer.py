"""
visualization/visualizer.py
============================
분류 결과 시각화.
  1. 혼동 행렬 (Confusion Matrix) 히트맵
  2. 랜덤 샘플 예측 결과 이미지 그리드
  3. 클래스별 분포 차트
"""

from __future__ import annotations

import os
import json
import random

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config.config import DISASTER_CLASSES, CLASS_COLORS, cfg


# ─────────────────────────────────────────────────────────
#  Confusion Matrix
# ─────────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm:       np.ndarray,
    save_path: str | None = None,
    title:    str         = "Confusion Matrix",
):
    """Confusion Matrix 히트맵을 시각화합니다."""
    n = len(DISASTER_CLASSES)
    fig, ax = plt.subplots(figsize=(8, 7))

    # 행 정규화 (각 GT 클래스별 비율)
    cm_norm = cm.astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm_norm / row_sum, 0)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=range(n), yticks=range(n),
        xticklabels=DISASTER_CLASSES,
        yticklabels=DISASTER_CLASSES,
        xlabel="Predicted", ylabel="Ground Truth",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # 셀에 숫자 표시
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            color = "white" if cm_norm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.0%})",
                    ha="center", va="center", fontsize=9, color=color)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Confusion matrix saved → {save_path}")
    plt.show()
    plt.close()


# ─────────────────────────────────────────────────────────
#  Sample prediction grid
# ─────────────────────────────────────────────────────────

def run_sample_predictions(
    model:     nn.Module,
    dataset:   Dataset,
    device:    torch.device,
    n_samples: int         = 12,
    pred_dir:  str | None  = None,
    seed:      int | None  = None,
) -> list[dict]:
    """
    랜덤 샘플에 대해 예측을 수행하고 결과를 시각화합니다.
    정답/오답을 테두리 색으로 구분합니다.
      - 초록 테두리: 정답
      - 빨간 테두리: 오답
    """
    pred_dir = pred_dir or cfg.train.pred_dir
    os.makedirs(pred_dir, exist_ok=True)

    n_samples = min(n_samples, len(dataset))
    indices   = (random.Random(seed).sample(range(len(dataset)), n_samples)
                 if seed is not None
                 else random.sample(range(len(dataset)), n_samples))

    model.eval()
    results = []

    n_cols = 4
    n_rows = (n_samples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    print(f"\n{'='*60}")
    print(f"  PREDICTIONS — {n_samples} random samples")
    print(f"{'='*60}")

    for plot_i, idx in enumerate(indices):
        sample = dataset.data[idx]
        batch  = dataset[idx]
        image  = batch["image"].unsqueeze(0).to(device)
        gt     = batch["disaster"]

        with torch.no_grad():
            logits = model(image)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            pred_idx = probs.argmax()
            pred     = DISASTER_CLASSES[pred_idx]
            conf     = probs[pred_idx]

        correct = (pred == gt)
        color   = "green" if correct else "red"

        # 이미지 로드 (정규화 전 원본)
        img_pil = Image.open(sample["post"]).convert("RGB").resize((224, 224))
        ax = axes[plot_i]
        ax.imshow(img_pil)
        ax.set_title(
            f"GT: {gt}\nPred: {pred} ({conf:.0%})",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        # 테두리 색
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
        ax.axis("off")

        results.append({
            "id":       sample["id"],
            "gt":       gt,
            "pred":     pred,
            "conf":     float(conf),
            "correct":  correct,
        })
        mark = "✓" if correct else "✗"
        print(f"  {mark} {sample['id']:<35}  GT={gt:<12}  Pred={pred:<12}  {conf:.0%}")

    # 빈 subplot 숨기기
    for ax in axes[len(indices):]:
        ax.axis("off")

    plt.suptitle("Disaster Classification Predictions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(pred_dir, "sample_predictions.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()
    print(f"\n✓ Grid saved → {save_path}")

    # 정확도 요약
    correct_cnt = sum(r["correct"] for r in results)
    print(f"  Sample accuracy: {correct_cnt}/{n_samples} = {correct_cnt/n_samples:.1%}")

    # JSON 저장
    with open(os.path.join(pred_dir, "predictions.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─────────────────────────────────────────────────────────
#  Class distribution
# ─────────────────────────────────────────────────────────

def plot_class_distribution(index_path: str, save_dir: str | None = None):
    """학습 데이터 클래스 분포를 막대 그래프로 시각화합니다."""
    import json
    from collections import Counter

    save_dir = save_dir or cfg.train.pred_dir
    os.makedirs(save_dir, exist_ok=True)

    with open(index_path) as f:
        data = json.load(f)

    counter = Counter(item["disaster"] for item in data)
    classes = DISASTER_CLASSES
    counts  = [counter.get(cls, 0) for cls in classes]
    colors  = [CLASS_COLORS.get(cls, "#888888") for cls in classes]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(classes, counts, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Training Data — Class Distribution", fontsize=13)
    plt.ylabel("Sample Count")
    plt.xticks(rotation=15)

    for bar, cnt in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{cnt:,}", ha="center", fontsize=10)

    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Distribution chart saved → {path}")
