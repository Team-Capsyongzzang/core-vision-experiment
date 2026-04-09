"""
evaluation/evaluator.py
=======================
분류 모델 평가 모듈.
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics.metrics import MetricAggregator, print_metrics


def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    split_name: str = "Evaluation",
) -> dict:
    """val / test 모두 처리하는 단일 평가 함수."""
    model.eval()
    aggregator = MetricAggregator()
    total_loss, total = 0.0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating [{split_name}]"):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item() * images.size(0)
                total      += images.size(0)

            aggregator.update(logits, labels)

    results = aggregator.compute()
    results["loss"] = total_loss / max(total, 1)

    print_metrics(results, title=f"{split_name} Set Results")
    print(f"  Loss : {results['loss']:.4f}")

    return results


def load_and_evaluate(
    model:           nn.Module,
    loader:          DataLoader,
    criterion:       nn.Module,
    device:          torch.device,
    checkpoint_path: str,
    split_name:      str = "Test",
) -> dict:
    """체크포인트를 로드하고 평가합니다."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    val_acc = ckpt.get("val_acc", float("nan"))
    print(f"✓ Loaded checkpoint  epoch={ckpt.get('epoch','?')}  val_acc={val_acc:.4f}")

    return evaluate(model, loader, criterion, device, split_name=split_name)
