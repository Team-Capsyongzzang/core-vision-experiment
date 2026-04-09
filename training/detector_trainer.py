"""
training/detector_trainer.py
=============================
재난 탐지기(이진 분류) 학습 엔진.

분류기 Trainer와 다른 점:
  - 손실: BCEWithLogitsLoss (이진 분류)
  - 클래스 불균형: pos_weight로 처리
    (재난 있음 샘플이 없음보다 많으면 pos_weight < 1)
  - 메트릭: Accuracy + F1 + Precision + Recall
  - 임계값(threshold) 조정 가능
"""

from __future__ import annotations

import math
import os
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.config import cfg, DetectorConfig


# ─────────────────────────────────────────────────────────
#  Loss
# ─────────────────────────────────────────────────────────

def build_detector_criterion(
    train_index_path: str,
    device: torch.device,
) -> nn.BCEWithLogitsLoss:
    """
    클래스 불균형을 pos_weight로 보정한 BCE Loss.

    pos_weight = n_negative / n_positive
    → 양성(재난) 샘플이 적으면 높은 가중치 부여
    """
    with open(train_index_path) as f:
        data = json.load(f)

    n_pos = sum(1 for d in data if d["has_disaster"] == 1)
    n_neg = len(data) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device)

    print(f"  Detector loss: BCEWithLogitsLoss")
    print(f"  Positive (재난 있음) : {n_pos:,}")
    print(f"  Negative (재난 없음) : {n_neg:,}")
    print(f"  pos_weight           : {pos_weight.item():.3f}")

    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ─────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────

def binary_metrics(
    logits:    torch.Tensor,
    targets:   torch.Tensor,
    threshold: float,
) -> dict:
    """이진 분류 메트릭 계산."""
    probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
    preds  = (probs >= threshold).astype(int)
    tgts   = targets.cpu().numpy().astype(int)

    tp = int(((preds == 1) & (tgts == 1)).sum())
    fp = int(((preds == 1) & (tgts == 0)).sum())
    fn = int(((preds == 0) & (tgts == 1)).sum())
    tn = int(((preds == 0) & (tgts == 0)).sum())

    acc       = (tp + tn) / max(len(tgts), 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "acc": acc, "precision": precision,
        "recall": recall, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


# ─────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────

def _save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path)


def _load_checkpoint(path: str, model: nn.Module, optimizer=None) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        saved   = len(ckpt["optimizer_state_dict"]["param_groups"])
        current = len(optimizer.param_groups)
        if saved == current:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        else:
            print(f"  ℹ️  Optimizer groups mismatch ({saved}→{current}) — weights only")
    val_f1 = ckpt.get("val_f1", float("nan"))
    print(f"✓ Loaded detector checkpoint  epoch={ckpt.get('epoch','?')}  val_f1={val_f1:.4f}")
    return ckpt


def _plot_curves(train_vals, val_vals, ylabel, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(train_vals, label=f"Train {ylabel}", marker="o")
    plt.plot(val_vals,   label=f"Val {ylabel}",   marker="s")
    plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.title(f"Detector — {ylabel}"); plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ─────────────────────────────────────────────────────────
#  Core epoch functions
# ─────────────────────────────────────────────────────────

def run_epoch_train(
    model, loader, criterion, optimizer,
    scheduler, device, grad_clip, epoch_desc,
    threshold, use_cosine,
) -> tuple[float, dict]:
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for batch in tqdm(loader, desc=epoch_desc, leave=False):
        diffs   = batch["diff"        ].to(device, non_blocking=True)
        targets = batch["has_disaster"].to(device, non_blocking=True).float()

        optimizer.zero_grad()
        logits = model(diffs).view(-1)        # (B,)
        loss   = criterion(logits, targets)

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if use_cosine:
            scheduler.step()

        total_loss += loss.item() * diffs.size(0)
        all_logits .append(logits.detach().cpu())
        all_targets.append(targets.cpu())

    n = sum(t.shape[0] for t in all_targets)
    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics     = binary_metrics(all_logits, all_targets, threshold)
    return total_loss / max(n, 1), metrics


def run_epoch_val(
    model, loader, criterion, device, epoch_desc, threshold,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=epoch_desc, leave=False):
            diffs   = batch["diff"        ].to(device, non_blocking=True)
            targets = batch["has_disaster"].to(device, non_blocking=True).float()

            logits = model(diffs).view(-1)
            loss   = criterion(logits, targets)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * diffs.size(0)
            all_logits .append(logits.cpu())
            all_targets.append(targets.cpu())

    n = sum(t.shape[0] for t in all_targets)
    all_logits  = torch.cat(all_logits)
    all_targets = torch.cat(all_targets)
    metrics     = binary_metrics(all_logits, all_targets, threshold)
    return total_loss / max(n, 1), metrics


# ─────────────────────────────────────────────────────────
#  DetectorTrainer
# ─────────────────────────────────────────────────────────

class DetectorTrainer:
    """
    재난 탐지기 학습 파이프라인.

    best model 선택 기준: val F1
    (Accuracy 대신 F1을 쓰는 이유: 클래스 불균형 시 Accuracy가 무의미할 수 있음)
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
        criterion:    nn.Module,
        det_cfg:      DetectorConfig | None = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.criterion    = criterion
        self.dc           = det_cfg or cfg.detector
        self.best_val_f1  = 0.0

    def train(self):
        dc = self.dc
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=dc.lr, weight_decay=dc.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dc.num_epochs * len(self.train_loader)
        )
        use_cosine = (dc.scheduler == "cosine")

        train_losses, val_losses = [], []
        train_f1s,    val_f1s   = [], []

        print(f"\n{'='*70}")
        print(f"  DETECTOR PHASE 1 — {dc.num_epochs} EPOCHS  (backbone frozen)")
        print(f"{'='*70}")

        for epoch in range(1, dc.num_epochs + 1):
            train_loss, train_m = run_epoch_train(
                self.model, self.train_loader, self.criterion,
                optimizer, scheduler, self.device, dc.grad_clip,
                f"Epoch {epoch}/{dc.num_epochs} [Train]",
                dc.threshold, use_cosine,
            )
            val_loss, val_m = run_epoch_val(
                self.model, self.val_loader, self.criterion,
                self.device, f"Epoch {epoch}/{dc.num_epochs} [Val]",
                dc.threshold,
            )

            train_losses.append(train_loss); val_losses.append(val_loss)
            train_f1s   .append(train_m["f1"]); val_f1s.append(val_m["f1"])

            print(f"\n  Epoch {epoch} | "
                  f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                  f"F1 {train_m['f1']:.4f}/{val_m['f1']:.4f} | "
                  f"Prec {val_m['precision']:.4f} | "
                  f"Rec {val_m['recall']:.4f}")

            self._save(epoch, optimizer, val_m["f1"])

        _plot_curves(train_losses, val_losses, "Loss",
                     os.path.join(dc.log_dir, "det_phase1_loss.png"))
        _plot_curves(train_f1s, val_f1s, "F1",
                     os.path.join(dc.log_dir, "det_phase1_f1.png"))

    def finetune(self):
        dc = self.dc
        if not dc.finetune:
            return

        best_path = os.path.join(dc.checkpoint_dir, "best.pth")
        if not os.path.exists(best_path):
            print(f"⚠️  No checkpoint at {best_path}")
            return

        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone(freeze=False)

        param_groups = self.model.get_param_groups(dc.finetune_lr)
        optimizer    = torch.optim.Adam(param_groups, weight_decay=dc.weight_decay)
        _load_checkpoint(best_path, self.model, optimizer)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=dc.finetune_epochs * len(self.train_loader)
        )
        self.best_val_f1 = 0.0

        ft_losses, ft_f1s = [], []

        print(f"\n{'='*70}")
        print(f"  DETECTOR PHASE 2 FINE-TUNING — {dc.finetune_epochs} EPOCHS")
        print(f"  backbone lr={dc.finetune_lr * dc.backbone_lr_factor:.2e}  "
              f"head lr={dc.finetune_lr:.2e}")
        print(f"{'='*70}")

        offset = dc.num_epochs
        for i in range(1, dc.finetune_epochs + 1):
            epoch = offset + i
            train_loss, train_m = run_epoch_train(
                self.model, self.train_loader, self.criterion,
                optimizer, scheduler, self.device, dc.grad_clip,
                f"Epoch {epoch} [FT Train]", dc.threshold, True,
            )
            val_loss, val_m = run_epoch_val(
                self.model, self.val_loader, self.criterion,
                self.device, f"Epoch {epoch} [FT Val]", dc.threshold,
            )

            ft_losses.append(val_loss); ft_f1s.append(val_m["f1"])

            print(f"\n  [FT] Epoch {epoch} | "
                  f"Loss {train_loss:.4f}/{val_loss:.4f} | "
                  f"F1 {train_m['f1']:.4f}/{val_m['f1']:.4f} | "
                  f"Prec {val_m['precision']:.4f} | "
                  f"Rec {val_m['recall']:.4f}")

            self._save(epoch, optimizer, val_m["f1"], suffix="_ft")

        _plot_curves(ft_losses, ft_f1s, "F1",
                     os.path.join(dc.log_dir, "det_phase2_f1.png"))

    def _save(self, epoch, optimizer, val_f1, suffix=""):
        dc = self.dc
        state = {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_f1":               val_f1,
        }
        _save_checkpoint(state, os.path.join(dc.checkpoint_dir, f"latest{suffix}.pth"))
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            _save_checkpoint(state, os.path.join(dc.checkpoint_dir, f"best{suffix}.pth"))
            print(f"  ✓ Best detector updated (val_f1={val_f1:.4f})")
