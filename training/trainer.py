"""
training/trainer.py
===================
재난 분류 학습 엔진.

2단계 학습 전략:
  Phase 1: backbone frozen → head만 학습 (lr=1e-4, 20 epochs)
  Phase 2: backbone unfrozen → 전체 fine-tuning
           backbone lr = finetune_lr * 0.1
           head lr     = finetune_lr
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from config.config import cfg, TrainConfig
from metrics.metrics import MetricAggregator, print_metrics


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

    val_acc = ckpt.get("val_acc", float("nan"))
    print(f"✓ Loaded checkpoint  epoch={ckpt.get('epoch','?')}  val_acc={val_acc:.4f}")
    return ckpt


def _plot_curves(train_vals, val_vals, ylabel, title, save_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_vals, label=f"Train {ylabel}", marker="o")
    plt.plot(val_vals,   label=f"Val {ylabel}",   marker="s")
    plt.xlabel("Epoch"); plt.ylabel(ylabel)
    plt.title(title); plt.legend(); plt.grid(True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ Curve saved → {save_path}")


def _build_scheduler(optimizer, tc: TrainConfig, n_steps_per_epoch: int):
    if tc.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=tc.num_epochs * n_steps_per_epoch,
        )
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )


# ─────────────────────────────────────────────────────────
#  Core train / validate
# ─────────────────────────────────────────────────────────

def run_epoch_train(
    model:      nn.Module,
    loader:     DataLoader,
    criterion:  nn.Module,
    optimizer:  torch.optim.Optimizer,
    scheduler,
    device:     torch.device,
    grad_clip:  float,
    epoch_desc: str = "Train",
    use_cosine_scheduler: bool = True,
) -> tuple[float, float]:
    """
    Returns
    -------
    (avg_loss, accuracy)
    """
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc=epoch_desc, leave=False):
        images  = batch["image" ].to(device, non_blocking=True)
        labels  = batch["label" ].to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print("  ⚠️  NaN/Inf loss — skipped")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # cosine scheduler는 step마다 업데이트
        if use_cosine_scheduler:
            scheduler.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def run_epoch_val(
    model:     nn.Module,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
    epoch_desc: str = "Val",
) -> tuple[float, float]:
    """
    Returns
    -------
    (avg_loss, accuracy)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=epoch_desc, leave=False):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            logits = model(images)
            loss   = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


# ─────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────

class Trainer:
    """
    재난 분류 학습 파이프라인.

    Usage
    -----
    trainer = Trainer(model, train_loader, val_loader, device, criterion)
    trainer.train()     # Phase 1: backbone frozen
    trainer.finetune()  # Phase 2: backbone unfrozen
    """

    def __init__(
        self,
        model:        nn.Module,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        device:       torch.device,
        criterion:    nn.Module,
        train_cfg:    TrainConfig | None = None,
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.criterion    = criterion
        self.tc           = train_cfg or cfg.train

        self.best_val_acc  = 0.0
        self.train_losses: list[float] = []
        self.val_losses:   list[float] = []
        self.train_accs:   list[float] = []
        self.val_accs:     list[float] = []

    # ── Phase 1 ──────────────────────────────────────────

    def train(self):
        """Phase 1: backbone frozen → head만 학습."""
        tc = self.tc

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=tc.lr,
            weight_decay=tc.weight_decay,
        )
        scheduler = _build_scheduler(optimizer, tc, len(self.train_loader))
        use_cosine = (tc.scheduler == "cosine")

        print(f"\n{'='*70}")
        print(f"  PHASE 1 TRAINING — {tc.num_epochs} EPOCHS  (backbone frozen)")
        print(f"{'='*70}")

        for epoch in range(1, tc.num_epochs + 1):
            train_loss, train_acc = run_epoch_train(
                self.model, self.train_loader, self.criterion,
                optimizer, scheduler, self.device, tc.grad_clip,
                epoch_desc=f"Epoch {epoch}/{tc.num_epochs} [Train]",
                use_cosine_scheduler=use_cosine,
            )
            val_loss, val_acc = run_epoch_val(
                self.model, self.val_loader, self.criterion,
                self.device,
                epoch_desc=f"Epoch {epoch}/{tc.num_epochs} [Val]",
            )

            if not use_cosine:
                scheduler.step(val_acc)

            self._record(train_loss, val_loss, train_acc, val_acc)
            self._log(epoch, train_loss, val_loss, train_acc, val_acc)
            self._save(epoch, optimizer, val_acc)

        _plot_curves(self.train_losses, self.val_losses,
                     "Loss", "Phase 1 Loss",
                     os.path.join(tc.log_dir, "cls_phase1_loss.png"))
        _plot_curves(self.train_accs, self.val_accs,
                     "Accuracy", "Phase 1 Accuracy",
                     os.path.join(tc.log_dir, "cls_phase1_acc.png"))

    # ── Phase 2 ──────────────────────────────────────────

    def finetune(self):
        """Phase 2: backbone unfrozen → 전체 fine-tuning."""
        tc = self.tc
        if not tc.finetune:
            print("ℹ️  Finetune disabled — skipping.")
            return

        best_path = os.path.join(tc.checkpoint_dir, "best.pth")
        if not os.path.exists(best_path):
            print(f"⚠️  No checkpoint at {best_path} — run train() first.")
            return

        # backbone unfreeze + 차등 lr
        if hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone(freeze=False)

        param_groups = self.model.get_param_groups(tc.finetune_lr)
        optimizer    = torch.optim.Adam(param_groups, weight_decay=tc.weight_decay)

        _load_checkpoint(best_path, self.model, optimizer)

        # fine-tuning용 cosine scheduler
        ft_tc = TrainConfig(
            num_epochs=tc.finetune_epochs,
            scheduler="cosine",
        )
        scheduler  = _build_scheduler(optimizer, ft_tc, len(self.train_loader))
        self.best_val_acc = 0.0

        ft_train_losses, ft_val_losses = [], []
        ft_train_accs,   ft_val_accs   = [], []

        print(f"\n{'='*70}")
        print(f"  PHASE 2 FINE-TUNING — {tc.finetune_epochs} EPOCHS")
        print(f"  backbone lr={tc.finetune_lr*cfg.train.backbone_lr_factor:.2e}  "
              f"head lr={tc.finetune_lr:.2e}")
        print(f"{'='*70}")

        offset = tc.num_epochs
        for i in range(1, tc.finetune_epochs + 1):
            epoch = offset + i
            train_loss, train_acc = run_epoch_train(
                self.model, self.train_loader, self.criterion,
                optimizer, scheduler, self.device, tc.grad_clip,
                epoch_desc=f"Epoch {epoch} [FT Train]",
                use_cosine_scheduler=True,
            )
            val_loss, val_acc = run_epoch_val(
                self.model, self.val_loader, self.criterion,
                self.device,
                epoch_desc=f"Epoch {epoch} [FT Val]",
            )

            ft_train_losses.append(train_loss)
            ft_val_losses  .append(val_loss)
            ft_train_accs  .append(train_acc)
            ft_val_accs    .append(val_acc)

            self._log(epoch, train_loss, val_loss, train_acc, val_acc, tag="FT")
            self._save(epoch, optimizer, val_acc, suffix="_ft")

        _plot_curves(ft_train_losses, ft_val_losses,
                     "Loss", "Phase 2 Loss",
                     os.path.join(tc.log_dir, "cls_phase2_loss.png"))
        _plot_curves(ft_train_accs, ft_val_accs,
                     "Accuracy", "Phase 2 Accuracy",
                     os.path.join(tc.log_dir, "cls_phase2_acc.png"))

    # ── Helpers ──────────────────────────────────────────

    def _record(self, tl, vl, ta, va):
        self.train_losses.append(tl)
        self.val_losses  .append(vl)
        self.train_accs  .append(ta)
        self.val_accs    .append(va)

    def _log(self, epoch, tl, vl, ta, va, tag=""):
        tag_str = f"[{tag}] " if tag else ""
        print(f"\n  {tag_str}Epoch {epoch} | "
              f"Loss {tl:.4f}/{vl:.4f} | "
              f"Acc {ta:.4f}/{va:.4f}")

    def _save(self, epoch, optimizer, val_acc, suffix=""):
        tc    = self.tc
        state = {
            "epoch":                epoch,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc":              val_acc,
        }
        _save_checkpoint(state, os.path.join(tc.checkpoint_dir, f"latest{suffix}.pth"))

        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            _save_checkpoint(state, os.path.join(tc.checkpoint_dir, f"best{suffix}.pth"))
            print(f"  ✓ Best model updated (val_acc={val_acc:.4f})")
