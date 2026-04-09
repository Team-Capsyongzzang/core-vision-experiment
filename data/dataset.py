"""
data/dataset.py
===============
재난 분류용 Dataset / DataLoader.

입력: post 이미지 1장 (3ch, 224×224)
출력: 재난 종류 레이블 (int)

세그멘테이션과 달리:
  - 마스크 불필요
  - ImageNet 정규화 적용
  - 증강: RandomCrop, ColorJitter 등 분류 표준 augmentation
"""

from __future__ import annotations

import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from config.config import cfg, NUM_CLASSES


class XBDClassDataset(Dataset):
    """
    xBD 재난 분류 Dataset.

    Returns
    -------
    dict:
        image   : FloatTensor (3, H, W)  — post 이미지
        label   : int                    — 재난 클래스 인덱스
        disaster: str                    — 재난 이름 (시각화용)
        id      : str                    — 샘플 ID
    """

    # ImageNet mean/std
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        index_file: str,
        augment:    bool      = False,
        image_size: int | None = None,
    ):
        with open(index_file) as f:
            self.data = json.load(f)

        self.augment    = augment
        self.image_size = image_size or cfg.data.image_size

        # ── 학습용 transform ─────────────────────────────
        self.train_tf = T.Compose([
            T.RandomResizedCrop(self.image_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])

        # ── 평가용 transform ─────────────────────────────
        self.val_tf = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(self.MEAN, self.STD),
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item  = self.data[idx]
        image = Image.open(item["post"]).convert("RGB")

        tf    = self.train_tf if self.augment else self.val_tf
        image = tf(image)

        return {
            "image":    image,
            "label":    item["label"],
            "disaster": item["disaster"],
            "id":       item["id"],
        }


def build_dataloaders(
    index_paths: dict[str, str],
    batch_size:  int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    train / val / test DataLoader를 생성합니다.
    """
    dc          = cfg.data
    batch_size  = batch_size  or dc.batch_size
    num_workers = num_workers or dc.num_workers

    train_ds = XBDClassDataset(index_paths["train"], augment=True)
    val_ds   = XBDClassDataset(index_paths["val"],   augment=False)
    test_ds  = XBDClassDataset(index_paths["test"],  augment=False)

    common = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = dc.pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,
                              persistent_workers=(num_workers > 0), **common)
    val_loader   = DataLoader(val_ds,   shuffle=False,
                              persistent_workers=(num_workers > 0), **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)

    print("✓ DataLoaders created")
    print(f"  Train : {len(train_ds):,} samples  ({len(train_loader):,} batches)")
    print(f"  Val   : {len(val_ds):,} samples  ({len(val_loader):,} batches)")
    print(f"  Test  : {len(test_ds):,} samples  ({len(test_loader):,} batches)")

    return train_loader, val_loader, test_loader
