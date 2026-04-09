"""
data/detector_dataset.py
=========================
탐지기(Detector)용 Dataset.

입력: |post - pre| pixel-level diff 이미지 (3ch)
출력: has_disaster (0 or 1)

Why pixel diff?
    pre 이미지와 post 이미지의 픽셀 차이를 절댓값으로 취하면
    재난으로 변화한 영역이 밝게 강조됩니다.
    변화가 없는 no_disaster 이미지는 diff가 거의 0(검정)에 가깝습니다.

    재난 있음:  diff 이미지에 밝은 영역 뚜렷
    재난 없음:  diff 이미지가 거의 검정 (pre == post이므로)
"""

from __future__ import annotations

import json

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

from config.config import cfg


class DiffDataset(Dataset):
    """
    |post - pre| diff 이미지 Dataset.

    Returns
    -------
    dict:
        diff        : FloatTensor (3, H, W)  — |post - pre| 정규화
        has_disaster: int (0 or 1)
        disaster    : str
        id          : str
    """

    MEAN = [0.5, 0.5, 0.5]   # diff 이미지는 ImageNet 통계보다 중립적 정규화
    STD  = [0.5, 0.5, 0.5]

    def __init__(
        self,
        index_file: str,
        augment:    bool       = False,
        image_size: int | None = None,
    ):
        with open(index_file) as f:
            self.data = json.load(f)

        self.augment    = augment
        self.image_size = image_size or cfg.data.image_size

        self.resize    = T.Resize((self.image_size, self.image_size))
        self.normalize = T.Normalize(self.MEAN, self.STD)

        # diff 이미지용 augmentation
        # 주의: pre/post를 동일하게 변환해야 diff가 의미있음
        #       → 아래에서 seed 기반 동기화
        self.aug_params = dict(
            hflip=0.5,
            vflip=0.5,
            rotation=15,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        pre  = Image.open(item["pre" ]).convert("RGB")
        post = Image.open(item["post"]).convert("RGB")

        # ── Resize ───────────────────────────────────────
        pre  = self.resize(pre)
        post = self.resize(post)

        # ── Augmentation (pre/post 동기화) ───────────────
        if self.augment:
            import random
            seed = random.randint(0, 2**31)
            pre, post = self._sync_augment(pre, post, seed)

        # ── Pixel diff 계산 ──────────────────────────────
        # float32로 변환 후 절댓값 차이
        pre_arr  = np.array(pre,  dtype=np.float32) / 255.0
        post_arr = np.array(post, dtype=np.float32) / 255.0
        diff_arr = np.abs(post_arr - pre_arr)          # (H, W, 3), [0, 1]

        # ── To Tensor + Normalize ────────────────────────
        diff_t = torch.from_numpy(diff_arr).permute(2, 0, 1)   # (3, H, W)
        diff_t = self.normalize(diff_t)

        return {
            "diff":         diff_t,
            "has_disaster": item["has_disaster"],
            "disaster":     item["disaster"],
            "id":           item["id"],
        }

    def _sync_augment(
        self,
        pre:  Image.Image,
        post: Image.Image,
        seed: int,
    ) -> tuple[Image.Image, Image.Image]:
        """pre/post에 동일한 augmentation을 적용합니다."""
        import torchvision.transforms.functional as TF
        import random

        random.seed(seed)
        if random.random() < self.aug_params["hflip"]:
            pre  = TF.hflip(pre)
            post = TF.hflip(post)

        random.seed(seed + 1)
        if random.random() < self.aug_params["vflip"]:
            pre  = TF.vflip(pre)
            post = TF.vflip(post)

        random.seed(seed + 2)
        angle = random.uniform(
            -self.aug_params["rotation"],
             self.aug_params["rotation"],
        )
        pre  = TF.rotate(pre,  angle)
        post = TF.rotate(post, angle)

        return pre, post


def build_detector_dataloaders(
    index_paths: dict[str, str],
    batch_size:  int | None = None,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dc          = cfg.data
    batch_size  = batch_size  or dc.batch_size
    num_workers = num_workers or dc.num_workers

    train_ds = DiffDataset(index_paths["train"], augment=True)
    val_ds   = DiffDataset(index_paths["val"],   augment=False)
    test_ds  = DiffDataset(index_paths["test"],  augment=False)

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

    # 레이블 분포 출력
    from collections import Counter
    import json
    with open(index_paths["train"]) as f:
        train_data = json.load(f)
    counter = Counter(item["has_disaster"] for item in train_data)
    total   = len(train_data)
    print("✓ Detector DataLoaders created")
    print(f"  Train : {len(train_ds):,} samples  "
          f"(disaster={counter[1]:,} / no_disaster={counter[0]:,})")
    print(f"  Val   : {len(val_ds):,} samples")
    print(f"  Test  : {len(test_ds):,} samples")

    return train_loader, val_loader, test_loader
