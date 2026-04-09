"""
data/indexer.py
===============
xBD 파일명에서 재난 종류 레이블을 자동 추출하여
train / val / test JSON 인덱스를 생성합니다.

핵심 로직:
    'socal-fire_00000371' → prefix 'socal-fire' → 'wildfire'
    'midwest-flooding_00000181' → prefix 'midwest-flooding' → 'flood'

알 수 없는 재난 종류(DISASTER_MAP에 없는 prefix)는 자동으로 제외합니다.

방식 A (no_disaster 클래스):
    재난 전(pre) 이미지를 no_disaster 샘플로 사용합니다.
    클래스 균형을 위해 no_disaster 샘플 수를 재난 클래스 평균에 맞춥니다.
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from glob import glob
from pathlib import Path
from tqdm import tqdm

from config.config import cfg, DISASTER_MAP, CLASS_TO_IDX


def extract_disaster(image_id: str) -> str | None:
    """
    파일명 ID에서 재난 종류를 추출합니다.

    예: 'socal-fire_00000371' → 'wildfire'
        'midwest-flooding_00000181' → 'flood'
        'unknown-event_00000001' → None (제외)
    """
    # 마지막 '_숫자' 부분을 제거해 prefix 추출
    parts = image_id.rsplit("_", 1)
    if len(parts) < 2:
        return None
    prefix = parts[0]
    return DISASTER_MAP.get(prefix, None)


def index_split(
    data_root: str,
    split_name: str,
    ratio: float,
    seed: int,
) -> list[dict]:
    """
    tier1 / tier3 / test 중 하나를 인덱싱합니다.
    post 이미지 경로와 재난 레이블을 반환합니다.
    """
    rng = random.Random(seed)

    img_dir   = os.path.join(data_root, split_name, "images")
    # post 이미지만 수집 (재난 탐지는 재난 후 이미지로 판단)
    post_images = sorted(glob(os.path.join(img_dir, "*_post_disaster.png")))
    rng.shuffle(post_images)
    selected = post_images[: int(len(post_images) * ratio)]

    index   = []
    skipped = 0

    for post_path in tqdm(selected, desc=f"Indexing {split_name}"):
        base_name = Path(post_path).stem.replace("_post_disaster", "")
        disaster  = extract_disaster(base_name)

        if disaster is None:
            skipped += 1
            continue

        pre_path = post_path.replace("_post_disaster.png", "_pre_disaster.png")
        index.append({
            "id":         base_name,
            "pre":        pre_path,   # 방법 B 탐지기에서 diff 계산에 사용
            "post":       post_path,
            "disaster":   disaster,
            "label":      CLASS_TO_IDX[disaster],
            "has_disaster": 1,        # 탐지기용 이진 레이블
        })

    if skipped:
        print(f"  ⚠️  {split_name}: {skipped} samples skipped (unknown disaster type)")

    return index


def index_no_disaster(
    data_root:   str,
    split_name:  str,
    n_samples:   int,
    seed:        int,
) -> list[dict]:
    """
    재난 전(pre) 이미지를 no_disaster 샘플로 수집합니다.

    Parameters
    ----------
    n_samples : 수집할 샘플 수.
                클래스 균형을 위해 재난 클래스 평균과 맞추는 것을 권장합니다.
                -1이면 가능한 모든 샘플을 수집합니다.

    Why pre 이미지?
        - pre 이미지는 재난 발생 전의 정상 상태 위성 이미지
        - 실제 서비스에서 재난 없는 이미지와 시각적으로 가장 유사
        - 별도 데이터셋 없이 xBD 안에서 해결 가능
    """
    rng     = random.Random(seed)
    img_dir = os.path.join(data_root, split_name, "images")

    pre_images = sorted(glob(os.path.join(img_dir, "*_pre_disaster.png")))
    rng.shuffle(pre_images)

    if n_samples > 0:
        pre_images = pre_images[:n_samples]

    no_dis_label = CLASS_TO_IDX["no_disaster"]
    index = []

    for pre_path in tqdm(pre_images, desc=f"Indexing {split_name} [no_disaster]"):
        base_name = Path(pre_path).stem.replace("_pre_disaster", "")
        index.append({
            "id":         base_name,
            "pre":        pre_path,    # no_disaster는 pre 자체가 정상 이미지
            "post":       pre_path,    # post도 pre로 설정 → diff ≈ 0
            "disaster":   "no_disaster",
            "label":      no_dis_label,
            "has_disaster": 0,         # 탐지기용 이진 레이블
        })

    return index


def print_label_distribution(data: list[dict], name: str):
    """레이블 분포를 출력합니다."""
    counter = Counter(item["disaster"] for item in data)
    total   = len(data)
    print(f"\n  {name} distribution ({total} samples):")
    for cls, cnt in sorted(counter.items()):
        bar = "█" * int(cnt / total * 30)
        print(f"    {cls:<15} {cnt:>4}  {cnt/total:>5.1%}  {bar}")


def build_indexes(
    data_root: str | None = None,
    out_dir:   str | None = None,
    force:     bool       = False,
) -> dict[str, str]:
    """
    전체 인덱싱 파이프라인 실행.

    Returns
    -------
    {"train": path, "val": path, "test": path}
    """
    dc       = cfg.data
    data_root = data_root or dc.data_root
    out_dir   = out_dir   or dc.index_dir
    os.makedirs(out_dir, exist_ok=True)

    paths = {
        split: os.path.join(out_dir, f"{split}.json")
        for split in ("train", "val", "test")
    }

    if not force and all(os.path.exists(p) for p in paths.values()):
        print("✓ Index files already exist (use force=True to rebuild)")
        return paths

    # ── 재난 클래스 인덱싱 ───────────────────────────────
    tier1 = index_split(data_root, "tier1", dc.tier1_ratio, dc.seed)
    tier3 = index_split(data_root, "tier3", dc.tier3_ratio, dc.seed + 1)
    test  = index_split(data_root, "test",  dc.test_ratio,  dc.seed + 2)

    # train / val 분할
    combined = tier1 + tier3
    rng      = random.Random(dc.seed)
    rng.shuffle(combined)
    split_idx  = int(dc.train_val_split * len(combined))
    train_data = combined[:split_idx]
    val_data   = combined[split_idx:]

    # ── no_disaster 샘플 추가 ────────────────────────────
    # 재난 클래스 평균 샘플 수에 맞춰 no_disaster 수집
    # → 클래스 불균형 최소화
    from collections import Counter
    disaster_counts = Counter(item["disaster"] for item in train_data)
    n_disaster_cls  = len(disaster_counts)
    avg_per_cls     = len(train_data) // max(n_disaster_cls, 1)
    print(f"  no_disaster target samples: {avg_per_cls} (avg per disaster class)")

    train_no_dis = index_no_disaster(
        data_root, "tier1",
        n_samples=int(avg_per_cls * dc.train_val_split),
        seed=dc.seed + 10,
    )
    val_no_dis = index_no_disaster(
        data_root, "tier1",
        n_samples=int(avg_per_cls * (1 - dc.train_val_split)),
        seed=dc.seed + 11,
    )
    test_no_dis = index_no_disaster(
        data_root, "test",
        n_samples=len(test) // max(n_disaster_cls, 1),
        seed=dc.seed + 12,
    )

    train_data = train_data + train_no_dis
    val_data   = val_data   + val_no_dis
    test       = test       + test_no_dis

    # 최종 shuffle (no_disaster가 뒤에 몰리지 않도록)
    rng.shuffle(train_data)
    rng.shuffle(val_data)
    rng.shuffle(test)

    # ── 저장 ────────────────────────────────────────────
    for key, data in [("train", train_data), ("val", val_data), ("test", test)]:
        with open(paths[key], "w") as f:
            json.dump(data, f, indent=2)

    # ── 분포 출력 ────────────────────────────────────────
    print("\n✓ Indexing complete")
    print_label_distribution(train_data, "Train")
    print_label_distribution(val_data,   "Val  ")
    print_label_distribution(test,       "Test ")

    return paths
