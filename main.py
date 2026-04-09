"""
main.py
=======
xBD 재난 분류 파이프라인 — 2단계 (탐지기 + 분류기).

구조:
  탐지기 (DisasterDetector)
    입력: |post - pre| diff 이미지
    출력: 재난 있음(1) / 없음(0)

  분류기 (DisasterClassifier)
    입력: post 이미지
    출력: 재난 종류 (7개 클래스)

출력 경로:
  checkpoints/
  ├── detector/     ← 탐지기 체크포인트
  └── classifier/   ← 분류기 체크포인트

  logs/
  ├── detector/     ← 탐지기 학습 커브
  └── classifier/   ← 분류기 학습 커브

  predictions/      ← 혼동 행렬, 샘플 예측 그리드

Usage
-----
python main.py                               # 전체 파이프라인
python main.py --steps train_detector        # 탐지기 학습만
python main.py --steps eval predict          # 평가 + 시각화만
python main.py --force-index                 # 인덱스 재생성 포함
"""

import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from config.config import cfg, DISASTER_CLASSES
from utils.seed import set_seed
from data.indexer import build_indexes
from data.dataset import build_dataloaders, XBDClassDataset
from data.detector_dataset import build_detector_dataloaders
from models.detector import build_detector
from models.classifier import build_model as build_classifier
from losses.losses import build_criterion
from training.detector_trainer import DetectorTrainer, build_detector_criterion
from training.trainer import Trainer
from metrics.metrics import MetricAggregator, print_metrics
from visualization.visualizer import (
    plot_confusion_matrix,
    run_sample_predictions,
    plot_class_distribution,
)


STEPS_ALL = [
    "index",
    "explore",
    "train_detector",
    "finetune_detector",
    "train_classifier",
    "finetune_classifier",
    "eval",
    "predict",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="xBD 재난 분류 파이프라인 (탐지기 + 분류기)"
    )
    parser.add_argument(
        "--steps", nargs="+", default=STEPS_ALL, choices=STEPS_ALL,
        help="실행할 단계 목록"
    )
    parser.add_argument("--force-index", action="store_true",
                        help="인덱스 강제 재생성")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
#  파이프라인 평가
# ─────────────────────────────────────────────────────────

def evaluate_pipeline(
    detector:   torch.nn.Module,
    classifier: torch.nn.Module,
    index_path: str,
    device:     torch.device,
    split_name: str = "Test",
) -> dict:
    """
    탐지기 → 분류기 순차 실행.

    탐지기 "없음" → no_disaster 반환
    탐지기 "있음" → 분류기로 재난 종류 판단
    """
    from data.detector_dataset import DiffDataset
    from data.dataset import XBDClassDataset
    from torch.utils.data import DataLoader

    diff_ds    = DiffDataset(index_path, augment=False)
    cls_ds     = XBDClassDataset(index_path, augment=False)
    no_dis_idx = DISASTER_CLASSES.index("no_disaster")
    threshold  = cfg.detector.threshold

    common = dict(
        batch_size  = cfg.data.batch_size,
        num_workers = cfg.data.num_workers,
        pin_memory  = True,
    )
    diff_loader = DataLoader(diff_ds, **common)
    cls_loader  = DataLoader(cls_ds,  **common)

    detector.eval()
    classifier.eval()

    all_preds, all_targets = [], []
    n_total = n_detected = 0

    print(f"\n{'='*60}")
    print(f"  PIPELINE EVALUATION [{split_name}]")
    print(f"  Detector threshold : {threshold}")
    print(f"{'='*60}")

    with torch.no_grad():
        for diff_batch, cls_batch in tqdm(
            zip(diff_loader, cls_loader),
            total=len(diff_loader),
            desc="Pipeline",
        ):
            diffs  = diff_batch["diff" ].to(device)
            images = cls_batch ["image"].to(device)
            labels = cls_batch ["label"].cpu().numpy()

            # ── 탐지기 ──────────────────────────────────
            probs        = torch.sigmoid(detector(diffs).view(-1)).cpu().numpy()
            has_disaster = probs >= threshold

            # ── 분류기 (탐지된 것만) ─────────────────────
            preds = np.full(len(labels), no_dis_idx, dtype=np.int64)
            if has_disaster.any():
                detected_images = images[torch.from_numpy(has_disaster)]
                cls_preds = classifier(detected_images).argmax(dim=1).cpu().numpy()
                preds[has_disaster] = cls_preds

            all_preds  .extend(preds.tolist())
            all_targets.extend(labels.tolist())
            n_total    += len(labels)
            n_detected += int(has_disaster.sum())

    print(f"\n  탐지기 통계:")
    print(f"    전체      : {n_total:,}")
    print(f"    재난 감지 : {n_detected:,} ({n_detected/max(n_total,1):.1%})")
    print(f"    재난 없음 : {n_total-n_detected:,} "
          f"({(n_total-n_detected)/max(n_total,1):.1%})")

    agg          = MetricAggregator()
    agg._preds   = all_preds
    agg._targets = all_targets
    results      = agg.compute()
    print_metrics(results, title=f"Pipeline {split_name} Results")

    return results


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    steps = set(args.steps)

    set_seed(cfg.data.seed)
    cfg.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. 인덱스 생성 ───────────────────────────────────
    if "index" in steps:
        print("\n[STEP 1] Building indexes...")
        index_paths = build_indexes(force=args.force_index)
    else:
        index_paths = {
            s: os.path.join(cfg.data.index_dir, f"{s}.json")
            for s in ("train", "val", "test")
        }

    # ── 2. 데이터 분포 탐색 ──────────────────────────────
    if "explore" in steps:
        print("\n[STEP 2] Class distribution...")
        plot_class_distribution(
            index_paths["train"],
            save_dir=cfg.train.pred_dir,
        )

    # ── DataLoaders ──────────────────────────────────────
    det_train, det_val, _ = build_detector_dataloaders(index_paths)
    cls_train, cls_val, _ = build_dataloaders(index_paths)

    # ── 탐지기 ───────────────────────────────────────────
    detector      = build_detector(device, pretrained=True, freeze_backbone=True)
    det_criterion = build_detector_criterion(index_paths["train"], device)
    det_trainer   = DetectorTrainer(
        detector, det_train, det_val, device, det_criterion
    )

    # ── 3. 탐지기 Phase 1 ────────────────────────────────
    if "train_detector" in steps:
        print("\n[STEP 3] Detector Phase 1 (backbone frozen)...")
        det_trainer.train()

    # ── 4. 탐지기 Phase 2 ────────────────────────────────
    if "finetune_detector" in steps:
        print("\n[STEP 4] Detector Phase 2 fine-tuning...")
        det_trainer.finetune()

    # ── 분류기 ───────────────────────────────────────────
    classifier    = build_classifier(device, pretrained=True, freeze_backbone=True)
    cls_criterion = build_criterion(
        index_paths["train"], device, loss_type="focal"
    )
    cls_trainer   = Trainer(
        classifier, cls_train, cls_val, device, cls_criterion
    )

    # ── 5. 분류기 Phase 1 ────────────────────────────────
    if "train_classifier" in steps:
        print("\n[STEP 5] Classifier Phase 1 (backbone frozen)...")
        cls_trainer.train()

    # ── 6. 분류기 Phase 2 ────────────────────────────────
    if "finetune_classifier" in steps:
        print("\n[STEP 6] Classifier Phase 2 fine-tuning...")
        cls_trainer.finetune()

    # ── 최적 체크포인트 경로 ─────────────────────────────
    def best_ckpt(base_dir: str) -> str:
        ft = os.path.join(base_dir, "best_ft.pth")
        return ft if os.path.exists(ft) else os.path.join(base_dir, "best.pth")

    det_ckpt = best_ckpt(cfg.detector.checkpoint_dir)
    cls_ckpt = best_ckpt(cfg.train.checkpoint_dir)

    # ── 7. 파이프라인 평가 ───────────────────────────────
    if "eval" in steps:
        print("\n[STEP 7] Pipeline evaluation...")

        ckpt = torch.load(det_ckpt, map_location=device)
        detector.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Detector  loaded "
              f"(epoch={ckpt.get('epoch','?')}, "
              f"val_f1={ckpt.get('val_f1', float('nan')):.4f})")

        ckpt = torch.load(cls_ckpt, map_location=device)
        classifier.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Classifier loaded "
              f"(epoch={ckpt.get('epoch','?')}, "
              f"val_acc={ckpt.get('val_acc', float('nan')):.4f})")

        results = evaluate_pipeline(
            detector, classifier,
            index_paths["test"], device, split_name="Test",
        )
        plot_confusion_matrix(
            results["confusion_matrix"],
            save_path=os.path.join(cfg.train.pred_dir, "confusion_matrix.png"),
            title="Confusion Matrix — Test Set",
        )

    # ── 8. 샘플 예측 시각화 ──────────────────────────────
    if "predict" in steps:
        print("\n[STEP 8] Sample predictions...")

        # eval 건너뛴 경우를 위한 체크포인트 로드
        if "eval" not in steps:
            ckpt = torch.load(det_ckpt, map_location=device)
            detector.load_state_dict(ckpt["model_state_dict"])
            ckpt = torch.load(cls_ckpt, map_location=device)
            classifier.load_state_dict(ckpt["model_state_dict"])

        val_ds = XBDClassDataset(index_paths["val"], augment=False)
        run_sample_predictions(
            classifier, val_ds, device,
            n_samples=12,
            pred_dir=cfg.train.pred_dir,
        )

    # ── 완료 요약 ─────────────────────────────────────────
    print("\n✅ Pipeline complete.")
    print(f"\n  출력 경로:")
    print(f"    체크포인트 (탐지기)  : {cfg.detector.checkpoint_dir}/")
    print(f"    체크포인트 (분류기)  : {cfg.train.checkpoint_dir}/")
    print(f"    학습 커브  (탐지기)  : {cfg.detector.log_dir}/")
    print(f"    학습 커브  (분류기)  : {cfg.train.log_dir}/")
    print(f"    평가 결과 / 시각화   : {cfg.train.pred_dir}/")


if __name__ == "__main__":
    main()
