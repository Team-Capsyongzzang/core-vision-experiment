"""
experiment.py
=============
6개 백본 자동 실험 스크립트.

각 백본에 대해 동일한 조건으로 학습/평가를 수행하고
결과를 results/experiment_results.json에 누적 저장합니다.

Usage
-----
# 전체 6개 백본 순차 실험
python experiment.py

# 특정 백본만
python experiment.py --backbones efficientnet_b0 mobilenet_v3_large

# 탐지기는 고정, 분류기 백본만 교체
python experiment.py --target classifier

# 결과 비교표 출력 (학습 없이)
python experiment.py --compare-only
"""

import argparse
import json
import os
import time

import torch
import numpy as np

from config.config import cfg, DISASTER_CLASSES
from models.backbone import SUPPORTED_BACKBONES, BACKBONE_INFO, list_backbones
from utils.seed import set_seed


RESULTS_DIR  = "./results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "experiment_results.json")


# ─────────────────────────────────────────────────────────
#  결과 저장 / 로드
# ─────────────────────────────────────────────────────────

def load_results() -> dict:
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            return json.load(f)
    return {}


def save_results(results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)


# ─────────────────────────────────────────────────────────
#  추론 속도 측정
# ─────────────────────────────────────────────────────────

def measure_inference_time(
    model:      torch.nn.Module,
    device:     torch.device,
    input_size: tuple = (1, 3, 224, 224),
    n_runs:     int   = 50,
) -> float:
    """
    배치 1장 기준 평균 추론 시간(ms)을 측정합니다.

    warmup 10회 후 n_runs회 평균을 반환합니다.
    """
    model.eval()
    dummy = torch.randn(input_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    # 측정
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)  # ms

    return float(np.mean(times))


# ─────────────────────────────────────────────────────────
#  단일 백본 실험
# ─────────────────────────────────────────────────────────

def run_experiment(
    backbone_name: str,
    target:        str,   # "both" | "classifier" | "detector"
    device:        torch.device,
) -> dict:
    """
    단일 백본에 대해 학습 → 평가 → 추론 속도 측정을 수행합니다.

    체크포인트는 results/{backbone_name}/checkpoints/ 에 저장됩니다.
    """
    print(f"\n{'#'*70}")
    print(f"  실험: {backbone_name.upper()}")
    print(f"{'#'*70}")

    # ── config 업데이트 ──────────────────────────────────
    if target in ("both", "classifier"):
        cfg.model.backbone = backbone_name
    if target in ("both", "detector"):
        cfg.detector.backbone = backbone_name

    # 백본별 체크포인트/로그 경로 분리
    base = os.path.join(RESULTS_DIR, backbone_name)
    cfg.train.checkpoint_dir    = os.path.join(base, "checkpoints", "classifier")
    cfg.train.log_dir           = os.path.join(base, "logs", "classifier")
    cfg.detector.checkpoint_dir = os.path.join(base, "checkpoints", "detector")
    cfg.detector.log_dir        = os.path.join(base, "logs", "detector")
    cfg.train.pred_dir          = os.path.join(base, "predictions")
    cfg.make_dirs()

    set_seed(cfg.data.seed)

    # ── imports (지연 import로 config 변경 후 반영) ───────
    from data.indexer import build_indexes
    from data.dataset import build_dataloaders
    from data.detector_dataset import build_detector_dataloaders
    from models.detector import build_detector
    from models.classifier import build_model as build_classifier
    from losses.losses import build_criterion
    from training.detector_trainer import DetectorTrainer, build_detector_criterion
    from training.trainer import Trainer
    from metrics.metrics import MetricAggregator, print_metrics
    from visualization.visualizer import plot_confusion_matrix

    index_paths = build_indexes(force=False)

    det_train, det_val, _ = build_detector_dataloaders(index_paths)
    cls_train, cls_val, _ = build_dataloaders(index_paths)

    # ── 탐지기 ───────────────────────────────────────────
    detector      = build_detector(device, pretrained=True, freeze_backbone=True)
    det_criterion = build_detector_criterion(index_paths["train"], device)
    det_trainer   = DetectorTrainer(detector, det_train, det_val, device, det_criterion)

    if target in ("both", "detector"):
        print(f"\n  → 탐지기 학습...")
        det_trainer.train()
        det_trainer.finetune()

    # ── 분류기 ───────────────────────────────────────────
    classifier    = build_classifier(device, pretrained=True, freeze_backbone=True)
    cls_criterion = build_criterion(index_paths["train"], device, loss_type="focal")
    cls_trainer   = Trainer(classifier, cls_train, cls_val, device, cls_criterion)

    if target in ("both", "classifier"):
        print(f"\n  → 분류기 학습...")
        cls_trainer.train()
        cls_trainer.finetune()

    # ── 체크포인트 로드 ───────────────────────────────────
    def best_ckpt(base_dir):
        ft = os.path.join(base_dir, "best_ft.pth")
        return ft if os.path.exists(ft) else os.path.join(base_dir, "best.pth")

    ckpt = torch.load(best_ckpt(cfg.detector.checkpoint_dir), map_location=device)
    detector.load_state_dict(ckpt["model_state_dict"])

    ckpt = torch.load(best_ckpt(cfg.train.checkpoint_dir), map_location=device)
    classifier.load_state_dict(ckpt["model_state_dict"])

    # ── 파이프라인 평가 ───────────────────────────────────
    from main import evaluate_pipeline
    results = evaluate_pipeline(
        detector, classifier,
        index_paths["test"], device, split_name="Test",
    )

    # 혼동 행렬 저장
    plot_confusion_matrix(
        results["confusion_matrix"],
        save_path=os.path.join(cfg.train.pred_dir, "confusion_matrix.png"),
        title=f"Confusion Matrix — {backbone_name}",
    )

    # ── 추론 속도 측정 ────────────────────────────────────
    print(f"\n  → 추론 속도 측정...")
    det_ms = measure_inference_time(detector,   device)
    cls_ms = measure_inference_time(classifier, device)

    # ── 결과 정리 ─────────────────────────────────────────
    info = BACKBONE_INFO[backbone_name]
    experiment_result = {
        "backbone":       backbone_name,
        "params_m":       info["params_m"],
        "imagenet_acc":   info["imagenet_acc"],
        "accuracy":       round(results["accuracy"],  4),
        "f1_macro":       round(results["f1_macro"],  4),
        "f1_per_cls":     {
            cls: round(float(results["f1_per_cls"][i]), 4)
            for i, cls in enumerate(DISASTER_CLASSES)
        },
        "det_inf_ms":     round(det_ms, 2),
        "cls_inf_ms":     round(cls_ms, 2),
        "total_inf_ms":   round(det_ms + cls_ms, 2),
        # 효율 점수: F1 / (Params / 25M) — 파라미터 대비 성능
        "efficiency":     round(results["f1_macro"] / (info["params_m"] / 25.0), 4),
    }

    print(f"\n  ✓ 실험 완료: {backbone_name}")
    print(f"    Accuracy    : {experiment_result['accuracy']:.4f}")
    print(f"    Macro F1    : {experiment_result['f1_macro']:.4f}")
    print(f"    Inf. time   : {experiment_result['total_inf_ms']:.1f}ms "
          f"(det={det_ms:.1f} + cls={cls_ms:.1f})")
    print(f"    Efficiency  : {experiment_result['efficiency']:.4f}")

    return experiment_result


# ─────────────────────────────────────────────────────────
#  결과 비교표 출력
# ─────────────────────────────────────────────────────────

def print_comparison(results: dict):
    if not results:
        print("  실험 결과가 없습니다.")
        return

    print(f"\n{'='*90}")
    print(f"  백본 실험 결과 비교")
    print(f"{'='*90}")
    print(f"  {'백본':<22} {'Params':>7} {'Acc':>7} {'F1':>7} "
          f"{'Inf(ms)':>9} {'Effic.':>8}  비고")
    print(f"  {'-'*80}")

    # F1 기준 정렬
    sorted_results = sorted(results.values(), key=lambda x: x["f1_macro"], reverse=True)

    best_f1   = max(r["f1_macro"]  for r in sorted_results)
    best_eff  = max(r["efficiency"] for r in sorted_results)
    best_spd  = min(r["total_inf_ms"] for r in sorted_results)

    for r in sorted_results:
        marks = []
        if r["f1_macro"]    == best_f1:   marks.append("🏆F1")
        if r["efficiency"]  == best_eff:  marks.append("⚡Eff")
        if r["total_inf_ms"] == best_spd: marks.append("🚀Spd")

        print(f"  {r['backbone']:<22} "
              f"{r['params_m']:>5.1f}M "
              f"{r['accuracy']:>7.4f} "
              f"{r['f1_macro']:>7.4f} "
              f"{r['total_inf_ms']:>8.1f}ms "
              f"{r['efficiency']:>8.4f}  "
              f"{' '.join(marks)}")

    print(f"\n  클래스별 F1:")
    print(f"  {'백본':<22}", end="")
    for cls in DISASTER_CLASSES:
        print(f"  {cls[:6]:>8}", end="")
    print()
    print(f"  {'-'*80}")

    for r in sorted_results:
        print(f"  {r['backbone']:<22}", end="")
        for cls in DISASTER_CLASSES:
            print(f"  {r['f1_per_cls'].get(cls, 0):>8.4f}", end="")
        print()

    print(f"\n  효율 점수 = Macro F1 / (Params / 25M)")
    print(f"  높을수록 파라미터 대비 성능이 좋음")
    print(f"{'='*90}\n")


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="백본 실험 자동화 스크립트")
    parser.add_argument(
        "--backbones", nargs="+",
        default=SUPPORTED_BACKBONES,
        choices=SUPPORTED_BACKBONES,
        help="실험할 백본 목록 (기본: 전체 6개)",
    )
    parser.add_argument(
        "--target", default="both",
        choices=["both", "classifier", "detector"],
        help="백본을 교체할 대상 (기본: 둘 다)",
    )
    parser.add_argument(
        "--compare-only", action="store_true",
        help="학습 없이 저장된 결과 비교표만 출력",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="지원되는 백본 목록 출력",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list:
        list_backbones()
        return

    if args.compare_only:
        results = load_results()
        print_comparison(results)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print(f"\n실험 대상 백본 ({len(args.backbones)}개): {args.backbones}")
    print(f"교체 대상: {args.target}")

    all_results = load_results()

    for backbone_name in args.backbones:
        if backbone_name in all_results:
            print(f"\n  ℹ️  {backbone_name} — 이미 실험 완료, 스킵")
            print(f"      (재실험하려면 results/{backbone_name}/ 폴더 삭제 후 재실행)")
            continue

        try:
            result = run_experiment(backbone_name, args.target, device)
            all_results[backbone_name] = result
            save_results(all_results)
            print(f"  ✓ 결과 저장: {RESULTS_FILE}")
        except Exception as e:
            print(f"  ✗ {backbone_name} 실험 실패: {e}")
            import traceback
            traceback.print_exc()

    print("\n\n" + "="*70)
    print("  전체 실험 완료 — 최종 비교표")
    print_comparison(all_results)


if __name__ == "__main__":
    main()
