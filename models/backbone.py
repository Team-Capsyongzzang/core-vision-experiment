"""
models/backbone.py
==================
6개 백본 팩토리.

지원 백본:
    resnet50        25M   76.1%   현재 베이스라인
    resnet101       44M   77.4%   성능 상한선
    efficientnet_b0  5M   77.7%   경량 고성능
    efficientnet_b3 12M   82.0%   중간 균형점
    mobilenet_v3_large 5M 75.2%  경량화 대표
    mobilenet_v3_small 3M 67.7%  최경량 (1단계 스크리닝 후보)

사용법:
    backbone, feat_dim = build_backbone("efficientnet_b0", pretrained=True)
    # backbone: nn.Module — GlobalAvgPool까지 포함, 출력 (B, feat_dim, 1, 1)
    # feat_dim: int       — head FC 입력 차원

백본별 출력 feature 차원:
    resnet50/101         → 2048
    efficientnet_b0      → 1280
    efficientnet_b3      → 1536
    mobilenet_v3_large   → 960
    mobilenet_v3_small   → 576
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm


# ─────────────────────────────────────────────────────────
#  백본 메타 정보
# ─────────────────────────────────────────────────────────

BACKBONE_INFO = {
    "resnet50": {
        "params_m":   25.0,
        "imagenet_acc": 76.1,
        "feat_dim":   2048,
        "description": "베이스라인 — 성능/크기 균형",
    },
    "resnet101": {
        "params_m":   44.0,
        "imagenet_acc": 77.4,
        "feat_dim":   2048,
        "description": "성능 상한선 확인용",
    },
    "efficientnet_b0": {
        "params_m":   5.3,
        "imagenet_acc": 77.7,
        "feat_dim":   1280,
        "description": "경량 고성능 — ResNet50보다 가볍고 비슷한 성능",
    },
    "efficientnet_b3": {
        "params_m":   12.0,
        "imagenet_acc": 82.0,
        "feat_dim":   1536,
        "description": "중간 균형점 — 성능/크기 최적",
    },
    "mobilenet_v3_large": {
        "params_m":   5.4,
        "imagenet_acc": 75.2,
        "feat_dim":   960,
        "description": "경량화 대표",
    },
    "mobilenet_v3_small": {
        "params_m":   2.5,
        "imagenet_acc": 67.7,
        "feat_dim":   576,
        "description": "최경량 — 1단계 CPU 스크리닝 후보",
    },
}

SUPPORTED_BACKBONES = list(BACKBONE_INFO.keys())


# ─────────────────────────────────────────────────────────
#  개별 백본 로더
# ─────────────────────────────────────────────────────────

def _load_resnet50(pretrained: bool) -> nn.Module:
    weights = tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tvm.resnet50(weights=weights)
    # 마지막 FC 제거 → GlobalAvgPool까지 (출력: B, 2048, 1, 1)
    return nn.Sequential(*list(model.children())[:-1])


def _load_resnet101(pretrained: bool) -> nn.Module:
    weights = tvm.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tvm.resnet101(weights=weights)
    return nn.Sequential(*list(model.children())[:-1])


def _load_efficientnet_b0(pretrained: bool) -> nn.Module:
    weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tvm.efficientnet_b0(weights=weights)
    # EfficientNet: features + avgpool (classifier 제거)
    # model.features → (B, 1280, 7, 7)
    # model.avgpool  → (B, 1280, 1, 1)
    return nn.Sequential(model.features, model.avgpool)


def _load_efficientnet_b3(pretrained: bool) -> nn.Module:
    weights = tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tvm.efficientnet_b3(weights=weights)
    return nn.Sequential(model.features, model.avgpool)


def _load_mobilenet_v3_large(pretrained: bool) -> nn.Module:
    weights = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tvm.mobilenet_v3_large(weights=weights)
    # MobileNetV3: features + avgpool (classifier 제거)
    # avgpool → AdaptiveAvgPool2d(1) → (B, 960, 1, 1)
    return nn.Sequential(model.features, model.avgpool)


def _load_mobilenet_v3_small(pretrained: bool) -> nn.Module:
    weights = tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model   = tvm.mobilenet_v3_small(weights=weights)
    return nn.Sequential(model.features, model.avgpool)


_LOADERS = {
    "resnet50":           _load_resnet50,
    "resnet101":          _load_resnet101,
    "efficientnet_b0":    _load_efficientnet_b0,
    "efficientnet_b3":    _load_efficientnet_b3,
    "mobilenet_v3_large": _load_mobilenet_v3_large,
    "mobilenet_v3_small": _load_mobilenet_v3_small,
}


# ─────────────────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────────────────

def build_backbone(
    name:       str,
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """
    백본을 생성하고 feature 차원을 반환합니다.

    Parameters
    ----------
    name       : SUPPORTED_BACKBONES 중 하나
    pretrained : ImageNet pretrained 가중치 사용 여부

    Returns
    -------
    (backbone, feat_dim)
        backbone : nn.Module — 출력 (B, feat_dim, 1, 1)
        feat_dim : int       — head FC 입력 차원
    """
    if name not in _LOADERS:
        raise ValueError(
            f"Unknown backbone: '{name}'\n"
            f"Available: {SUPPORTED_BACKBONES}"
        )

    backbone = _LOADERS[name](pretrained)
    feat_dim = BACKBONE_INFO[name]["feat_dim"]
    return backbone, feat_dim


def print_backbone_info(name: str):
    """백본 메타 정보를 출력합니다."""
    info = BACKBONE_INFO[name]
    print(f"  Backbone        : {name}")
    print(f"  Params          : ~{info['params_m']}M")
    print(f"  ImageNet Top-1  : {info['imagenet_acc']}%")
    print(f"  Feature dim     : {info['feat_dim']}")
    print(f"  Description     : {info['description']}")


def list_backbones():
    """지원되는 모든 백본 정보를 출력합니다."""
    print(f"\n{'='*65}")
    print(f"  {'백본':<22} {'Params':>8} {'ImgNet':>8} {'Feat dim':>10}  설명")
    print(f"  {'-'*60}")
    for name, info in BACKBONE_INFO.items():
        print(f"  {name:<22} {info['params_m']:>6.1f}M "
              f"{info['imagenet_acc']:>7.1f}% "
              f"{info['feat_dim']:>9}  "
              f"{info['description']}")
    print(f"{'='*65}\n")
