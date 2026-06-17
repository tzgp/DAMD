from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def repo_root() -> Path:
    return REPO_ROOT


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (REPO_ROOT / candidate).resolve()


def _env_or_default(env_name: str, default_relative: str) -> Path:
    raw = os.environ.get(env_name)
    if raw:
        return Path(raw).expanduser()
    return resolve_repo_path(default_relative)


def default_rgb_checkpoint() -> Path:
    return _env_or_default("DAMD_RGB_BACKBONE_CHECKPOINT", "checkpoints/dino_vitbase8_pretrain.pth")


def default_fusion_checkpoint() -> Path:
    return _env_or_default("DAMD_FUSION_CHECKPOINT", "checkpoints/uff_pretrain.pth")


def default_ssl_checkpoint() -> Path:
    return _env_or_default("DAMD_SSL_CHECKPOINT", "checkpoints/ssl_pretrain.pth")


def default_dataset_root(dataset_type: str = "mvtec3d") -> Path:
    normalized = dataset_type.lower()
    env_name = "DAMD_MVTEC3D_ROOT" if normalized == "mvtec3d" else "DAMD_EYECANDIES_ROOT"
    return _env_or_default(env_name, f"datasets/{normalized}")


def default_feature_cache(dataset_type: str = "mvtec3d") -> Path:
    normalized = dataset_type.lower()
    return _env_or_default("DAMD_FEATURE_CACHE_ROOT", f"outputs/feature_cache/{normalized}")


def default_results_dir() -> Path:
    return _env_or_default("DAMD_RESULTS_DIR", "results")
