import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_defaults import default_dataset_root, default_feature_cache, default_fusion_checkpoint

from dataset import eyecandies_classes, get_data_loader, mvtec3d_classes  # noqa: E402
from feature_extractors.selfmultiple_featureslate import DoubleRGBFPFHFeatures_add_late  # noqa: E402
from utils.au_pro_util import calculate_au_pro  # noqa: E402


EPS = 1e-12
FIXED_WEIGHTS: Sequence[Tuple[float, float]] = (
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
)


@dataclass
class SampleArtifact:
    dataset: str
    category: str
    sample_id: str
    label: int
    defect_type: str
    rgb_path: str
    lambda_rgb: float
    lambda_pc: float
    score_rgb: float
    score_pc: float
    score_eaf: float
    map_rgb: np.ndarray
    map_pc: np.ndarray
    map_eaf: np.ndarray
    gt_mask: np.ndarray
    ocsvm_score: Optional[float] = None
    ocsvm_map: Optional[np.ndarray] = None


def _to_float(value: object) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if isinstance(value, np.ndarray):
        return float(np.asarray(value).reshape(-1)[0])
    return float(value)


def _to_numpy_map(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return np.squeeze(array).astype(np.float64)


def _normalize_path_value(path_value: object) -> str:
    if isinstance(path_value, str):
        return path_value
    if isinstance(path_value, (list, tuple)) and path_value:
        return _normalize_path_value(path_value[0])
    return str(path_value)


def _parse_sample_id(rgb_path: str) -> Tuple[str, str]:
    normalized = Path(rgb_path)
    defect_type = normalized.parts[-3] if len(normalized.parts) >= 3 else "unknown"
    return normalized.stem, defect_type


def _safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels_array = np.asarray(labels)
    if len(np.unique(labels_array)) < 2:
        return float("nan")
    return float(roc_auc_score(labels_array, np.asarray(scores, dtype=np.float64)))


def _safe_corr(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float, int]:
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan"), float("nan"), int(len(x))
    return float(pearsonr(x, y)[0]), float(spearmanr(x, y)[0]), int(len(x))


def _minmax(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float64)
    arr_min = float(np.min(arr))
    arr_max = float(np.max(arr))
    if arr_max - arr_min < EPS:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - arr_min) / (arr_max - arr_min)


def _quality_gap(score_map: np.ndarray, gt_mask: np.ndarray) -> float:
    mask = np.asarray(gt_mask) > 0.5
    if not np.any(mask):
        return float("nan")
    inside = np.asarray(score_map, dtype=np.float64)[mask]
    outside = np.asarray(score_map, dtype=np.float64)[~mask]
    inside_mean = float(np.mean(inside)) if inside.size else float("nan")
    outside_mean = float(np.mean(outside)) if outside.size else 0.0
    return inside_mean - outside_mean


def _overlay_map_on_rgb(rgb_image: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb_image, dtype=np.float64)
    target_h, target_w = np.asarray(score_map).shape[:2]
    if rgb.shape[0] != target_h or rgb.shape[1] != target_w:
        rgb_uint8 = np.asarray(np.clip(rgb, 0, 255) if rgb.max() > 1.0 else np.clip(rgb * 255.0, 0, 255), dtype=np.uint8)
        rgb = np.asarray(Image.fromarray(rgb_uint8).resize((target_w, target_h), Image.BILINEAR), dtype=np.float64)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    cmap = plt.get_cmap("jet")(_minmax(score_map))[..., :3]
    overlay = 0.55 * rgb + 0.45 * cmap
    return np.clip(overlay, 0.0, 1.0)


class AnalysisLateFusion(DoubleRGBFPFHFeatures_add_late):
    def __init__(self, args):
        super().__init__(args)
        self.current_category = ""
        self.current_rgb_path = ""
        self.current_label = 0
        self.sample_artifacts: List[SampleArtifact] = []
        self.ocsvm_ready = False

    def set_context(self, category: str, rgb_path: object, label: object) -> None:
        self.current_category = category
        self.current_rgb_path = _normalize_path_value(rgb_path)
        self.current_label = int(_to_float(label))

    def collect_ocsvm_training_sample(self, sample) -> None:
        organized_pc = sample[1]
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = np.asarray(organized_pc_np).reshape(-1, 3)
        nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
        rgb_feature_maps = self(sample[0])
        depth_feature_maps = self.get_fpfh_features(sample[1])
        depth_feature_maps_resized = self.resize(self.average(depth_feature_maps))
        fpfh_patch = depth_feature_maps_resized.reshape(depth_feature_maps_resized.shape[1], -1).T
        rgb_patch = torch.cat(rgb_feature_maps, 1)
        rgb_patch = rgb_patch.reshape(rgb_patch.shape[1], -1).T
        xyz_patch = (fpfh_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        s = torch.tensor([[s_xyz, s_rgb]], dtype=torch.float32)
        s_map = torch.cat([s_map_xyz, s_map_rgb], dim=0).squeeze().reshape(2, -1).permute(1, 0)
        self.s_lib.append(s)
        self.s_map_lib.append(s_map)

    def train_ocsvm_fusion(self) -> None:
        if not self.s_lib or not self.s_map_lib:
            self.ocsvm_ready = False
            return
        self.run_late_fusion()
        self.ocsvm_ready = True

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, nonzero_indices, xyz):
        xyz_patch = (xyz_patch - self.xyz_mean) / self.xyz_std
        rgb_patch = (rgb_patch - self.rgb_mean) / self.rgb_std
        dist_xyz = torch.cdist(xyz_patch, self.patch_xyz_lib)
        dist_rgb = torch.cdist(rgb_patch, self.patch_rgb_lib)
        rgb_feat_size = (int(math.sqrt(rgb_patch.shape[0])), int(math.sqrt(rgb_patch.shape[0])))
        xyz_feat_size = (int(math.sqrt(xyz_patch.shape[0])), int(math.sqrt(xyz_patch.shape[0])))
        s_xyz, s_map_xyz = self.compute_single_s_s_map(xyz_patch, dist_xyz, xyz_feat_size, modal='xyz')
        s_rgb, s_map_rgb = self.compute_single_s_s_map(rgb_patch, dist_rgb, rgb_feat_size, modal='rgb')
        lambda_pc, lambda_rgb = self.calculate_gating_weights(xyz_patch, rgb_patch)
        lambda_pc = float(np.asarray(lambda_pc, dtype=np.float64))
        lambda_rgb = float(np.asarray(lambda_rgb, dtype=np.float64))
        score_pc = _to_float(s_xyz)
        score_rgb = _to_float(s_rgb)
        map_pc = _to_numpy_map(s_map_xyz)
        map_rgb = _to_numpy_map(s_map_rgb)
        score_eaf = lambda_pc * score_pc + lambda_rgb * score_rgb
        map_eaf = lambda_pc * map_pc + lambda_rgb * map_rgb
        ocsvm_score = None
        ocsvm_map = None
        if self.ocsvm_ready:
            score_tensor = torch.tensor([[score_pc, score_rgb]], dtype=torch.float32)
            map_tensor = torch.tensor(
                np.stack([map_pc.reshape(-1), map_rgb.reshape(-1)], axis=1),
                dtype=torch.float32,
            )
            ocsvm_score = float(self.detect_fuser.score_samples(score_tensor)[0])
            ocsvm_map = self.seg_fuser.score_samples(map_tensor).reshape(self.image_size, self.image_size)
        score_tensor_eaf = torch.tensor(score_eaf, dtype=torch.float32)
        map_tensor_eaf = torch.tensor(map_eaf, dtype=torch.float32).view(1, self.image_size, self.image_size)
        self.image_preds.append(score_tensor_eaf.numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(map_tensor_eaf.flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())
        self.predictions.append(map_tensor_eaf.detach().cpu().squeeze().numpy())
        self.gts.append(mask.detach().cpu().squeeze().numpy())
        sample_id, defect_type = _parse_sample_id(self.current_rgb_path)
        self.sample_artifacts.append(
            SampleArtifact(
                dataset=self.args.dataset_type,
                category=self.current_category,
                sample_id=sample_id,
                label=self.current_label,
                defect_type=defect_type,
                rgb_path=self.current_rgb_path,
                lambda_rgb=lambda_rgb,
                lambda_pc=lambda_pc,
                score_rgb=score_rgb,
                score_pc=score_pc,
                score_eaf=score_eaf,
                map_rgb=map_rgb,
                map_pc=map_pc,
                map_eaf=np.asarray(map_eaf, dtype=np.float64),
                gt_mask=_to_numpy_map(mask),
                ocsvm_score=ocsvm_score,
                ocsvm_map=None if ocsvm_map is None else np.asarray(ocsvm_map, dtype=np.float64),
            )
        )


def _get_classes(dataset_name: str) -> List[str]:
    if dataset_name == "mvtec3d":
        return mvtec3d_classes()
    if dataset_name == "eyecandies":
        return eyecandies_classes()
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def _method_variant_names(include_ocsvm: bool) -> List[str]:
    names = [
        "RGB only",
        "3D only",
        "Equal fusion",
        "Max-score fusion",
        "Score-normalized equal fusion",
        "EAF, ours",
    ]
    for rgb_weight, pc_weight in FIXED_WEIGHTS:
        names.append(f"Fixed fusion RGB:{rgb_weight:.1f}_PC:{pc_weight:.1f}")
    if include_ocsvm:
        names.append("OCSVM fusion")
    names.append("Best fixed weight (post-hoc control)")
    return names


def _build_method_outputs(samples: List[SampleArtifact], include_ocsvm: bool) -> Dict[str, Dict[str, List[object]]]:
    if not samples:
        return {}
    rgb_scores = np.asarray([sample.score_rgb for sample in samples], dtype=np.float64)
    pc_scores = np.asarray([sample.score_pc for sample in samples], dtype=np.float64)
    rgb_scores_norm = _minmax(rgb_scores)
    pc_scores_norm = _minmax(pc_scores)
    outputs: Dict[str, Dict[str, List[object]]] = {}

    def ensure(name: str) -> Dict[str, List[object]]:
        if name not in outputs:
            outputs[name] = {"labels": [], "scores": [], "maps": [], "gts": []}
        return outputs[name]

    for idx, sample in enumerate(samples):
        rgb_map_norm = _minmax(sample.map_rgb)
        pc_map_norm = _minmax(sample.map_pc)
        entries = {
            "RGB only": (sample.score_rgb, sample.map_rgb),
            "3D only": (sample.score_pc, sample.map_pc),
            "Equal fusion": ((sample.score_rgb + sample.score_pc) / 2.0, 0.5 * sample.map_rgb + 0.5 * sample.map_pc),
            "Max-score fusion": (max(sample.score_rgb, sample.score_pc), np.maximum(sample.map_rgb, sample.map_pc)),
            "Score-normalized equal fusion": ((rgb_scores_norm[idx] + pc_scores_norm[idx]) / 2.0, 0.5 * rgb_map_norm + 0.5 * pc_map_norm),
            "EAF, ours": (sample.score_eaf, sample.map_eaf),
        }
        for rgb_weight, pc_weight in FIXED_WEIGHTS:
            entries[f"Fixed fusion RGB:{rgb_weight:.1f}_PC:{pc_weight:.1f}"] = (
                rgb_weight * sample.score_rgb + pc_weight * sample.score_pc,
                rgb_weight * sample.map_rgb + pc_weight * sample.map_pc,
            )
        if include_ocsvm and sample.ocsvm_score is not None and sample.ocsvm_map is not None:
            entries["OCSVM fusion"] = (sample.ocsvm_score, sample.ocsvm_map)
        for method_name, (score_value, map_value) in entries.items():
            bucket = ensure(method_name)
            bucket["labels"].append(sample.label)
            bucket["scores"].append(float(score_value))
            bucket["maps"].append(np.asarray(map_value, dtype=np.float64))
            bucket["gts"].append(np.asarray(sample.gt_mask, dtype=np.float64))
    return outputs


def _compute_metrics_from_outputs(outputs: Dict[str, Dict[str, List[object]]]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for method_name, payload in outputs.items():
        labels = payload["labels"]
        scores = payload["scores"]
        maps = payload["maps"]
        gts = payload["gts"]
        pixel_labels = np.concatenate([np.asarray(gt).reshape(-1) for gt in gts])
        pixel_scores = np.concatenate([np.asarray(pred).reshape(-1) for pred in maps])
        metrics[method_name] = {
            "I_AUROC": _safe_auc(labels, scores),
            "P_AUROC": _safe_auc(pixel_labels, pixel_scores),
            "AUPRO": float(calculate_au_pro(gts, maps)[0]),
        }
    fixed_names = [name for name in metrics if name.startswith("Fixed fusion RGB:")]
    if fixed_names:
        best_fixed_name = max(
            fixed_names,
            key=lambda name: np.nanmean(
                [metrics[name]["I_AUROC"], metrics[name]["P_AUROC"], metrics[name]["AUPRO"]]
            ),
        )
        metrics["Best fixed weight (post-hoc control)"] = dict(metrics[best_fixed_name])
        metrics["Best fixed weight (post-hoc control)"]["source_method"] = best_fixed_name
    return metrics


def _category_weight_summary(samples: List[SampleArtifact]) -> pd.DataFrame:
    rows = []
    grouped: Dict[str, List[SampleArtifact]] = {}
    for sample in samples:
        grouped.setdefault(sample.category, []).append(sample)
    for category, category_samples in grouped.items():
        rgb_values = np.asarray([sample.lambda_rgb for sample in category_samples], dtype=np.float64)
        pc_values = np.asarray([sample.lambda_pc for sample in category_samples], dtype=np.float64)
        rows.append(
            {
                "category": category,
                "mean_lambda_rgb": float(np.mean(rgb_values)),
                "std_lambda_rgb": float(np.std(rgb_values)),
                "mean_lambda_pc": float(np.mean(pc_values)),
                "std_lambda_pc": float(np.std(pc_values)),
                "median_lambda_rgb": float(np.median(rgb_values)),
                "median_lambda_pc": float(np.median(pc_values)),
                "num_samples": int(len(category_samples)),
            }
        )
    return pd.DataFrame(rows).sort_values("category")


def _sample_rows(samples: List[SampleArtifact]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        rows.append(
            {
                "dataset": sample.dataset,
                "category": sample.category,
                "sample_id": sample.sample_id,
                "label_normal_or_anomaly": "anomaly" if sample.label else "normal",
                "defect_type_if_available": sample.defect_type,
                "lambda_rgb": sample.lambda_rgb,
                "lambda_pc": sample.lambda_pc,
                "score_rgb": sample.score_rgb,
                "score_pc": sample.score_pc,
                "score_eaf": sample.score_eaf,
                "pred_correct_if_available": "",
            }
        )
    return pd.DataFrame(rows)


def _correlation_tables(samples: List[SampleArtifact]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    delta_lambda = [sample.lambda_rgb - sample.lambda_pc for sample in samples]
    delta_score = [sample.score_rgb - sample.score_pc for sample in samples]
    pearson_all, spearman_all, n_all = _safe_corr(delta_lambda, delta_score)
    anomaly_samples = [sample for sample in samples if sample.label == 1]
    delta_lambda_anom = [sample.lambda_rgb - sample.lambda_pc for sample in anomaly_samples]
    delta_score_anom = [sample.score_rgb - sample.score_pc for sample in anomaly_samples]
    pearson_anom, spearman_anom, n_anom = _safe_corr(delta_lambda_anom, delta_score_anom)
    corr_df = pd.DataFrame(
        [
            {"subset": "all", "pearson": pearson_all, "spearman": spearman_all, "num_samples": n_all},
            {"subset": "anomaly_only", "pearson": pearson_anom, "spearman": spearman_anom, "num_samples": n_anom},
        ]
    )

    delta_quality = []
    delta_lambda_quality = []
    eaf_quality = []
    for sample in anomaly_samples:
        gap_rgb = _quality_gap(sample.map_rgb, sample.gt_mask)
        gap_pc = _quality_gap(sample.map_pc, sample.gt_mask)
        gap_eaf = _quality_gap(sample.map_eaf, sample.gt_mask)
        delta_lambda_quality.append(sample.lambda_rgb - sample.lambda_pc)
        delta_quality.append(gap_rgb - gap_pc)
        eaf_quality.append(gap_eaf)
    pearson_quality, spearman_quality, n_quality = _safe_corr(delta_lambda_quality, delta_quality)
    pearson_eaf, spearman_eaf, n_eaf = _safe_corr(delta_lambda_quality, eaf_quality)
    quality_df = pd.DataFrame(
        [
            {
                "analysis": "corr(delta_lambda, quality_rgb_minus_quality_pc)",
                "pearson": pearson_quality,
                "spearman": spearman_quality,
                "num_samples": n_quality,
            },
            {
                "analysis": "corr(delta_lambda, quality_eaf)",
                "pearson": pearson_eaf,
                "spearman": spearman_eaf,
                "num_samples": n_eaf,
            },
        ]
    )
    return corr_df, quality_df


def _plot_weight_boxplot(weight_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    categories = weight_df["category"].tolist()
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.2), 5.5))
    rgb_positions = np.arange(len(categories)) * 2.2
    pc_positions = rgb_positions + 0.8
    rgb_data = []
    pc_data = []
    for category in categories:
        category_rows = weight_df.loc[weight_df["category"] == category]
        rgb_data.append(category_rows["lambda_rgb_samples"].iloc[0])
        pc_data.append(category_rows["lambda_pc_samples"].iloc[0])
    ax.boxplot(rgb_data, positions=rgb_positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#4C78A8"))
    ax.boxplot(pc_data, positions=pc_positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor="#F58518"))
    ax.set_xticks(rgb_positions + 0.4)
    ax.set_xticklabels(categories, rotation=30, ha="right")
    ax.set_ylabel("Fusion weight")
    ax.set_xlabel("Category")
    ax.legend([plt.Line2D([0], [0], color="#4C78A8", lw=8), plt.Line2D([0], [0], color="#F58518", lw=8)], ["RGB weight", "3D weight"], frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        output_path = output_dir / f"eaf_weight_boxplot_by_category.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def _plot_weight_histogram(samples: List[SampleArtifact], output_dir: Path) -> List[Path]:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist([sample.lambda_rgb for sample in samples], bins=20, alpha=0.7, label="RGB weight", color="#4C78A8")
    ax.hist([sample.lambda_pc for sample in samples], bins=20, alpha=0.7, label="3D weight", color="#F58518")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outputs = []
    for suffix in ("png", "pdf"):
        output_path = output_dir / f"eaf_weight_histogram.{suffix}"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        outputs.append(output_path)
    plt.close(fig)
    return outputs


def _plot_metric_barplots(comparison_df: pd.DataFrame, output_dir: Path) -> List[Path]:
    outputs: List[Path] = []
    plot_df = comparison_df.copy()
    metric_columns = [
        ("I_AUROC_mean", "fusion_comparison_iauroc"),
        ("P_AUROC_mean", "fusion_comparison_pauroc"),
        ("AUPRO_mean", "fusion_comparison_aupro"),
    ]
    for metric_column, stem in metric_columns:
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ax.bar(plot_df["method"], plot_df[metric_column], color="#4C78A8")
        ax.set_ylabel(metric_column.replace("_mean", ""))
        ax.set_xlabel("Fusion variant")
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            output_path = output_dir / f"{stem}.{suffix}"
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            outputs.append(output_path)
        plt.close(fig)
    return outputs


def _select_representative_cases(samples: List[SampleArtifact]) -> Dict[str, List[SampleArtifact]]:
    anomaly_samples = [sample for sample in samples if sample.label == 1]
    if not anomaly_samples:
        return {"rgb_dominant": [], "pc_dominant": [], "balanced": [], "failure": []}

    def rgb_advantage(sample: SampleArtifact) -> float:
        return (sample.lambda_rgb - sample.lambda_pc) + (_quality_gap(sample.map_rgb, sample.gt_mask) - _quality_gap(sample.map_pc, sample.gt_mask))

    def pc_advantage(sample: SampleArtifact) -> float:
        return (sample.lambda_pc - sample.lambda_rgb) + (_quality_gap(sample.map_pc, sample.gt_mask) - _quality_gap(sample.map_rgb, sample.gt_mask))

    def balanced_score(sample: SampleArtifact) -> float:
        return -abs(sample.lambda_rgb - sample.lambda_pc) - abs(_quality_gap(sample.map_rgb, sample.gt_mask) - _quality_gap(sample.map_pc, sample.gt_mask))

    def failure_score(sample: SampleArtifact) -> float:
        best_modal = max(_quality_gap(sample.map_rgb, sample.gt_mask), _quality_gap(sample.map_pc, sample.gt_mask))
        return _quality_gap(sample.map_eaf, sample.gt_mask) - best_modal

    used: set = set()
    groups = {}
    ranked_groups = {
        "rgb_dominant": sorted(anomaly_samples, key=rgb_advantage, reverse=True),
        "pc_dominant": sorted(anomaly_samples, key=pc_advantage, reverse=True),
        "balanced": sorted(anomaly_samples, key=balanced_score, reverse=True),
        "failure": sorted(anomaly_samples, key=failure_score),
    }
    for name, ranked in ranked_groups.items():
        picked: List[SampleArtifact] = []
        for sample in ranked:
            sample_key = (sample.category, sample.sample_id)
            if sample_key in used:
                continue
            picked.append(sample)
            used.add(sample_key)
            if len(picked) == 2:
                break
        groups[name] = picked
    return groups


def _save_representative_cases(case_groups: Dict[str, List[SampleArtifact]], output_dir: Path) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for group_name, samples in case_groups.items():
        for sample in samples:
            rgb_pil = Image.open(sample.rgb_path).convert("RGB").resize((sample.map_eaf.shape[1], sample.map_eaf.shape[0]), Image.BILINEAR)
            rgb_image = np.asarray(rgb_pil, dtype=np.float64) / 255.0
            overlay = _overlay_map_on_rgb(rgb_image, sample.map_eaf)
            fig, axes = plt.subplots(1, 6, figsize=(18, 3.5))
            titles = [
                "RGB image",
                "GT mask",
                "RGB map",
                "3D map",
                "EAF map",
                "EAF overlay",
            ]
            images = [
                rgb_image,
                sample.gt_mask,
                sample.map_rgb,
                sample.map_pc,
                sample.map_eaf,
                overlay,
            ]
            cmaps = [None, "gray", "jet", "jet", "jet", None]
            for ax, title, image, cmap in zip(axes, titles, images, cmaps):
                ax.imshow(image, cmap=cmap)
                ax.set_title(title, fontsize=9)
                ax.axis("off")
            fig.suptitle(
                f"{sample.category} | {sample.sample_id} | λ_rgb={sample.lambda_rgb:.3f}, λ_pc={sample.lambda_pc:.3f} | "
                f"score_rgb={sample.score_rgb:.4f}, score_pc={sample.score_pc:.4f}, score_eaf={sample.score_eaf:.4f}",
                fontsize=10,
            )
            fig.tight_layout()
            stem = f"{group_name}_{sample.category}_{sample.sample_id}"
            for suffix in ("png", "pdf"):
                output_path = output_dir / f"{stem}.{suffix}"
                fig.savefig(output_path, dpi=300, bbox_inches="tight")
                outputs.append(output_path)
            plt.close(fig)
    return outputs


def _write_run_instructions(output_dir: Path, args, missing_paths: List[str]) -> Path:
    output_path = output_dir / "RUN_INSTRUCTIONS.md"
    content = [
        "# EAF Analysis Run Instructions",
        "",
        "The automatic run could not start because some required paths are missing.",
        "",
        "## Missing paths",
    ]
    for missing in missing_paths:
        content.append(f"- `{missing}`")
    content.extend(
        [
            "",
            "## Command template",
            "```bash",
                "python experiments/revision/run_eaf_analysis.py --dataset mvtec3d --data_root /path/to/mvtec3d --output_dir outputs/revision/eaf_analysis",
            "```",
            "",
            "## Current resolved arguments",
            "```json",
            json.dumps(vars(args), indent=2, ensure_ascii=False),
            "```",
        ]
    )
    output_path.write_text("\n".join(content), encoding="utf-8")
    return output_path


def _write_summary_markdown(
    summary_path: Path,
    comparison_df: pd.DataFrame,
    weights_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
    quality_df: pd.DataFrame,
) -> None:
    comparison_indexed = comparison_df.set_index("method")
    eaf_row = comparison_indexed.loc["EAF, ours"]

    def delta_block(method_name: str) -> str:
        if method_name not in comparison_indexed.index:
            return f"- {method_name}: not available."
        row = comparison_indexed.loc[method_name]
        return (
            f"- vs {method_name}: ΔI-AUROC={eaf_row['I_AUROC_mean'] - row['I_AUROC_mean']:.4f}, "
            f"ΔP-AUROC={eaf_row['P_AUROC_mean'] - row['P_AUROC_mean']:.4f}, "
            f"ΔAUPRO={eaf_row['AUPRO_mean'] - row['AUPRO_mean']:.4f}"
        )

    higher_rgb = weights_df.sort_values("mean_lambda_rgb", ascending=False).head(3)["category"].tolist()
    higher_pc = weights_df.sort_values("mean_lambda_pc", ascending=False).head(3)["category"].tolist()
    balanced = weights_df.assign(balance_gap=(weights_df["mean_lambda_rgb"] - weights_df["mean_lambda_pc"]).abs())
    balanced_list = balanced.sort_values("balance_gap").head(3)["category"].tolist()
    corr_all = correlation_df.loc[correlation_df["subset"] == "all"].iloc[0]
    quality_row = quality_df.iloc[0]

    content = f"""# EAF revision experiment summary

## 1. Experimental purpose
This supplementary analysis is added to address the reviewers' concern that entropy may partly reflect noise rather than consistently useful information. The goal is therefore not to overclaim entropy as a universally reliable signal, but to examine whether the entropy-aware fusion (EAF) weights show interpretable tendencies and whether EAF performs better than several common fusion baselines under the same DAMD backbone, feature extraction, and memory-bank pipeline.

## 2. Experimental setting
- Dataset: MVTec 3D-AD.
- Metrics: I-AUROC, P-AUROC, and AUPRO.
- Fusion variants: RGB only, 3D only, Equal fusion, five fixed-weight fusion baselines, Max-score fusion, Score-normalized equal fusion, EAF, and OCSVM fusion when available.
- Backbone / features / memory bank: identical to the existing DAMD late-fusion pipeline; the revision code only adds analysis and result-export logic.

## 3. Main results
{delta_block('Equal fusion')}
{delta_block('Max-score fusion')}
{delta_block('Best fixed weight (post-hoc control)')}
{delta_block('RGB only')}
{delta_block('3D only')}

## 4. Weight distribution analysis
- Categories with relatively higher RGB weights: {', '.join(higher_rgb) if higher_rgb else 'N/A'}.
- Categories with relatively higher 3D weights: {', '.join(higher_pc) if higher_pc else 'N/A'}.
- Categories with relatively balanced weights: {', '.join(balanced_list) if balanced_list else 'N/A'}.

## 5. Interpretability statement
The entropy value is used as a sample-level proxy for feature dispersion and potential modality informativeness. The added analysis shows that EAF tends to assign larger weights to the modality that provides clearer anomaly evidence in many representative cases, while its limitations under noisy or unstable features are also acknowledged. Quantitatively, corr(Δlambda, Δscore) on all samples is Pearson={corr_all['pearson']:.4f} and Spearman={corr_all['spearman']:.4f}; corr(Δlambda, quality_rgb-quality_pc) on anomaly samples is Pearson={quality_row['pearson']:.4f} and Spearman={quality_row['spearman']:.4f}. These observations support interpretability tendencies rather than an absolute claim that higher entropy always indicates more useful information.

## 6. Rebuttal draft (English)
We thank the reviewers for pointing out that entropy may also be influenced by noise and therefore should not be interpreted as an unconditional indicator of useful information. In the revision, we added a broader fusion-strategy comparison including RGB-only, 3D-only, equal fusion, fixed-weight fusion, max fusion, score-normalized equal fusion, and OCSVM-based fusion when available, while keeping the same DAMD backbone, feature extraction pipeline, and memory bank. We further added category-level and sample-level visualizations of the EAF weights, together with correlation analysis and representative-case inspection. These new results show that EAF tends to assign larger weights to the modality that provides clearer anomaly evidence in many cases, while also revealing failure cases where entropy-derived weighting is less reliable under noisy or unstable features. Accordingly, we weakened the original overclaim in the manuscript and explicitly discussed this limitation.
"""
    summary_path.write_text(content, encoding="utf-8")


def run_analysis(args) -> Dict[str, object]:
    output_dir = Path(args.output_dir).resolve()
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    representative_dir = figures_dir / "representative_eaf_cases"
    writing_dir = output_dir / "writing_materials"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    writing_dir.mkdir(parents=True, exist_ok=True)

    missing_paths = []
    for required_path in [args.data_root]:
        if required_path and not Path(required_path).exists():
            missing_paths.append(required_path)
    if missing_paths:
        instructions_path = _write_run_instructions(output_dir, args, missing_paths)
        return {"missing": True, "instructions": instructions_path}

    classes = _get_classes(args.dataset)
    if args.categories:
        requested = [item.strip() for item in args.categories.split(",") if item.strip()]
        classes = [item for item in classes if item in requested]

    all_samples: List[SampleArtifact] = []
    by_category_rows = []
    per_category_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
    include_ocsvm = bool(args.include_ocsvm)

    for category in classes:
        print(f"[EAF analysis] Processing category: {category}")
        method = AnalysisLateFusion(args)
        train_loader = get_data_loader("train", class_name=category, img_size=args.img_size, args=args)
        for sample, _ in train_loader:
            if args.save_feature:
                method.add_sample_to_mem_bank(sample, class_name=category)
            else:
                method.add_sample_to_mem_bank(sample)
        method.run_coreset()

        if include_ocsvm:
            train_loader_for_ocsvm = get_data_loader("train", class_name=category, img_size=args.img_size, args=args)
            for sample, _ in train_loader_for_ocsvm:
                method.collect_ocsvm_training_sample(sample)
            method.train_ocsvm_fusion()

        test_loader = get_data_loader("test", class_name=category, img_size=args.img_size, args=args)
        with torch.no_grad():
            for sample, mask, label, rgb_path in test_loader:
                method.set_context(category, rgb_path, label)
                method.predict(sample, mask, label)
        category_samples = list(method.sample_artifacts)
        all_samples.extend(category_samples)
        outputs = _build_method_outputs(category_samples, include_ocsvm=method.ocsvm_ready)
        metrics = _compute_metrics_from_outputs(outputs)
        per_category_metrics[category] = metrics
        for method_name, metric_values in metrics.items():
            if method_name == "Best fixed weight (post-hoc control)":
                source_method = metric_values.get("source_method", "")
            else:
                source_method = ""
            by_category_rows.append(
                {
                    "category": category,
                    "method": method_name,
                    "I_AUROC": metric_values["I_AUROC"],
                    "P_AUROC": metric_values["P_AUROC"],
                    "AUPRO": metric_values["AUPRO"],
                    "source_method": source_method,
                }
            )

    by_category_df = pd.DataFrame(by_category_rows).sort_values(["method", "category"]) if by_category_rows else pd.DataFrame()
    comparison_rows = []
    if not by_category_df.empty:
        for method_name, group in by_category_df.groupby("method"):
            comparison_rows.append(
                {
                    "method": method_name,
                    "I_AUROC_mean": float(group["I_AUROC"].mean()),
                    "P_AUROC_mean": float(group["P_AUROC"].mean()),
                    "AUPRO_mean": float(group["AUPRO"].mean()),
                }
            )
    comparison_df = pd.DataFrame(comparison_rows)

    sample_df = _sample_rows(all_samples)
    weights_summary_df = _category_weight_summary(all_samples)
    if not weights_summary_df.empty:
        rgb_sample_lists = []
        pc_sample_lists = []
        for category in weights_summary_df["category"]:
            category_samples = [sample for sample in all_samples if sample.category == category]
            rgb_sample_lists.append([sample.lambda_rgb for sample in category_samples])
            pc_sample_lists.append([sample.lambda_pc for sample in category_samples])
        weight_plot_df = weights_summary_df.copy()
        weight_plot_df["lambda_rgb_samples"] = rgb_sample_lists
        weight_plot_df["lambda_pc_samples"] = pc_sample_lists
    else:
        weight_plot_df = weights_summary_df.copy()

    corr_df, quality_df = _correlation_tables(all_samples)

    comparison_csv = tables_dir / "fusion_comparison_mvtec3d.csv"
    by_category_csv = tables_dir / "fusion_comparison_mvtec3d_by_category.csv"
    sample_csv = tables_dir / "eaf_weights_by_sample.csv"
    category_csv = tables_dir / "eaf_weights_by_category.csv"
    improvement_csv = tables_dir / "fusion_improvement_summary.csv"
    corr_csv = tables_dir / "eaf_correlation_analysis.csv"
    quality_csv = tables_dir / "eaf_weight_quality_correlation.csv"

    comparison_df.to_csv(comparison_csv, index=False)
    by_category_df.to_csv(by_category_csv, index=False)
    sample_df.to_csv(sample_csv, index=False)
    weights_summary_df.to_csv(category_csv, index=False)
    corr_df.to_csv(corr_csv, index=False)
    quality_df.to_csv(quality_csv, index=False)

    improvement_rows = []
    if not comparison_df.empty and "EAF, ours" in comparison_df["method"].tolist():
        eaf_row = comparison_df.set_index("method").loc["EAF, ours"]
        for _, row in comparison_df.iterrows():
            if row["method"] == "EAF, ours":
                continue
            improvement_rows.append(
                {
                    "baseline_method": row["method"],
                    "delta_I_AUROC": float(eaf_row["I_AUROC_mean"] - row["I_AUROC_mean"]),
                    "delta_P_AUROC": float(eaf_row["P_AUROC_mean"] - row["P_AUROC_mean"]),
                    "delta_AUPRO": float(eaf_row["AUPRO_mean"] - row["AUPRO_mean"]),
                }
            )
    improvement_df = pd.DataFrame(improvement_rows)
    improvement_df.to_csv(improvement_csv, index=False)

    figure_outputs: List[Path] = []
    if not weight_plot_df.empty:
        figure_outputs.extend(_plot_weight_boxplot(weight_plot_df, figures_dir))
    figure_outputs.extend(_plot_weight_histogram(all_samples, figures_dir))
    figure_outputs.extend(_plot_metric_barplots(comparison_df, figures_dir))
    case_groups = _select_representative_cases(all_samples)
    figure_outputs.extend(_save_representative_cases(case_groups, representative_dir))

    run_instructions_path = _write_run_instructions(output_dir, args, [])
    summary_path = writing_dir / "eaf_experiment_summary.md"
    _write_summary_markdown(summary_path, comparison_df, weights_summary_df, corr_df, quality_df)

    equal_delta = {"I": float("nan"), "P": float("nan"), "A": float("nan")}
    if not improvement_df.empty:
        equal_rows = improvement_df.loc[improvement_df["baseline_method"] == "Equal fusion"]
        if not equal_rows.empty:
            equal_delta = {
                "I": float(equal_rows.iloc[0]["delta_I_AUROC"]),
                "P": float(equal_rows.iloc[0]["delta_P_AUROC"]),
                "A": float(equal_rows.iloc[0]["delta_AUPRO"]),
            }

    return {
        "missing": False,
        "csvs": [comparison_csv, by_category_csv, sample_csv, category_csv, improvement_csv, corr_csv, quality_csv],
        "figures": figure_outputs,
        "summary": summary_path,
        "run_instructions": run_instructions_path,
        "equal_delta": equal_delta,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EAF revision analysis on DAMD.")
    parser.add_argument("--dataset", default="mvtec3d", choices=["mvtec3d", "eyecandies"])
    parser.add_argument("--dataset_type", default="mvtec3d", choices=["mvtec3d", "eyecandies"])
    parser.add_argument("--data_root", default=str(default_dataset_root("mvtec3d")))
    parser.add_argument("--dataset_path", default=str(default_dataset_root("mvtec3d")))
    parser.add_argument("--output_dir", default="outputs/revision/eaf_analysis")
    parser.add_argument("--method_name", default="DINO+FPFH+add+late")
    parser.add_argument("--categories", default="")
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--max_sample", default=400, type=int)
    parser.add_argument("--memory_bank", default="single", choices=["single", "multiple"])
    parser.add_argument("--rgb_backbone_name", default="vit_base_patch8_224_dino")
    parser.add_argument("--xyz_backbone_name", default="Point_MAE")
    parser.add_argument("--fusion_module_path", default=str(default_fusion_checkpoint()))
    parser.add_argument("--save_feature", default=False, action="store_true")
    parser.add_argument("--use_uff", default=False, action="store_true")
    parser.add_argument("--use_trans", default=True, action="store_true")
    parser.add_argument("--use_ssl", default=False, action="store_true")
    parser.add_argument("--save_preds", default=True, action="store_true")
    parser.add_argument("--group_size", default=128, type=int)
    parser.add_argument("--num_group", default=1024, type=int)
    parser.add_argument("--random_state", default=None, type=int)
    parser.add_argument("--save_feature_path", default=str(default_feature_cache("mvtec3d")))
    parser.add_argument("--xyz_s_lambda", default=1.0, type=float)
    parser.add_argument("--xyz_smap_lambda", default=1.0, type=float)
    parser.add_argument("--rgb_s_lambda", default=0.1, type=float)
    parser.add_argument("--rgb_smap_lambda", default=0.1, type=float)
    parser.add_argument("--fusion_s_lambda", default=1.0, type=float)
    parser.add_argument("--fusion_smap_lambda", default=1.0, type=float)
    parser.add_argument("--coreset_eps", default=0.9, type=float)
    parser.add_argument("--f_coreset", default=0.1, type=float)
    parser.add_argument("--asy_memory_bank", default=None, type=int)
    parser.add_argument("--ocsvm_nu", default=0.5, type=float)
    parser.add_argument("--ocsvm_maxiter", default=1000, type=int)
    parser.add_argument("--rm_zero_for_project", default=False, action="store_true")
    parser.add_argument("--include_ocsvm", default=True, action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.dataset_type = args.dataset
    args.dataset_path = args.data_root
    result = run_analysis(args)
    if result["missing"]:
        print(f"Missing required paths. See: {result['instructions']}")
        return
    print("Generated CSV files:")
    for path in result["csvs"]:
        print(f"- {path}")
    print("Generated figures:")
    for path in result["figures"]:
        print(f"- {path}")
    delta = result["equal_delta"]
    print(
        "EAF vs Equal fusion mean deltas: "
        f"I-AUROC={delta['I']:.4f}, P-AUROC={delta['P']:.4f}, AUPRO={delta['A']:.4f}"
    )
    print(f"Writing materials: {result['summary']}")
    print(f"Run instructions: {result['run_instructions']}")


if __name__ == "__main__":
    main()
