
#!/usr/bin/env python3
"""
Robustness Experiment Runner for DAMD (Revision - Reviewer 1)
=============================================================
Experiments:
  2.1 Point cloud random sparsification
  2.2 Gaussian noise perturbation  
  2.3 Local missing / block dropout
  2.4 MDFE variants (FPFH only, raw density, normalized density)

Design principle: Training memory bank is built ONCE on clean data.
All perturbations are applied ONLY to test point clouds.
Original code in the codebase is NOT modified - all logic is in this script.

Author: Auto-generated for revision experiments
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

# ─── Path setup ───────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_defaults import default_dataset_root, default_feature_cache, default_fusion_checkpoint

from dataset import get_data_loader, mvtec3d_classes, eyecandies_classes
from feature_extractors.selfmultiple_featureslate import DoubleRGBFPFHFeatures_add_late
from utils.au_pro_util import calculate_au_pro
from utils.mvtec3d_util import organized_pc_to_unorganized_pc

# ══════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════

# Sparsification ratios
SPARSIFY_RATIOS = [1.0, 0.75, 0.50, 0.25]
SPARSIFY_SEEDS = [42, 123, 999]

# Gaussian noise sigma ratios (relative to bounding box diagonal D)
NOISE_SIGMA_RATIOS = [0.0, 0.001, 0.003, 0.005, 0.010]
NOISE_SEEDS = [42, 123, 999]

# Missing region ratios
MISSING_RATIOS = [0.0, 0.05, 0.10, 0.20]
MISSING_SEEDS = [42, 123, 999]

# Smoke test config
SMOKE_CATEGORIES = ["bagel"]

# ══════════════════════════════════════════════════════════════════════
# Perturbation Functions
# ══════════════════════════════════════════════════════════════════════

def _get_valid_mask(organized_pc: torch.Tensor) -> torch.Tensor:
    """Get boolean mask of valid (non-zero) points in organized point cloud.
    organized_pc: [1, 3, H, W] or [3, H, W]
    Returns: [H, W] boolean mask
    """
    if organized_pc.dim() == 4:
        pc = organized_pc.squeeze(0)  # [3, H, W]
    else:
        pc = organized_pc
    # A point is valid if ANY channel is non-zero
    return (pc.abs().sum(dim=0) > 0)


def _get_bbox_diagonal(organized_pc: torch.Tensor) -> float:
    """Compute bounding box diagonal of valid points."""
    valid = _get_valid_mask(organized_pc)
    if organized_pc.dim() == 4:
        pc = organized_pc.squeeze(0)
    else:
        pc = organized_pc
    valid_coords = pc[:, valid]  # [3, N_valid]
    if valid_coords.shape[1] == 0:
        return 1.0
    bbox_min = valid_coords.min(dim=1).values
    bbox_max = valid_coords.max(dim=1).values
    return float(torch.norm(bbox_max - bbox_min).item())


def perturb_sparsify(organized_pc: torch.Tensor, keep_ratio: float, seed: int = 42) -> torch.Tensor:
    """Randomly zero out points in organized point cloud.
    
    Args:
        organized_pc: [1, 3, H, W] float tensor
        keep_ratio: fraction of valid points to keep (0.0 to 1.0)
        seed: random seed for reproducibility
    
    Returns:
        [1, 3, H, W] tensor with some valid points set to (0,0,0)
    """
    if keep_ratio >= 1.0:
        return organized_pc
    
    rng = np.random.RandomState(seed)
    result = organized_pc.clone()
    valid_mask = _get_valid_mask(organized_pc)  # [H, W]
    valid_indices = torch.nonzero(valid_mask, as_tuple=False)  # [N_valid, 2]
    
    if valid_indices.shape[0] == 0:
        return result
    
    n_keep = max(1, int(valid_indices.shape[0] * keep_ratio))
    keep_idx = rng.choice(valid_indices.shape[0], size=n_keep, replace=False)
    # Get indices to drop (set to zero)
    drop_mask = np.ones(valid_indices.shape[0], dtype=bool)
    drop_mask[keep_idx] = False
    drop_indices = valid_indices[drop_mask]  # [N_drop, 2]
    
    if result.dim() == 4:
        result[0, :, drop_indices[:, 0], drop_indices[:, 1]] = 0.0
    else:
        result[:, drop_indices[:, 0], drop_indices[:, 1]] = 0.0
    
    return result


def perturb_gaussian_noise(organized_pc: torch.Tensor, sigma_ratio: float, 
                           seed: int = 42) -> torch.Tensor:
    """Add Gaussian noise to valid point coordinates.
    
    Args:
        organized_pc: [1, 3, H, W] float tensor
        sigma_ratio: sigma = sigma_ratio * D, where D = bbox diagonal
        seed: random seed
    
    Returns:
        [1, 3, H, W] tensor with noise added to valid points
    """
    if sigma_ratio <= 0.0:
        return organized_pc
    
    rng = np.random.RandomState(seed)
    D = _get_bbox_diagonal(organized_pc)
    sigma = sigma_ratio * D
    
    result = organized_pc.clone()
    valid_mask = _get_valid_mask(organized_pc)  # [H, W]
    
    if result.dim() == 4:
        noise = rng.normal(0, sigma, size=(valid_mask.sum().item(), 3))
        noise_tensor = torch.from_numpy(noise).float().to(result.device)
        result[0, :, valid_mask] += noise_tensor.T
    else:
        noise = rng.normal(0, sigma, size=(valid_mask.sum().item(), 3))
        noise_tensor = torch.from_numpy(noise).float().to(result.device)
        result[:, valid_mask] += noise_tensor.T
    
    return result


def perturb_block_dropout(organized_pc: torch.Tensor, missing_ratio: float, 
                          seed: int = 42) -> torch.Tensor:
    """Zero out a rectangular block of points (simulates depth sensor failure).
    
    Args:
        organized_pc: [1, 3, H, W] float tensor
        missing_ratio: fraction of the image area to zero out (~0.05 to 0.20)
        seed: random seed
    
    Returns:
        [1, 3, H, W] tensor with a block of points zeroed out
    """
    if missing_ratio <= 0.0:
        return organized_pc
    
    rng = np.random.RandomState(seed)
    result = organized_pc.clone()
    
    # Get H, W
    if result.dim() == 4:
        _, _, H, W = result.shape
        pc_view = result[0]
    else:
        _, H, W = result.shape
        pc_view = result
    
    # Block size: sqrt(missing_ratio) * min(H,W)
    block_size = int(math.sqrt(missing_ratio) * min(H, W))
    block_size = max(4, block_size)
    
    # Random block position
    h_start = rng.randint(0, H - block_size)
    w_start = rng.randint(0, W - block_size)
    
    pc_view[:, h_start:h_start + block_size, w_start:w_start + block_size] = 0.0
    
    return result


# ══════════════════════════════════════════════════════════════════════
# Feature Variant Classes (Exp 2.4)
# ══════════════════════════════════════════════════════════════════════

class FPFHOnlyFeatures(DoubleRGBFPFHFeatures_add_late):
    """Variant: FPFH features only (33-dim), no density channels."""
    
    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        """Override: return FPFH without density channels."""
        # Reuse the FPFH computation from parent, but strip density channels
        full_features = super().get_fpfh_features(organized_pc, voxel_size, savefpfh_path)
        # full_features is [1, 37, H, W], keep only first 33 (FPFH)
        return full_features[:, :33, :, :]


class FPFHNormDensityFeatures(DoubleRGBFPFHFeatures_add_late):
    """Variant: FPFH + normalized density (min-max normalize each radius).
    
    Normalizes raw point counts to [0, 1] per radius across all points,
    making density features scale-invariant and comparable across radii.
    """
    
    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        """Override: normalize density channels per radius."""
        import open3d as o3d
        
        # ── FPFH computation (same as parent) ──
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)
        nonzero_mask = np.all(unorganized_pc != 0, axis=1)
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]
        
        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        bbox = o3d_pc.get_axis_aligned_bounding_box()
        avg_size = np.mean(bbox.get_extent())
        voxel_scales = [0.05]
        
        for scale in voxel_scales:
            current_voxel_size = avg_size * scale
            radius_feature = current_voxel_size * 5
            o3d_pc.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=current_voxel_size, max_nn=30
                )
            )
            pcd_fpfh = o3d.registration.compute_fpfh_feature(
                o3d_pc,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=radius_feature, max_nn=100
                )
            )
            fpfh = pcd_fpfh.data.T
        
        # ── Multi-scale density (same radii as parent) ──
        points = np.asarray(o3d_pc.points)
        kdtree = o3d.geometry.KDTreeFlann(o3d_pc)
        
        radii = [
            voxel_size * 0.03,
            voxel_size * 0.07,
            voxel_size * 0.11,
            voxel_size * 0.15,
        ]
        
        raw_densities = []
        for r in radii:
            dens = np.zeros(len(points))
            for i in range(len(points)):
                [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], r)
                dens[i] = k
            raw_densities.append(dens)
        
        # ── Normalize each density channel to [0, 1] ──
        normalized_densities = []
        for dens in raw_densities:
            d_min, d_max = dens.min(), dens.max()
            if d_max - d_min > 0:
                norm_dens = (dens - d_min) / (d_max - d_min)
            else:
                norm_dens = np.zeros_like(dens)
            normalized_densities.append(norm_dens)
        
        # ── Assemble features: FPFH (33) + 4 normalized densities ──
        fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 4))
        fpfh_with_density[:, :-4] = fpfh
        fpfh_with_density[:, -4] = normalized_densities[3]  # largest radius
        fpfh_with_density[:, -3] = normalized_densities[2]
        fpfh_with_density[:, -2] = normalized_densities[1]
        fpfh_with_density[:, -1] = normalized_densities[0]  # smallest radius
        
        full_fpfh_with_density = np.zeros((unorganized_pc.shape[0], fpfh_with_density.shape[1]))
        full_fpfh_with_density[nonzero_indices, :] = fpfh_with_density
        full_fpfhwithden_reshaped = full_fpfh_with_density.reshape(
            (organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1] + 4))
        result = torch.tensor(full_fpfhwithden_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
        return result


# ══════════════════════════════════════════════════════════════════════
# Experiment Runner
# ══════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    """Stores results for one experiment configuration."""
    exp_name: str
    exp_type: str  # 'sparsification', 'noise', 'missing', 'variants'
    category: str
    config_label: str  # e.g. "keep_0.75_seed_42"
    config_detail: Dict = field(default_factory=dict)
    i_auroc: float = 0.0
    p_auroc: float = 0.0
    aupro: float = 0.0


def run_evaluation_with_perturbation(
    method: DoubleRGBFPFHFeatures_add_late,
    class_name: str,
    args,
    perturbation_fn: Optional[Callable] = None,
) -> Tuple[float, float, float]:
    """Run evaluation with optional perturbation on test point clouds.
    
    Args:
        method: Trained feature extractor (memory bank already built)
        class_name: Category name
        args: Command line arguments
        perturbation_fn: Function (organized_pc) -> perturbed_pc, or None
    
    Returns:
        (I-AUROC, P-AUROC, AUPRO)
    """
    test_loader = get_data_loader("test", class_name=class_name, 
                                   img_size=args.img_size, args=args)
    
    # Reset metrics
    method.image_preds = []
    method.image_labels = []
    method.pixel_preds = []
    method.pixel_labels = []
    method.gts = []
    method.predictions = []
    
    with torch.no_grad():
        for sample, mask, label, rgb_path in test_loader:
            # Apply perturbation if specified
            if perturbation_fn is not None:
                perturbed_pc = perturbation_fn(sample[1])
                perturbed_sample = (sample[0], perturbed_pc, sample[2])
            else:
                perturbed_sample = sample
            
            method.predict(perturbed_sample, mask, label)
    
    method.calculate_metrics()
    return method.image_rocauc, method.pixel_rocauc, method.au_pro


def train_memory_bank(method_class, args, class_name) -> DoubleRGBFPFHFeatures_add_late:
    """Train memory bank on clean data.
    
    Returns:
        Trained method instance (memory bank built, coreset applied)
    """
    method = method_class(args)
    
    train_loader = get_data_loader("train", class_name=class_name, 
                                    img_size=args.img_size, args=args)
    flag = 0
    for sample, _ in tqdm(train_loader, desc=f'  Train {class_name}', leave=False):
        if args.save_feature:
            method.add_sample_to_mem_bank(sample, class_name=class_name)
        else:
            method.add_sample_to_mem_bank(sample)
        flag += 1
        if flag > args.max_sample:
            break
    
    method.run_coreset()
    return method


def run_sparsification_experiment(args, categories: List[str], 
                                   output_dir: Path) -> List[ExperimentResult]:
    """Experiment 2.1: Point cloud random sparsification."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category in tqdm(categories, desc="Sparsification categories"):
        print(f"\n[Sparsification] Training on {category} (clean data)...")
        method = train_memory_bank(DoubleRGBFPFHFeatures_add_late, args, category)
        
        for ratio in SPARSIFY_RATIOS:
            for seed in SPARSIFY_SEEDS:
                label = f"keep_{ratio:.2f}_seed_{seed}"
                print(f"  Evaluating: {label}")
                
                def perturb_fn(pc, r=ratio, s=seed):
                    return perturb_sparsify(pc, r, s)
                
                i_auc, p_auc, aupro = run_evaluation_with_perturbation(
                    method, category, args, perturbation_fn=perturb_fn)
                
                result = ExperimentResult(
                    exp_name="sparsification",
                    exp_type="sparsification",
                    category=category,
                    config_label=label,
                    config_detail={"keep_ratio": ratio, "seed": seed},
                    i_auroc=round(i_auc, 4),
                    p_auroc=round(p_auc, 4),
                    aupro=round(aupro, 4),
                )
                results.append(result)
                print(f"    I-AUROC={i_auc:.4f}, P-AUROC={p_auc:.4f}, AUPRO={aupro:.4f}")
        
        # Clean up
        del method
        torch.cuda.empty_cache()
    
    return results


def run_noise_experiment(args, categories: List[str], 
                          output_dir: Path) -> List[ExperimentResult]:
    """Experiment 2.2: Gaussian noise perturbation."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category in tqdm(categories, desc="Noise categories"):
        print(f"\n[Noise] Training on {category} (clean data)...")
        method = train_memory_bank(DoubleRGBFPFHFeatures_add_late, args, category)
        
        for sigma_ratio in NOISE_SIGMA_RATIOS:
            for seed in NOISE_SEEDS:
                label = f"sigma_{sigma_ratio:.4f}_seed_{seed}"
                print(f"  Evaluating: {label}")
                
                def perturb_fn(pc, sr=sigma_ratio, s=seed):
                    return perturb_gaussian_noise(pc, sr, s)
                
                i_auc, p_auc, aupro = run_evaluation_with_perturbation(
                    method, category, args, perturbation_fn=perturb_fn)
                
                result = ExperimentResult(
                    exp_name="noise",
                    exp_type="noise",
                    category=category,
                    config_label=label,
                    config_detail={"sigma_ratio": sigma_ratio, "seed": seed},
                    i_auroc=round(i_auc, 4),
                    p_auroc=round(p_auc, 4),
                    aupro=round(aupro, 4),
                )
                results.append(result)
                print(f"    I-AUROC={i_auc:.4f}, P-AUROC={p_auc:.4f}, AUPRO={aupro:.4f}")
        
        del method
        torch.cuda.empty_cache()
    
    return results


def run_missing_experiment(args, categories: List[str], 
                            output_dir: Path) -> List[ExperimentResult]:
    """Experiment 2.3: Local missing / block dropout."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for category in tqdm(categories, desc="Missing categories"):
        print(f"\n[Missing] Training on {category} (clean data)...")
        method = train_memory_bank(DoubleRGBFPFHFeatures_add_late, args, category)
        
        for ratio in MISSING_RATIOS:
            for seed in MISSING_SEEDS:
                label = f"missing_{ratio:.2f}_seed_{seed}"
                print(f"  Evaluating: {label}")
                
                def perturb_fn(pc, r=ratio, s=seed):
                    return perturb_block_dropout(pc, r, s)
                
                i_auc, p_auc, aupro = run_evaluation_with_perturbation(
                    method, category, args, perturbation_fn=perturb_fn)
                
                result = ExperimentResult(
                    exp_name="missing",
                    exp_type="missing",
                    category=category,
                    config_label=label,
                    config_detail={"missing_ratio": ratio, "seed": seed},
                    i_auroc=round(i_auc, 4),
                    p_auroc=round(p_auc, 4),
                    aupro=round(aupro, 4),
                )
                results.append(result)
                print(f"    I-AUROC={i_auc:.4f}, P-AUROC={p_auc:.4f}, AUPRO={aupro:.4f}")
        
        del method
        torch.cuda.empty_cache()
    
    return results


def run_variants_experiment(args, categories: List[str], 
                             output_dir: Path) -> List[ExperimentResult]:
    """Experiment 2.4: MDFE feature variants comparison."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    variants = {
        "FPFH_only": FPFHOnlyFeatures,
        "FPFH_raw_density": DoubleRGBFPFHFeatures_add_late,  # current method
        "FPFH_norm_density": FPFHNormDensityFeatures,
    }
    
    for category in tqdm(categories, desc="Variants categories"):
        for variant_name, variant_class in variants.items():
            print(f"\n[Variants] {variant_name} on {category}")
            
            method = train_memory_bank(variant_class, args, category)
            i_auc, p_auc, aupro = run_evaluation_with_perturbation(
                method, category, args, perturbation_fn=None)
            
            result = ExperimentResult(
                exp_name="variants",
                exp_type="variants",
                category=category,
                config_label=variant_name,
                config_detail={"variant": variant_name},
                i_auroc=round(i_auc, 4),
                p_auroc=round(p_auc, 4),
                aupro=round(aupro, 4),
            )
            results.append(result)
            print(f"    {variant_name}: I-AUROC={i_auc:.4f}, P-AUROC={p_auc:.4f}, AUPRO={aupro:.4f}")
            
            del method
            torch.cuda.empty_cache()
    
    return results


# ══════════════════════════════════════════════════════════════════════
# Result Aggregation & Saving
# ══════════════════════════════════════════════════════════════════════

def save_and_summarize(results: List[ExperimentResult], output_dir: Path, 
                       exp_type: str):
    """Save detailed CSV and compute summary statistics."""
    # Save detailed results
    rows = []
    for r in results:
        rows.append({
            "exp_type": r.exp_type,
            "category": r.category,
            "config": r.config_label,
            "I_AUROC": r.i_auroc,
            "P_AUROC": r.p_auroc,
            "AUPRO": r.aupro,
            **r.config_detail,
        })
    df = pd.DataFrame(rows)
    detail_path = output_dir / f"{exp_type}_detailed.csv"
    df.to_csv(detail_path, index=False)
    print(f"\nDetailed results saved to: {detail_path}")
    
    # Compute per-condition summary (mean ± std across seeds)
    if exp_type in ("sparsification", "noise", "missing"):
        if exp_type == "sparsification":
            group_col = "keep_ratio"
        elif exp_type == "noise":
            group_col = "sigma_ratio"
        else:
            group_col = "missing_ratio"
        
        summary_rows = []
        for cond_value in df[group_col].unique():
            cond_df = df[df[group_col] == cond_value]
            summary_rows.append({
                group_col: cond_value,
                "I_AUROC_mean": cond_df["I_AUROC"].mean(),
                "I_AUROC_std": cond_df["I_AUROC"].std(),
                "P_AUROC_mean": cond_df["P_AUROC"].mean(),
                "P_AUROC_std": cond_df["P_AUROC"].std(),
                "AUPRO_mean": cond_df["AUPRO"].mean(),
                "AUPRO_std": cond_df["AUPRO"].std(),
                "num_samples": len(cond_df),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / f"{exp_type}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
        print("\nSummary:")
        print(summary_df.to_markdown(index=False))
    
    elif exp_type == "variants":
        # Per-variant mean across categories
        summary_rows = []
        for variant in df["variant"].unique():
            vdf = df[df["variant"] == variant]
            summary_rows.append({
                "variant": variant,
                "I_AUROC_mean": round(vdf["I_AUROC"].mean(), 4),
                "P_AUROC_mean": round(vdf["P_AUROC"].mean(), 4),
                "AUPRO_mean": round(vdf["AUPRO"].mean(), 4),
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_path = output_dir / f"{exp_type}_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Summary saved to: {summary_path}")
        print("\nVariant Comparison (mean over categories):")
        print(summary_df.to_markdown(index=False))
    
    return df


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def build_args():
    """Build argument parser matching the original DAMD args."""
    parser = argparse.ArgumentParser(description="DAMD Robustness Experiments")
    
    # Experiment selection
    parser.add_argument("--exp", type=str, default="all",
                        choices=["all", "sparsification", "noise", "missing", "variants"],
                        help="Which experiment(s) to run")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: run only on 1 category with reduced configs")
    parser.add_argument("--categories", type=str, default="",
                        help="Comma-separated category list (default: all MVTec 3D-AD)")
    parser.add_argument("--dataset", type=str, default="mvtec3d",
                        choices=["mvtec3d", "eyecandies"])
    
    # DAMD model args (matching main.py defaults)
    parser.add_argument("--method_name", default="DINO+FPFH+add+late", type=str)
    parser.add_argument("--max_sample", default=400, type=int)
    parser.add_argument("--memory_bank", default="single", type=str)
    parser.add_argument("--rgb_backbone_name", default="vit_base_patch8_224_dino", type=str)
    parser.add_argument("--xyz_backbone_name", default="Point_MAE", type=str)
    parser.add_argument("--fusion_module_path", 
                        default=str(default_fusion_checkpoint()), type=str)
    parser.add_argument("--save_feature", default=False, action="store_true")
    parser.add_argument("--use_uff", default=False, action="store_true")
    parser.add_argument("--use_trans", default=True, action="store_true")
    parser.add_argument("--use_ssl", default=False, action="store_true")
    parser.add_argument("--save_preds", default=True, action="store_true")
    parser.add_argument("--group_size", default=128, type=int)
    parser.add_argument("--num_group", default=1024, type=int)
    parser.add_argument("--random_state", default=None, type=int)
    parser.add_argument("--dataset_type", default="mvtec3d", type=str)
    parser.add_argument("--dataset_path", 
                        default=str(default_dataset_root("mvtec3d")), type=str)
    parser.add_argument("--save_feature_path", 
                        default=str(default_feature_cache("mvtec3d")), type=str)
    parser.add_argument("--img_size", default=224, type=int)
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
    
    return parser


def main():
    parser = build_args()
    args = parser.parse_args()
    
    # ── Output directory ──
    results_root = Path(__file__).resolve().parent / "results"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # ── Dataset selection ──
    if args.dataset == "mvtec3d":
        all_classes = mvtec3d_classes()
    else:
        all_classes = eyecandies_classes()
    
    if args.categories:
        categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    elif args.smoke:
        categories = SMOKE_CATEGORIES
    else:
        categories = all_classes
    
    print(f"=" * 70)
    print(f"DAMD Robustness Experiments")
    print(f"=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"Categories: {categories}")
    print(f"Experiment: {args.exp}")
    print(f"Smoke test: {args.smoke}")
    print(f"Results root: {results_root}")
    print(f"=" * 70)
    
    # ── Global override for smoke test ──
    global SPARSIFY_RATIOS, SPARSIFY_SEEDS
    global NOISE_SIGMA_RATIOS, NOISE_SEEDS
    global MISSING_RATIOS, MISSING_SEEDS
    
    if args.smoke:
        SPARSIFY_RATIOS = [1.0, 0.50]
        SPARSIFY_SEEDS = [42]
        NOISE_SIGMA_RATIOS = [0.0, 0.005]
        NOISE_SEEDS = [42]
        MISSING_RATIOS = [0.0, 0.10]
        MISSING_SEEDS = [42]
    
    all_results = []
    
    # ── Run experiments ──
    start_time = time.time()
    
    if args.exp in ("all", "sparsification"):
        print("\n" + "=" * 50)
        print("Experiment 2.1: Point Cloud Sparsification")
        print("=" * 50)
        output_dir = results_root / "sparsification"
        results = run_sparsification_experiment(args, categories, output_dir)
        save_and_summarize(results, output_dir, "sparsification")
        all_results.append(("sparsification", results))
    
    if args.exp in ("all", "noise"):
        print("\n" + "=" * 50)
        print("Experiment 2.2: Gaussian Noise Perturbation")
        print("=" * 50)
        output_dir = results_root / "noise"
        results = run_noise_experiment(args, categories, output_dir)
        save_and_summarize(results, output_dir, "noise")
        all_results.append(("noise", results))
    
    if args.exp in ("all", "missing"):
        print("\n" + "=" * 50)
        print("Experiment 2.3: Local Missing / Block Dropout")
        print("=" * 50)
        output_dir = results_root / "missing"
        results = run_missing_experiment(args, categories, output_dir)
        save_and_summarize(results, output_dir, "missing")
        all_results.append(("missing", results))
    
    if args.exp in ("all", "variants"):
        print("\n" + "=" * 50)
        print("Experiment 2.4: MDFE Feature Variants")
        print("=" * 50)
        output_dir = results_root / "variants"
        results = run_variants_experiment(args, categories, output_dir)
        save_and_summarize(results, output_dir, "variants")
        all_results.append(("variants", results))
    
    elapsed = time.time() - start_time
    print(f"\n{'=' * 70}")
    print(f"All experiments completed in {elapsed/60:.1f} minutes")
    print(f"Results saved to: {results_root}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
