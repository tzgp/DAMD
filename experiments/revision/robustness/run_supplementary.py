
#!/usr/bin/env python3
"""Supplementary robustness experiments: gentle sparsification + single-scale density."""

import sys, os, time, argparse, math
from pathlib import Path
from typing import List
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from path_defaults import default_dataset_root, default_feature_cache, default_fusion_checkpoint

from dataset import get_data_loader, mvtec3d_classes
from feature_extractors.selfmultiple_featureslate import DoubleRGBFPFHFeatures_add_late
from utils.au_pro_util import calculate_au_pro
from utils.mvtec3d_util import organized_pc_to_unorganized_pc

# ── Import existing perturbation functions ──
from experiments.revision.robustness.run_robustness import (
    perturb_sparsify, perturb_gaussian_noise, perturb_block_dropout,
    FPFHOnlyFeatures, FPFHNormDensityFeatures,
    run_evaluation_with_perturbation, ExperimentResult,
)

# ══════════════════════════════════════════════════════════════════
# NEW: Single-scale density variant
# ══════════════════════════════════════════════════════════════════

class FPFHSingleScaleDensityFeatures(DoubleRGBFPFHFeatures_add_late):
    """Variant: FPFH + single-scale density (only 1 density channel, 34-dim).
    
    Uses only the medium density radius (voxel_size * 0.07) for a compact 
    33 (FPFH) + 1 (density) = 34-dimensional feature.
    """

    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        import open3d as o3d
        organized_pc_np = organized_pc.squeeze().permute(1, 2, 0).numpy()
        unorganized_pc = organized_pc_to_unorganized_pc(organized_pc_np)
        nonzero_mask = np.all(unorganized_pc != 0, axis=1)
        nonzero_indices = np.nonzero(nonzero_mask)[0]
        unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

        o3d_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(unorganized_pc_no_zeros))
        bbox = o3d_pc.get_axis_aligned_bounding_box()
        avg_size = np.mean(bbox.get_extent())

        # ── FPFH computation ──
        current_voxel_size = avg_size * 0.05
        radius_feature = current_voxel_size * 5
        o3d_pc.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=current_voxel_size, max_nn=30))
        pcd_fpfh = o3d.registration.compute_fpfh_feature(
            o3d_pc, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        fpfh = pcd_fpfh.data.T  # [N, 33]

        # ── Single-scale density (only one radius) ──
        points = np.asarray(o3d_pc.points)
        kdtree = o3d.geometry.KDTreeFlann(o3d_pc)
        single_radius = voxel_size * 0.07  # medium scale
        densities = np.zeros(len(points))
        for i in range(len(points)):
            [k, idx, _] = kdtree.search_radius_vector_3d(o3d_pc.points[i], single_radius)
            densities[i] = k

        # ── Assemble: 33 FPFH + 1 density = 34 dim ──
        fpfh_with_density = np.zeros((fpfh.shape[0], fpfh.shape[1] + 1))
        fpfh_with_density[:, :-1] = fpfh
        fpfh_with_density[:, -1] = densities

        full = np.zeros((unorganized_pc.shape[0], fpfh_with_density.shape[1]))
        full[nonzero_indices, :] = fpfh_with_density
        full_reshaped = full.reshape((organized_pc_np.shape[0], organized_pc_np.shape[1], fpfh.shape[1] + 1))
        return torch.tensor(full_reshaped).permute(2, 0, 1).unsqueeze(dim=0)


# ══════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════

def run_gentle_sparsification(args, categories, output_dir):
    """Run sparsification with 95% and 90% keep ratios (3 seeds each)."""
    results = []
    ratios = [0.95, 0.90]
    seeds = [42, 123, 999]

    for category in tqdm(categories, desc="Gentle sparsification"):
        print(f"\n[GentleSparsify] Training {category}...")
        method = DoubleRGBFPFHFeatures_add_late(args)
        train_loader = get_data_loader("train", class_name=category, img_size=args.img_size, args=args)
        flag = 0
        for sample, _ in tqdm(train_loader, desc=f'  Train {category}', leave=False):
            method.add_sample_to_mem_bank(sample)
            flag += 1
            if flag > args.max_sample: break
        method.run_coreset()

        for ratio in ratios:
            for seed in seeds:
                label = f"keep_{ratio:.2f}_seed_{seed}"
                print(f"  Evaluating: {label}")

                def perturb_fn(pc, r=ratio, s=seed):
                    return perturb_sparsify(pc, r, s)

                i_auc, p_auc, aupro = run_evaluation_with_perturbation(method, category, args, perturb_fn)
                results.append(ExperimentResult(
                    exp_name="gentle_sparsification", exp_type="sparsification",
                    category=category, config_label=label,
                    config_detail={"keep_ratio": ratio, "seed": seed},
                    i_auroc=round(i_auc, 4), p_auroc=round(p_auc, 4), aupro=round(aupro, 4)))
                print(f"    I-AUROC={i_auc:.4f}, P-AUROC={p_auc:.4f}, AUPRO={aupro:.4f}")

        del method; torch.cuda.empty_cache()

    # Save
    import pandas as pd
    rows = [{"exp_type": r.exp_type, "category": r.category, "config": r.config_label,
             "I_AUROC": r.i_auroc, "P_AUROC": r.p_auroc, "AUPRO": r.aupro, **r.config_detail}
            for r in results]
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "gentle_sparsification_detailed.csv", index=False)

    # Summary
    summary_rows = []
    for ratio in ratios:
        rdf = df[df["keep_ratio"] == ratio]
        summary_rows.append({"keep_ratio": ratio, "I_AUROC_mean": rdf["I_AUROC"].mean(),
                             "I_AUROC_std": rdf["I_AUROC"].std(), "P_AUROC_mean": rdf["P_AUROC"].mean(),
                             "P_AUROC_std": rdf["P_AUROC"].std(), "AUPRO_mean": rdf["AUPRO"].mean(),
                             "AUPRO_std": rdf["AUPRO"].std()})
    sdf = pd.DataFrame(summary_rows)
    sdf.to_csv(output_dir / "gentle_sparsification_summary.csv", index=False)
    print("\nGentle sparsification summary:")
    print(sdf.to_markdown(index=False))
    return results


def run_single_scale_variant(args, categories, output_dir):
    """Run the single-scale density variant."""
    results = []
    for category in tqdm(categories, desc="Single-scale density"):
        print(f"\n[SingleScaleDensity] {category}")
        method = FPFHSingleScaleDensityFeatures(args)
        train_loader = get_data_loader("train", class_name=category, img_size=args.img_size, args=args)
        flag = 0
        for sample, _ in tqdm(train_loader, desc=f'  Train {category}', leave=False):
            method.add_sample_to_mem_bank(sample)
            flag += 1
            if flag > args.max_sample: break
        method.run_coreset()

        i_auc, p_auc, aupro = run_evaluation_with_perturbation(method, category, args, None)
        results.append(ExperimentResult(
            exp_name="single_scale_density", exp_type="variants",
            category=category, config_label="FPFH_single_scale_density",
            config_detail={"variant": "FPFH_single_scale_density"},
            i_auroc=round(i_auc, 4), p_auroc=round(p_auc, 4), aupro=round(aupro, 4)))
        print(f"    Single-scale: I-AUROC={i_auc:.4f}, P-AUROC={p_auc:.4f}, AUPRO={aupro:.4f}")

        del method; torch.cuda.empty_cache()

    import pandas as pd
    rows = [{"exp_type": r.exp_type, "category": r.category, "config": r.config_label,
             "I_AUROC": r.i_auroc, "P_AUROC": r.p_auroc, "AUPRO": r.aupro, **r.config_detail}
            for r in results]
    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "single_scale_variant_detailed.csv", index=False)

    sdf = pd.DataFrame([{"variant": "FPFH_single_scale_density",
                         "I_AUROC_mean": df["I_AUROC"].mean(), "P_AUROC_mean": df["P_AUROC"].mean(),
                         "AUPRO_mean": df["AUPRO"].mean()}])
    sdf.to_csv(output_dir / "single_scale_variant_summary.csv", index=False)
    print("\nSingle-scale variant summary:")
    print(sdf.to_markdown(index=False))
    return results


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--exp", default="all", choices=["all", "gentle_sparsify", "single_scale"])
    parser.add_argument("--dataset_path", default=str(default_dataset_root("mvtec3d")))
    parser.add_argument("--dataset_type", default="mvtec3d")
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--max_sample", default=400, type=int)
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
    parser.add_argument("--memory_bank", default="single")
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
    parser.add_argument("--method_name", default="DINO+FPFH+add+late")
    return parser


def main():
    parser = build_args()
    args = parser.parse_args()

    categories = mvtec3d_classes()
    if args.smoke:
        categories = ["bagel"]

    results_root = Path(__file__).resolve().parent.parent / "results"

    print(f"=" * 60)
    print(f"Supplementary Robustness Experiments")
    print(f"Categories: {categories}")
    print(f"=" * 60)

    t0 = time.time()

    if args.exp in ("all", "gentle_sparsify"):
        print("\n>>> Gentle Sparsification (95%%, 90%%)")
        out = results_root / "sparsification"
        run_gentle_sparsification(args, categories, out)

    if args.exp in ("all", "single_scale"):
        print("\n>>> Single-Scale Density Variant")
        out = results_root / "variants"
        run_single_scale_variant(args, categories, out)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
