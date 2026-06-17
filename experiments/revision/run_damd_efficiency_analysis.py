import argparse
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import psutil
import torch


REPO_ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) >= 3 else Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from path_defaults import default_dataset_root, default_feature_cache, default_fusion_checkpoint

from dataset import get_data_loader, mvtec3d_classes, eyecandies_classes  # noqa: E402
from feature_extractors.selfmultiple_featureslate import DoubleRGBFPFHFeatures_add_late  # noqa: E402


DEVNULL = open(os.devnull, "w")


def default_args(dataset_type: str, dataset_path: str, max_sample: int) -> SimpleNamespace:
    return SimpleNamespace(
        method_name="DINO+FPFH+add+late",
        max_sample=max_sample,
        memory_bank="single",
        rgb_backbone_name="vit_base_patch8_224_dino",
        xyz_backbone_name="Point_MAE",
        fusion_module_path=str(default_fusion_checkpoint()),
        save_feature=False,
        use_uff=False,
        use_trans=True,
        use_ssl=False,
        save_preds=True,
        group_size=128,
        num_group=1024,
        random_state=None,
        dataset_type=dataset_type,
        dataset_path=dataset_path,
        save_feature_path=str(default_feature_cache(dataset_type)),
        img_size=224,
        xyz_s_lambda=1.0,
        xyz_smap_lambda=1.0,
        rgb_s_lambda=0.1,
        rgb_smap_lambda=0.1,
        fusion_s_lambda=1.0,
        fusion_smap_lambda=1.0,
        coreset_eps=0.9,
        f_coreset=0.1,
        asy_memory_bank=None,
        ocsvm_nu=0.5,
        ocsvm_maxiter=1000,
        rm_zero_for_project=False,
    )


def _sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def bytes_to_mb(num_bytes: int) -> float:
    return float(num_bytes) / (1024.0 ** 2)


class TimedLateFusion(DoubleRGBFPFHFeatures_add_late):
    def __init__(self, args):
        super().__init__(args)
        self.phase = None
        self.timing_enabled = False
        self.rgb_times = []
        self.fpfh_times = []
        self.retrieval_times = []
        self.total_predict_times = []
        self.total_fit_times = []
        self.valid_point_counts = []

    def _record(self, bucket, value: float) -> None:
        if self.timing_enabled:
            bucket.append(float(value))

    def __call__(self, rgb):
        _sync_cuda()
        start = time.perf_counter()
        output = super().__call__(rgb)
        _sync_cuda()
        elapsed = time.perf_counter() - start
        if self.phase == "predict":
            self._record(self.rgb_times, elapsed)
        return output

    def get_fpfh_features(self, organized_pc, voxel_size=0.05, savefpfh_path="./savefpfh_path"):
        start = time.perf_counter()
        output = super().get_fpfh_features(organized_pc, voxel_size=voxel_size, savefpfh_path=savefpfh_path)
        elapsed = time.perf_counter() - start
        if self.phase == "predict":
            self._record(self.fpfh_times, elapsed)
        return output

    def compute_s_s_map(self, xyz_patch, rgb_patch, feature_map_dims, mask, label, nonzero_indices, xyz):
        start = time.perf_counter()
        output = super().compute_s_s_map(xyz_patch, rgb_patch, feature_map_dims, mask, label, nonzero_indices, xyz)
        elapsed = time.perf_counter() - start
        if self.phase == "predict":
            self._record(self.retrieval_times, elapsed)
        return output

    def add_sample_to_mem_bank(self, sample, class_name=None):
        previous_phase = self.phase
        self.phase = "fit"
        start = time.perf_counter()
        try:
            return super().add_sample_to_mem_bank(sample, class_name=class_name)
        finally:
            elapsed = time.perf_counter() - start
            self._record(self.total_fit_times, elapsed)
            self.phase = previous_phase

    def predict(self, sample, mask, label):
        previous_phase = self.phase
        self.phase = "predict"
        nonzero = int(torch.count_nonzero(torch.any(sample[1] != 0, dim=1)).item())
        start = time.perf_counter()
        try:
            result = super().predict(sample, mask, label)
        finally:
            elapsed = time.perf_counter() - start
            self._record(self.total_predict_times, elapsed)
            if self.timing_enabled:
                self.valid_point_counts.append(nonzero)
            self.phase = previous_phase
        return result


def class_names_for(dataset_type: str):
    if dataset_type == "mvtec3d":
        return mvtec3d_classes()
    return eyecandies_classes()


def fit_method(method: TimedLateFusion, args, class_name: str) -> dict:
    train_loader = get_data_loader("train", class_name=class_name, img_size=args.img_size, args=args)
    flag = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for sample, _ in train_loader:
        with contextlib.redirect_stdout(DEVNULL):
            method.add_sample_to_mem_bank(sample)
        flag += 1
        if flag > args.max_sample:
            break
    with contextlib.redirect_stdout(DEVNULL):
        method.run_coreset()
    total_time = time.perf_counter() - start
    fit_gpu_peak_mb = bytes_to_mb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    return {
        "fit_wall_time_s": total_time,
        "fit_samples": flag,
        "fit_gpu_peak_mb": fit_gpu_peak_mb,
    }


def evaluate_method(method: TimedLateFusion, args, class_name: str, warmup: int, timed_samples: int) -> dict:
    test_loader = get_data_loader("test", class_name=class_name, img_size=args.img_size, args=args)
    total_seen = 0
    timed_seen = 0
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    with torch.no_grad():
        for sample, mask, label, _rgb_path in test_loader:
            method.timing_enabled = total_seen >= warmup and timed_seen < timed_samples
            with contextlib.redirect_stdout(DEVNULL):
                method.predict(sample, mask, label)
            if method.timing_enabled:
                timed_seen += 1
            total_seen += 1
            if timed_seen >= timed_samples:
                break
    method.timing_enabled = False
    wall_time = time.perf_counter() - wall_start
    predict_gpu_peak_mb = bytes_to_mb(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0.0
    return {
        "warmup_samples": min(warmup, total_seen),
        "timed_samples": timed_seen,
        "predict_wall_time_s": wall_time,
        "predict_gpu_peak_mb": predict_gpu_peak_mb,
    }


def summarize_method_storage(method: TimedLateFusion) -> dict:
    params = list(method.deep_feature_extractor.parameters())
    model_param_count = int(sum(p.numel() for p in params))
    model_param_bytes = int(sum(p.numel() * p.element_size() for p in params))
    rgb_bank_count = int(method.patch_rgb_lib.shape[0])
    xyz_bank_count = int(method.patch_xyz_lib.shape[0])
    rgb_bank_dim = int(method.patch_rgb_lib.shape[1])
    xyz_bank_dim = int(method.patch_xyz_lib.shape[1])
    rgb_bank_bytes = tensor_bytes(method.patch_rgb_lib)
    xyz_bank_bytes = tensor_bytes(method.patch_xyz_lib)
    process_rss_mb = bytes_to_mb(psutil.Process(os.getpid()).memory_info().rss)
    return {
        "model_param_count": model_param_count,
        "model_param_mb": bytes_to_mb(model_param_bytes),
        "rgb_bank_count": rgb_bank_count,
        "rgb_bank_dim": rgb_bank_dim,
        "rgb_bank_mb": bytes_to_mb(rgb_bank_bytes),
        "xyz_bank_count": xyz_bank_count,
        "xyz_bank_dim": xyz_bank_dim,
        "xyz_bank_mb": bytes_to_mb(xyz_bank_bytes),
        "total_bank_mb": bytes_to_mb(rgb_bank_bytes + xyz_bank_bytes),
        "process_rss_mb": process_rss_mb,
    }


def safe_stats(values):
    if not values:
        return {"mean_ms": float("nan"), "std_ms": float("nan")}
    arr = np.asarray(values, dtype=np.float64) * 1000.0
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std(ddof=0))}


def run_one_category(args, class_name: str, warmup: int, timed_samples: int) -> dict:
    method = TimedLateFusion(args)
    fit_info = fit_method(method, args, class_name)
    eval_info = evaluate_method(method, args, class_name, warmup=warmup, timed_samples=timed_samples)
    storage = summarize_method_storage(method)
    total_stats = safe_stats(method.total_predict_times)
    rgb_stats = safe_stats(method.rgb_times)
    fpfh_stats = safe_stats(method.fpfh_times)
    retrieval_stats = safe_stats(method.retrieval_times)
    valid_points = np.asarray(method.valid_point_counts, dtype=np.float64) if method.valid_point_counts else np.asarray([])
    result = {
        "category": class_name,
        **fit_info,
        **eval_info,
        **storage,
        "mean_valid_points": float(valid_points.mean()) if valid_points.size else float("nan"),
        "std_valid_points": float(valid_points.std(ddof=0)) if valid_points.size else float("nan"),
        "total_mean_ms": total_stats["mean_ms"],
        "total_std_ms": total_stats["std_ms"],
        "rgb_mean_ms": rgb_stats["mean_ms"],
        "rgb_std_ms": rgb_stats["std_ms"],
        "fpfh_mean_ms": fpfh_stats["mean_ms"],
        "fpfh_std_ms": fpfh_stats["std_ms"],
        "retrieval_mean_ms": retrieval_stats["mean_ms"],
        "retrieval_std_ms": retrieval_stats["std_ms"],
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-type", default="mvtec3d", choices=["mvtec3d", "eyecandies"])
    parser.add_argument("--dataset-path", default=str(default_dataset_root("mvtec3d")))
    parser.add_argument("--categories", nargs="*", default=["all"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timed-samples", type=int, default=10)
    parser.add_argument("--max-sample", type=int, default=400)
    parser.add_argument("--output-dir", default="experiments/revision/efficiency/results")
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    args = default_args(cli_args.dataset_type, cli_args.dataset_path, cli_args.max_sample)
    categories = class_names_for(cli_args.dataset_type) if cli_args.categories == ["all"] else cli_args.categories
    output_dir = Path(cli_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for category in categories:
        print(f"[EFFICIENCY] Running category: {category}")
        row = run_one_category(args, category, warmup=cli_args.warmup, timed_samples=cli_args.timed_samples)
        rows.append(row)
        print(json.dumps({
            "category": category,
            "total_mean_ms": row["total_mean_ms"],
            "rgb_mean_ms": row["rgb_mean_ms"],
            "fpfh_mean_ms": row["fpfh_mean_ms"],
            "retrieval_mean_ms": row["retrieval_mean_ms"],
            "total_bank_mb": row["total_bank_mb"],
        }, ensure_ascii=False))

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "efficiency_by_category.csv", index=False)

    numeric_cols = [
        "fit_wall_time_s",
        "fit_samples",
        "fit_gpu_peak_mb",
        "warmup_samples",
        "timed_samples",
        "predict_wall_time_s",
        "predict_gpu_peak_mb",
        "model_param_count",
        "model_param_mb",
        "rgb_bank_count",
        "rgb_bank_dim",
        "rgb_bank_mb",
        "xyz_bank_count",
        "xyz_bank_dim",
        "xyz_bank_mb",
        "total_bank_mb",
        "process_rss_mb",
        "mean_valid_points",
        "std_valid_points",
        "total_mean_ms",
        "total_std_ms",
        "rgb_mean_ms",
        "rgb_std_ms",
        "fpfh_mean_ms",
        "fpfh_std_ms",
        "retrieval_mean_ms",
        "retrieval_std_ms",
    ]
    summary = {col: float(df[col].mean()) for col in numeric_cols if col in df}
    summary["categories"] = categories
    summary["num_categories"] = len(categories)
    summary["max_total_bank_mb"] = float(df["total_bank_mb"].max())
    summary["max_total_mean_ms"] = float(df["total_mean_ms"].max())
    summary["hardware"] = {
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    with open(output_dir / "efficiency_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("[EFFICIENCY] Saved:", output_dir / "efficiency_by_category.csv")
    print("[EFFICIENCY] Saved:", output_dir / "efficiency_summary.json")


if __name__ == "__main__":
    main()
