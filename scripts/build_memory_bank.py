from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from run_reproduction import _load_config, _preprocess, _resolve, _run


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build DAMD memory-bank artifacts from a reproduction config without running evaluation."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file under configs/.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument(
        "--dataset-root",
        default="",
        help="Override the dataset root used by the config (for mvtec3d this is the processed dataset path; for eyecandies this is the raw dataset root).",
    )
    parser.add_argument(
        "--processed-root",
        default="",
        help="Override the processed dataset root for eyecandies memory-bank construction.",
    )
    args = parser.parse_args()

    config_path = _resolve(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")

    config = _load_config(config_path)
    dataset_cfg = config["dataset"]
    outputs_cfg = config["outputs"]
    run_cfg = config["run"]
    ckpt_cfg = config["checkpoints"]

    feature_cache = _resolve(outputs_cfg["save_feature_path"])
    log_path = _resolve(outputs_cfg.get("memory_bank_log_path", "results/logs/memory_bank_build.log"))
    if feature_cache is None or log_path is None:
        raise ValueError("save_feature_path and memory_bank_log_path must resolve to valid paths")
    feature_cache.mkdir(parents=True, exist_ok=True)

    if args.dataset_root:
        if dataset_cfg["name"] == "mvtec3d":
            os.environ["DAMD_MVTEC3D_ROOT"] = args.dataset_root
        else:
            os.environ["DAMD_EYECANDIES_ROOT"] = args.dataset_root
    if args.processed_root:
        os.environ["DAMD_EYECANDIES_PREPROCESSED_ROOT"] = args.processed_root

    dataset_path = _preprocess(dataset_cfg, log_path, args.skip_preprocess)
    rgb_backbone = _resolve(ckpt_cfg["rgb_backbone_checkpoint"])
    fusion_ckpt = _resolve(ckpt_cfg.get("fusion_module_path", "checkpoints/uff_pretrain.pth"))
    if rgb_backbone is None or not rgb_backbone.exists():
        raise FileNotFoundError(f"Missing RGB backbone checkpoint: {rgb_backbone}")

    command = [
        sys.executable,
        "main.py",
        "--method_name",
        str(run_cfg["method_name"]),
        "--dataset_type",
        str(run_cfg["dataset_type"]),
        "--dataset_path",
        str(dataset_path),
        "--rgb_backbone_checkpoint",
        str(rgb_backbone),
        "--fusion_module_path",
        str(fusion_ckpt),
        "--save_feature_path",
        str(feature_cache),
        "--results_dir",
        str(_resolve(outputs_cfg["results_dir"])),
        "--max_sample",
        str(run_cfg["max_sample"]),
        "--memory_bank",
        str(run_cfg["memory_bank"]),
        "--img_size",
        str(run_cfg["img_size"]),
        "--rgb_backbone_name",
        str(run_cfg["rgb_backbone_name"]),
        "--group_size",
        str(run_cfg["group_size"]),
        "--num_group",
        str(run_cfg["num_group"]),
        "--xyz_s_lambda",
        str(run_cfg["xyz_s_lambda"]),
        "--xyz_smap_lambda",
        str(run_cfg["xyz_smap_lambda"]),
        "--rgb_s_lambda",
        str(run_cfg["rgb_s_lambda"]),
        "--rgb_smap_lambda",
        str(run_cfg["rgb_smap_lambda"]),
        "--fusion_s_lambda",
        str(run_cfg["fusion_s_lambda"]),
        "--fusion_smap_lambda",
        str(run_cfg["fusion_smap_lambda"]),
        "--coreset_eps",
        str(run_cfg["coreset_eps"]),
        "--f_coreset",
        str(run_cfg["f_coreset"]),
        "--save_feature",
        "--build_memory_bank_only",
    ]
    _run(command, log_path)


if __name__ == "__main__":
    main()
