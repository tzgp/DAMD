from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run(command: list[str], log_path: Path) -> None:
    print("[RUN]", " ".join(command))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        process = subprocess.run(command, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=True)
        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(command)}")


def _preprocess(dataset_cfg: dict, log_path: Path, skip_preprocess: bool) -> Path:
    dataset_name = dataset_cfg["name"]
    raw_path = _resolve(dataset_cfg.get("raw_path"))
    processed_path = _resolve(dataset_cfg.get("processed_path"))
    if dataset_name == "mvtec3d" and "DAMD_MVTEC3D_ROOT" in os.environ:
        raw_path = Path(os.environ["DAMD_MVTEC3D_ROOT"]).expanduser().resolve()
        processed_path = raw_path
    if dataset_name == "eyecandies":
        if "DAMD_EYECANDIES_ROOT" in os.environ:
            raw_path = Path(os.environ["DAMD_EYECANDIES_ROOT"]).expanduser().resolve()
        if "DAMD_EYECANDIES_PREPROCESSED_ROOT" in os.environ:
            processed_path = Path(os.environ["DAMD_EYECANDIES_PREPROCESSED_ROOT"]).expanduser().resolve()
    if processed_path is None:
        raise ValueError("processed_path is required in the reproduction config")
    if skip_preprocess or not dataset_cfg.get("preprocess", True):
        return processed_path
    if dataset_name == "mvtec3d":
        _run([sys.executable, "utils/preprocessing.py", str(processed_path)], log_path)
        return processed_path
    if dataset_name == "eyecandies":
        if raw_path is None:
            raise ValueError("raw_path is required for Eyecandies preprocessing")
        _run([
            sys.executable,
            "utils/preprocess_eyecandies.py",
            "--dataset_path",
            str(raw_path),
            "--target_dir",
            str(processed_path),
        ], log_path)
        return processed_path
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the DAMD reproduction pipeline from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file under configs/.")
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    config_path = _resolve(args.config)
    if config_path is None or not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    config = _load_config(config_path)

    dataset_cfg = config["dataset"]
    outputs_cfg = config["outputs"]
    run_cfg = config["run"]
    ckpt_cfg = config["checkpoints"]

    log_path = _resolve(outputs_cfg["log_path"])
    results_dir = _resolve(outputs_cfg["results_dir"])
    feature_cache = _resolve(outputs_cfg["save_feature_path"])
    assert log_path is not None and results_dir is not None and feature_cache is not None
    results_dir.mkdir(parents=True, exist_ok=True)
    feature_cache.mkdir(parents=True, exist_ok=True)

    dataset_path = _preprocess(dataset_cfg, log_path, args.skip_preprocess)
    rgb_backbone = _resolve(ckpt_cfg["rgb_backbone_checkpoint"])
    fusion_ckpt = _resolve(ckpt_cfg["fusion_module_path"])
    if rgb_backbone is None or not rgb_backbone.exists():
        raise FileNotFoundError(f"Missing RGB backbone checkpoint: {rgb_backbone}")

    command = [
        sys.executable,
        "main.py",
        "--method_name", str(run_cfg["method_name"]),
        "--dataset_type", str(run_cfg["dataset_type"]),
        "--dataset_path", str(dataset_path),
        "--rgb_backbone_checkpoint", str(rgb_backbone),
        "--fusion_module_path", str(fusion_ckpt),
        "--save_feature_path", str(feature_cache),
        "--results_dir", str(results_dir),
        "--max_sample", str(run_cfg["max_sample"]),
        "--memory_bank", str(run_cfg["memory_bank"]),
        "--img_size", str(run_cfg["img_size"]),
        "--rgb_backbone_name", str(run_cfg["rgb_backbone_name"]),
        "--xyz_backbone_name", str(run_cfg["xyz_backbone_name"]),
        "--group_size", str(run_cfg["group_size"]),
        "--num_group", str(run_cfg["num_group"]),
        "--xyz_s_lambda", str(run_cfg["xyz_s_lambda"]),
        "--xyz_smap_lambda", str(run_cfg["xyz_smap_lambda"]),
        "--rgb_s_lambda", str(run_cfg["rgb_s_lambda"]),
        "--rgb_smap_lambda", str(run_cfg["rgb_smap_lambda"]),
        "--fusion_s_lambda", str(run_cfg["fusion_s_lambda"]),
        "--fusion_smap_lambda", str(run_cfg["fusion_smap_lambda"]),
        "--coreset_eps", str(run_cfg["coreset_eps"]),
        "--f_coreset", str(run_cfg["f_coreset"]),
        "--ocsvm_nu", str(run_cfg["ocsvm_nu"]),
        "--ocsvm_maxiter", str(run_cfg["ocsvm_maxiter"]),
    ]
    if run_cfg.get("use_uff"):
        command.append("--use_uff")
    if run_cfg.get("use_trans", True):
        command.append("--use_trans")
    if run_cfg.get("use_ssl"):
        command.append("--use_ssl")
    if run_cfg.get("save_preds", True):
        command.append("--save_preds")

    _run(command, log_path)


if __name__ == "__main__":
    main()
