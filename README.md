# DAMD: Density-Augmented RGB-3D Fusion for Subtle Industrial Defect Detection

This repository contains the official implementation used in the paper under review at *The Visual Computer*.

Code, evaluation scripts, configuration files, and reproducibility documentation are publicly available in this repository. Zenodo archival metadata is provided via `CITATION.cff` and `.zenodo.json`; after a GitHub release is created, a DOI-backed archival snapshot can be minted and cited in the manuscript.

## What is included

- `main.py`: primary evaluation entrypoint for DAMD.
- `configs/`: reproducibility configs for MVTec 3D-AD and Eyecandies.
- `scripts/run_reproduction.py`: one-command reproduction wrapper.
- `scripts/build_memory_bank.py`: public memory-bank construction wrapper.
- `scripts/reproduce_mvtec3d.sh` and `scripts/reproduce_eyecandies.sh`: dataset-specific wrappers.
- `REPRODUCIBILITY.md`: exact reproduction workflow and checklist.
- `experiments/revision/`: scripts for revision experiments (EAF analysis, robustness, efficiency).
- `checkpoints/README.md`: checkpoint manifest and placement rules.

 

## Environment setup

We recommend reproducing the experiments with the exact environment captured from the original server.

### Option A: Conda environment (recommended)

```bash
conda env create -f environment.yml
conda activate damd
```

### Option B: Pip lockfile

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
```

The original experiments were run with Python 3.8.20, torch 1.10.0, torchvision 0.11.1, timm 0.9.12, and open3d 0.10.0.0.

## Datasets

### MVTec 3D-AD

- Official source: <https://www.mvtec.com/company/research/datasets/mvtec-3d-ad>
- DAMD uses the same training/testing split protocol as M3DM on the official MVTec 3D-AD categories.
- Preprocessing is performed in place:

```bash
python utils/preprocessing.py /path/to/mvtec3d
```

### Eyecandies

- Official source: <https://eyecan-ai.github.io/eyecandies/>
- DAMD uses the same training/testing split protocol as M3DM on the official Eyecandies release.
- Convert the raw release into the DAMD directory layout with:

```bash
python utils/preprocess_eyecandies.py \
  --dataset_path /path/to/Eyecandies \
  --target_dir /path/to/eyecandies_preprocessed
```

The datasets are not redistributed here because they remain subject to their original licenses and access conditions.

## Checkpoints

Place the required checkpoints under `checkpoints/`:

- `checkpoints/dino_vitbase8_pretrain.pth`
- `checkpoints/uff_pretrain.pth` (only required when UFF-based runs are enabled)

See `checkpoints/README.md` and `checkpoints/checksums.sha256` for provenance and checksum details.

## One-command reproduction

### MVTec 3D-AD

```bash
bash scripts/reproduce_mvtec3d.sh /path/to/mvtec3d
```

### Eyecandies

```bash
bash scripts/reproduce_eyecandies.sh /path/to/Eyecandies /path/to/eyecandies_preprocessed
```

These wrappers validate required paths, optionally preprocess the data, and launch the exact config-backed pipeline:

```bash
python scripts/run_reproduction.py --config configs/mvtec3d_reproduction.yaml
python scripts/run_reproduction.py --config configs/eyecandies_reproduction.yaml
```

## Memory-bank construction

To publicly expose the training memory-bank construction path, the repository provides a dedicated wrapper that reuses the same config and feature-extraction pipeline while skipping evaluation:

```bash
python scripts/build_memory_bank.py --config configs/mvtec3d_reproduction.yaml --skip-preprocess
```

This command saves per-sample training feature tensors under the configured `save_feature_path` (for example `outputs/feature_cache/mvtec3d`) and is intended as the public memory-bank construction artifact for reproducibility.

## Main evaluation command

The primary paper setting is:

```bash
python main.py \
  --method_name DINO+FPFH+add+late \
  --dataset_type mvtec3d \
  --dataset_path /path/to/mvtec3d \
  --rgb_backbone_checkpoint checkpoints/dino_vitbase8_pretrain.pth \
  --results_dir results/mvtec3d
```

For Eyecandies, replace `--dataset_type` and `--dataset_path` accordingly.

## Reproducibility artifacts

This repository includes:

- exact environment files (`environment.yml`, `requirements-lock.txt`)
- reproducibility configs (`configs/*.yaml`)
- dataset preprocessing scripts (`utils/preprocessing.py`, `utils/preprocess_eyecandies.py`)
- evaluation entrypoint (`main.py`)
- revision-analysis scripts (`experiments/revision/`)
- checkpoint manifest and checksums (`checkpoints/`)
- reproducibility checklist and reviewer-facing summary (`REPRODUCIBILITY.md`, `REVIEWER_RESPONSE_AND_WORK_SUMMARY.md`)

## Revision analysis scripts

Revision-specific analyses are maintained under `experiments/revision/`. Generated outputs are created at runtime and are not tracked as part of the repository snapshot.

## Citation and archival release

The repository includes release metadata for Zenodo and GitHub citation support:

- `CITATION.cff`
- `.zenodo.json`

After creating a GitHub release, Zenodo can mint a DOI-backed archival snapshot. The DOI should then be added to the manuscript and README.

## Acknowledgements

This repository builds on ideas and components from [3D-ADS](https://github.com/eliahuhorwitz/3D-ADS) and [M3DM](https://github.com/nomewang/M3DM). We thank the original authors for releasing their code.
