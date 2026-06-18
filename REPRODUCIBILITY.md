# Reproducibility guide

This document collects the exact materials needed to reproduce the DAMD results on MVTec 3D-AD and Eyecandies.

## 1. Environment

- Preferred: `conda env create -f environment.yml`
- Exact pip snapshot: `requirements-lock.txt`
- Original run interpreter: `/home/c1/zgp/miniconda/envs/damd/bin/python`

## 2. Required external assets

### Datasets

- MVTec 3D-AD: official website, original licensing retained by MVTec.
- Eyecandies: official website, original licensing retained by Eyecandies.
- The training/testing split protocol follows the same setup used in M3DM for the corresponding public dataset releases.

### Checkpoints

- `checkpoints/dino_vitbase8_pretrain.pth`
- `checkpoints/uff_pretrain.pth` when UFF is enabled

See `checkpoints/README.md` for hashes and provenance.

## 3. Config files

- `configs/mvtec3d_reproduction.yaml`
- `configs/eyecandies_reproduction.yaml`

These files record the dataset target, checkpoints, output directories, and the required hyperparameters used in the main DAMD evaluation path.

## 4. One-command reproduction

### MVTec 3D-AD

```bash
bash scripts/reproduce_mvtec3d.sh /path/to/mvtec3d
```

### Eyecandies

```bash
bash scripts/reproduce_eyecandies.sh /path/to/Eyecandies /path/to/eyecandies_preprocessed
```

## 5. Exact evaluation entrypoint

```bash
python main.py \
  --method_name DINO+FPFH+add+late \
  --dataset_type mvtec3d \
  --dataset_path /path/to/mvtec3d \
  --rgb_backbone_checkpoint checkpoints/dino_vitbase8_pretrain.pth \
  --results_dir results/mvtec3d
```

## 6. Revision-only analysis scripts

- `experiments/revision/run_eaf_analysis.py`
- `experiments/revision/run_damd_efficiency_analysis.py`
- `experiments/revision/robustness/run_robustness.py`
- `experiments/revision/robustness/run_supplementary.py`

The path defaults in these scripts have been converted from machine-specific absolute paths to repository-relative or environment-overridable defaults.

## 7. Artifact validation

Run:

```bash
python scripts/check_reproducibility.py
```

This checks that the expected reproducibility documents, configs, scripts, and archival metadata files are present.

## 8. Zenodo DOI release steps

The repository now contains release metadata, but DOI minting is still a manual account-level step:

1. Link the GitHub repository to Zenodo.
2. Create a GitHub release (for example `v1.0.0`).
3. Let Zenodo archive the release and mint the DOI.
4. Add the DOI to the README and manuscript availability statement.

## 9. Reproducibility checklist

- [x] Public code repository
- [x] Configuration files
- [x] Evaluation entrypoint
- [x] Preprocessing scripts
- [x] Exact environment snapshot
- [x] Checkpoint manifest and checksums
- [x] One-command reproduction wrappers
- [x] Reviewer-facing summary file
- [ ] DOI minted and inserted into manuscript after release
