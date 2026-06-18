# DAMD revision work summary and reviewer-response draft

## 1. What was done in this repository update

The reviewer requests were processed in order with a focus on reproducibility and archival readiness.

1. **Strengthened repository-level reproducibility materials**
   - Rewrote `README.md` so that code, configs, evaluation scripts, revision analyses, and checkpoint placement are clearly documented.
   - Added `REPRODUCIBILITY.md` with a concrete reproduction checklist and exact run commands.

2. **Added exact environment capture**
   - Added `environment.yml`.
   - Added `requirements-lock.txt` from the original DAMD server environment (`python 3.8.20`, `torch 1.10.0`, `torchvision 0.11.1`, `timm 0.9.12`, `open3d 0.10.0.0`, etc.).

3. **Added configuration files and one-command pipelines**
   - Added `configs/mvtec3d_reproduction.yaml`.
   - Added `configs/eyecandies_reproduction.yaml`.
   - Added `scripts/run_reproduction.py`.
   - Added `scripts/reproduce_mvtec3d.sh` and `scripts/reproduce_eyecandies.sh`.

4. **Removed machine-specific defaults from the main public entrypoints**
   - `main.py` now defaults to repository-relative or environment-overridable paths instead of `/home/...` paths.
   - `models/models.py` now actually respects the passed RGB checkpoint path and raises a clear error if it is missing.
   - `feature_extractors/features.py` now passes the RGB checkpoint path through explicitly.
   - `utils/preprocess_eyecandies.py` now uses portable defaults and idempotent directory creation.

5. **Improved revision-script portability**
   - Updated the main revision-analysis scripts under `experiments/revision/` and the qualitative EAF figure generator to remove hard-coded absolute server paths.

6. **Added checkpoint manifest and archival metadata**
   - Added `checkpoints/README.md`.
   - Added `checkpoints/checksums.sha256`.
   - Added `CITATION.cff` and `.zenodo.json` for GitHub/Zenodo archival release preparation.

7. **Added artifact validation support**
   - Added `scripts/check_reproducibility.py` to verify that the expected reproducibility files are present.

8. **Cleaned targeted runtime debug output and reran full MVTec validation**
   - Removed the requested runtime prints for density arrays, entropy values, weight normalization, and fusion score weighting from `feature_extractors/selfmultiple_featureslate.py`.
   - Re-ran the active DAMD path on the original full MVTec 3D-AD dataset in the original `damd` environment.
   - Observed full-run aggregate results: `I-AUROC = 0.954`, `P-AUROC = 0.993`, `AU-PRO = 0.969`.

9. **Added a public memory-bank construction script**
   - Added `scripts/build_memory_bank.py` as a public wrapper around the existing DAMD feature-saving path.
   - The script reuses the same config-driven setup and constructs training memory-bank artifacts without requiring a full evaluation run.

## 2. What still requires manual action

The following items cannot be completed automatically from the server alone and still require maintainer action:

1. **Zenodo DOI minting**
   - Link the GitHub repository to Zenodo.
   - Create a GitHub release.
   - Wait for Zenodo to mint the DOI.
   - Insert the real DOI into the manuscript and README.

2. **Public hosting of checkpoints / release bundles**
   - If `dino_vitbase8_pretrain.pth` and any author-trained checkpoints are to be redistributed, they should be uploaded to a stable public location such as Zenodo or the GitHub release assets.

3. **Manuscript text synchronization**
   - The paper text should be updated to reflect the exact final public URLs and DOI once the archival release is published.

4. **Publicly reachable checkpoints / release assets**
   - The checkpoint manifest and checksums are now present, but reviewers will still need publicly reachable files or release assets.
   - If the intended checkpoints are not yet attached to GitHub Releases or Zenodo, this remains the main reproducibility blocker.

5. **Expected-results matrix**
   - For strongest reviewer confidence, add a compact table in the README or repository docs that maps each reproduction command/config to expected output files and target metrics or acceptable tolerance ranges.

## 3. Suggested manuscript wording

### 3.1 Abstract

> Code, configuration files, evaluation scripts, and reproducibility materials are publicly available at the project repository.

After the DOI is minted, this can be strengthened to:

> Code, configuration files, evaluation scripts, and reproducibility materials are publicly available at the project repository and archived through Zenodo.

### 3.2 Data and code availability statement

> **Code and Data Availability.** The source code for DAMD, experimental configuration files, evaluation scripts, dataset preprocessing scripts, and revision-analysis materials are publicly available at `https://github.com/tzgp/DAMD`. The repository includes one-command reproduction wrappers for MVTec 3D-AD and Eyecandies, an exact environment file, and a checkpoint manifest with checksums. MVTec 3D-AD and Eyecandies remain available from their respective official sources and are not redistributed in this repository because they remain subject to their original access conditions and licenses. A DOI-backed archival release will be provided through Zenodo after the public release is minted.

After the DOI is minted, replace the last sentence with:

> A DOI-backed archival release is available through Zenodo at `DOI_PLACEHOLDER`.

### 3.3 “Why DAMD is useful” paragraph for Introduction or Conclusion

> DAMD is useful for industrial anomaly detection because it remains lightweight, avoids dependence on a heavy pretrained 3D backbone in the main evaluation setting, improves subtle defect detection on public RGB-3D benchmarks, provides interpretable modality-weight behavior through EAF-based analysis, and is reproducible on public datasets using released code, scripts, and configuration files.

## 4. Draft response to the reviewer

> **Response:** Thank you for the constructive suggestion. We have substantially strengthened the public reproducibility package. Specifically, we reorganized the repository to include a clearer README, exact environment files, configuration files for both MVTec 3D-AD and Eyecandies, dataset preprocessing scripts, one-command reproduction wrappers, a checkpoint manifest with checksums, and revision-analysis scripts/materials. We also removed machine-specific absolute paths from the main public entrypoints and added archival metadata files (`CITATION.cff` and `.zenodo.json`) so that the repository is ready for DOI-based archival release. In the revised manuscript, we will explicitly state that the code, configuration files, evaluation scripts, and reproducibility materials are publicly available, and we will update the availability statement with the Zenodo DOI once the archival release is minted.

> **Response on train/test splitting:** We also clarified that the training/testing split protocol follows the same setup used in M3DM for the corresponding public dataset releases, so that the evaluation setting is explicit and directly comparable.

> **Response on verification scope:** To avoid overclaiming, we have described the current repository update as a strengthened reproducibility package with validated entry points, portable defaults, and explicit environment/configuration support. We do not claim in the repository text that every manuscript result has already been rerun from a clean public environment unless and until those end-to-end verification logs are also released.

> **Response regarding reuse value:** We also agree that the reuse value of the paper should be made clearer. To address this, we prepared wording that highlights that DAMD is lightweight, does not rely on a heavy pretrained 3D backbone in the main setting, improves subtle defect detection, provides interpretable modality weights, and is reproducible on public datasets. These points can be incorporated into the Introduction or Conclusion in the revised manuscript.

> **Response on implementation cleanup and verification:** We additionally removed targeted runtime debug prints that exposed density arrays, entropy values, and fusion-weight outputs in the active DAMD evaluation path, and then reran the full MVTec 3D-AD evaluation in the original DAMD environment on the full dataset. The run completed successfully across all 10 categories, confirming that the cleanup did not break the active evaluation pipeline.

> **Response on public memory-bank construction:** Following the reviewer’s suggestion, we also added a public memory-bank construction script to the repository. This script exposes the DAMD training feature-cache generation path in a standalone, documented form, so that users can construct the published memory-bank artifacts directly from the released code and configuration files.

## 5. Recommended next manual steps

1. Push the updated repository contents to `https://github.com/tzgp/DAMD`.
2. Upload or attach the intended public checkpoints.
3. Create a GitHub release and connect it to Zenodo.
4. Replace DOI placeholders with the real DOI.
5. Copy the wording above into the manuscript and response letter.
6. If desired, do one more cosmetic cleanup pass for remaining shape/type debug prints that were outside the explicit density/entropy/weight request.

## 6. Final wording cautions

- Prefer **"reproducibility materials and reproduction scripts are provided"** over **"fully reproducible"** unless full clean-environment reruns are also documented.
- Prefer **"command-line entry points and reproducibility checks were validated"** over **"all experiments verified"**.
- Prefer **"reviewed machine-specific defaults were replaced"** over **"all paths are now portable"**, unless every auxiliary script has been audited.
