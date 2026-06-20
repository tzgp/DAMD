# Checkpoint

The backbone checkpoint is also available from the upstream M3DM ecosystem, while the public DAMD release assets below provide the exact files used for this repository.
## Required files

1. `dino_vitbase8_pretrain.pth`
   - Purpose: RGB backbone initialization for `vit_base_patch8_224_dino`
   - Current source used by the original server run: local copy originally referenced from the M3DM-style setup
   - SHA-256: `575f0efc4838938314afa09897a62ed2b87919928b5edcd133abb907328995eb`
   - Public release URL: <https://github.com/tzgp/DAMD/releases/download/v1.0.0/dino_vitbase8_pretrain.pth>

2. `uff_pretrain.pth`
   - Purpose: UFF fusion checkpoint for runs that enable `--use_uff`
   - SHA-256: `8a827aee483371955835f56b460292e0b415ef8a17513251e3fb24f5d0f9324d`
   - Public release URL: <https://github.com/tzgp/DAMD/releases/download/v1.0.0/uff_pretrain.pth>

## Placement

Copy or download the files into `checkpoints/` so that the following paths exist:

- `checkpoints/dino_vitbase8_pretrain.pth`
- `checkpoints/uff_pretrain.pth`

The code now resolves these paths relative to the repository root by default. You may also override them with:

- `--rgb_backbone_checkpoint`
- `--fusion_module_path`
- `DAMD_RGB_BACKBONE_CHECKPOINT`
- `DAMD_FUSION_CHECKPOINT`

## Archival recommendation

The checkpoint binaries are now hosted through the GitHub release `v1.0.0`. Once a Zenodo DOI is minted, add the DOI-backed archival URL here and in the README as an additional persistent citation target.
