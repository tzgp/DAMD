# Checkpoint 

Download from https://github.com/nomewang/M3DM
## Required files

1. `dino_vitbase8_pretrain.pth`
   - Purpose: RGB backbone initialization for `vit_base_patch8_224_dino`
   - Current source used by the original server run: local copy originally referenced from the M3DM-style setup
   - SHA-256: `575f0efc4838938314afa09897a62ed2b87919928b5edcd133abb907328995eb`

2. `uff_pretrain.pth`
   - Purpose: UFF fusion checkpoint for runs that enable `--use_uff`
   - SHA-256: `8a827aee483371955835f56b460292e0b415ef8a17513251e3fb24f5d0f9324d`

## Placement

Copy or download the files into `checkpoints/` so that the following paths exist:

- `checkpoints/dino_vitbase8_pretrain.pth`
- `checkpoints/uff_pretrain.pth`
