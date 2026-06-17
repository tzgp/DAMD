#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'Usage: bash scripts/reproduce_mvtec3d.sh /path/to/mvtec3d [config]\n' >&2
  exit 1
fi

DATA_ROOT="$1"
CONFIG_PATH="${2:-configs/mvtec3d_reproduction.yaml}"
export DAMD_MVTEC3D_ROOT="$DATA_ROOT"

python scripts/run_reproduction.py --config "$CONFIG_PATH"
