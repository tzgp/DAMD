#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  printf 'Usage: bash scripts/reproduce_eyecandies.sh /path/to/Eyecandies /path/to/eyecandies_preprocessed [config]\n' >&2
  exit 1
fi

RAW_ROOT="$1"
PROCESSED_ROOT="$2"
CONFIG_PATH="${3:-configs/eyecandies_reproduction.yaml}"
export DAMD_EYECANDIES_ROOT="$RAW_ROOT"
export DAMD_EYECANDIES_PREPROCESSED_ROOT="$PROCESSED_ROOT"

python scripts/run_reproduction.py --config "$CONFIG_PATH"
