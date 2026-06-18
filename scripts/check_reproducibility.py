from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "REPRODUCIBILITY.md",
    "REVIEWER_RESPONSE_AND_WORK_SUMMARY.md",
    "environment.yml",
    "requirements-lock.txt",
    "configs/mvtec3d_reproduction.yaml",
    "configs/eyecandies_reproduction.yaml",
    "scripts/run_reproduction.py",
    "scripts/build_memory_bank.py",
    "scripts/reproduce_mvtec3d.sh",
    "scripts/reproduce_eyecandies.sh",
    "checkpoints/README.md",
    "checkpoints/checksums.sha256",
    "CITATION.cff",
    ".zenodo.json",
]


def main() -> int:
    missing = []
    for relative in REQUIRED_FILES:
        path = REPO_ROOT / relative
        if not path.exists():
            missing.append(relative)
    if missing:
        print("Missing reproducibility artifacts:")
        for item in missing:
            print(f"- {item}")
        return 1
    print("All required reproducibility artifacts are present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
