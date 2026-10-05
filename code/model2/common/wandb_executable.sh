#!/usr/bin/env bash
# W&B service subprocess wrapper — runs the conda Python with NumPy 2.x patches applied.
MODEL2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REAL="${WANDB_REAL_PYTHON:-python}"
export PYTHONPATH="${MODEL2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${REAL}" "${MODEL2_ROOT}/common/wandb_executable.py" "$@"
