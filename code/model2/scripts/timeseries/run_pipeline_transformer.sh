#!/usr/bin/env bash
# Full timeseries Transformer pipeline.
# Usage: run_bucket_ts6h_pipeline.sh [epochs] [gpu]
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

EPOCHS="${1:-10}"
bind_gpu "${2:-1}" >/dev/null

STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="${LOG_DIR}/timeseries_pipeline_${STAMP}.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1
echo "=== timeseries pipeline start $(date -Is) ==="

require_ts6h_data
echo "[1/3] dataset ready: ${TEST_NPZ}"

echo "[2/3] train + baseline eval"
bash "${SCRIPT_DIR}/train_eval_transformer.sh" "${EPOCHS}"

echo "[3/3] grid eval"
bash "${SCRIPT_DIR}/eval_grid_transformer.sh" 0

echo "=== done $(date -Is) ==="
