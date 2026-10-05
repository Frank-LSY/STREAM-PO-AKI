#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

GPU="${1:-0}"
bind_gpu "${GPU}" >/dev/null

"${PYTHON}" "${TIMESERIES_DIR}/check_bucket_grid_predictions.py" \
  --result_root "${RESULT_DIR}" \
  --output_dir "${RESULT_DIR}/timeseries/transformer/grid_check" \
  --test_npz "${TEST_NPZ}" \
  --test_jsonl "${TEST_JSONL}" \
  --device cuda:0 \
  --batch_size 64 \
  --dpi 150

echo "Wrote ${RESULT_DIR}/timeseries/transformer/grid_check/"
