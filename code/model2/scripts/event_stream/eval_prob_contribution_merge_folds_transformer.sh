#!/usr/bin/env bash
# Merge per-fold event probability contribution summaries (after all folds finish).
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

MERGED_OUT="${RESULT_DIR}/event_stream/transformer/eval_prob_contribution_merged_test"
fold_dirs=()
for fold in {1..5}; do
  fold_dirs+=("${RESULT_DIR}/event_stream/transformer/eval_prob_contribution_fold${fold}_test")
done

echo "Merge across folds -> ${MERGED_OUT}"
"${PYTHON}" "${EVENT_STREAM_DIR}/aggregate_probability_contribution_folds.py" \
  --fold_dirs "${fold_dirs[@]}" \
  --output_dir "${MERGED_OUT}" \
  --min_count 10 \
  --top_k 50 \
  --top_k_report 5

echo "Merged top-5:"
echo "  ${MERGED_OUT}/top5_risk_increasing_events.csv"
echo "  ${MERGED_OUT}/top5_risk_decreasing_events.csv"
