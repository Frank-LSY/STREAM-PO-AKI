#!/usr/bin/env bash
# Run event probability contribution for one fold.
# Usage: eval_prob_contribution_transformer.sh [fold] [max_ep] [gpu]
#   gpu: physical GPU index (0..N) or omit for fold-1 mapping.
# Output is written to logs/evaluate_event_probability_contribution_fold{FOLD}.log
# when invoked from run_fold (stdout redirected by caller).
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

FOLD="${1:?fold required}"
MAX_EP="${2:-0}"
GPU="${3:-$((FOLD - 1))}"
OUT="${RESULT_DIR}/event_stream/transformer/eval_prob_contribution_fold${FOLD}_test"
LOG="$(log_file "evaluate_event_probability_contribution_fold${FOLD}.log")"

echo "Event contribution eval fold ${FOLD} -> ${OUT} (device=cuda:${GPU}, log=${LOG})"
"${PYTHON}" "${EVENT_STREAM_DIR}/evaluate_event_probability_contribution.py" \
  --checkpoint "$(event_stream_best_ckpt transformer "${FOLD}")" \
  --episode_path "${TEST_JSONL}" \
  --output_dir "${OUT}" \
  --device "$(device_cuda "${GPU}")" \
  --max_episodes "${MAX_EP}" \
  --max_preop_occlusions 60 \
  --max_postop_query_steps 25 \
  --max_events_per_query 60 \
  --lookback_hours 72 \
  --min_count 10 \
  --top_k 50 \
  --top_k_report 5 \
  --plot_top

echo "Done fold ${FOLD}. Top-5 CSVs:"
echo "  ${OUT}/top5_risk_increasing_events.csv"
echo "  ${OUT}/top5_risk_decreasing_events.csv"
