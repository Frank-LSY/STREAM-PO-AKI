#!/usr/bin/env bash
# Launch 5-fold event contribution eval in parallel on idle GPUs (background + logs).
# Usage: eval_prob_contribution_all_folds_transformer.sh [max_ep] [poll_sec]
#   max_ep: 0 = all test episodes (default)
#   poll_sec: seconds between merge readiness checks (default 120)
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

MAX_EP="${1:-0}"
POLL_SEC="${2:-120}"

run_fold() {
  local fold="$1" device="$2" log="$3"
  # device is cuda:N (physical GPU index)
  local gpu_idx="${device#cuda:}"
  bash "${SCRIPT_DIR}/eval_prob_contribution_transformer.sh" "${fold}" "${MAX_EP}" "${gpu_idx}" \
    > "${log}" 2>&1
}

run_all_folds_with_gpu_policy "evaluate_event_probability_contribution" run_fold

MERGE_LOG="$(log_file "evaluate_event_probability_contribution_merge.log")"
nohup bash "${SCRIPT_DIR}/eval_prob_contribution_wait_merge_transformer.sh" "${POLL_SEC}" \
  > "${MERGE_LOG}" 2>&1 &
disown

echo ""
echo "Background merge scheduled when all folds finish -> ${MERGE_LOG}"
echo "Per-fold logs: logs/evaluate_event_probability_contribution_fold{1..5}.log"
echo "Merged output: ${RESULT_DIR}/event_stream/transformer/eval_prob_contribution_merged_test/"
