#!/usr/bin/env bash
# Poll until all 5 fold summaries exist, then merge.
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

POLL_SEC="${1:-120}"
MERGE_LOG="$(log_file "evaluate_event_probability_contribution_merge.log")"

exec > >(tee -a "${MERGE_LOG}") 2>&1

echo "Waiting for 5 fold summaries (poll every ${POLL_SEC}s) ..."
while true; do
  ready=0
  for fold in {1..5}; do
    summary="${RESULT_DIR}/event_stream/transformer/eval_prob_contribution_fold${fold}_test/event_probability_contribution_summary.json"
    if [[ -f "${summary}" ]]; then
      ready=$((ready + 1))
    fi
  done
  echo "  ${ready}/5 fold summaries ready"
  if [[ "${ready}" -ge 5 ]]; then
    break
  fi
  sleep "${POLL_SEC}"
done

bash "${SCRIPT_DIR}/eval_prob_contribution_merge_folds_transformer.sh"
echo "Merge complete."
