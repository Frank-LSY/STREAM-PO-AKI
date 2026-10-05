#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

aggregate_fold_summaries \
  "baseline_eval_nocross_fold{fold}_test" \
  "baseline_eval_nocross_fold{fold}_test" \
  "${RESULT_DIR}/baseline_eval_nocross_5fold_summary.csv"
