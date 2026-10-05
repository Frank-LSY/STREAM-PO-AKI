#!/usr/bin/env bash
# Baseline eval for event-stream models.
# Usage: eval_baseline.sh ARCH [GPU] [plot_cal]
#   GPU is ignored when multiple idle GPUs exist (auto-assigned per fold).
#   plot_cal: 1 (default) = calibration plots; 0 = skip.
# Multiple idle GPUs: folds in parallel (wait until all finish); one GPU: sequential.
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ARCH="${1:?Usage: eval_baseline.sh ARCH [gpu] [plot_cal]}"
GPU="${2:-0}"
PLOT_CAL="${3:-1}"

run_fold() {
  local fold="$1" device="$2" log="$3"
  local gpu_idx="${device#cuda:}"
  {
    echo "Event-stream ${ARCH} baseline eval fold ${fold} device=${device}"
    run_event_stream_baseline_eval "${ARCH}" "${fold}" "${gpu_idx}" "${PLOT_CAL}"
    echo "Done: $(event_stream_eval_dir "${ARCH}" baseline "${fold}")/baseline_metrics.json"
  } > "${log}" 2>&1
}

echo "ARCH=${ARCH} plot_cal=${PLOT_CAL} (requested_gpu=${GPU}, idle policy overrides when multi-GPU)"
run_all_folds_with_gpu_policy "eval_event_stream_${ARCH}_baseline" run_fold 1

SUMMARY="${RESULT_DIR}/event_stream/${ARCH}/eval_baseline_nocross_5fold_summary.csv"
aggregate_fold_summaries \
  "event_stream/${ARCH}/eval_baseline_fold{fold}_test" \
  "${ARCH}_baseline_eval_nocross_fold{fold}_test" \
  "${SUMMARY}"
echo "Wrote ${SUMMARY}"
echo "Per-fold logs: logs/eval_event_stream_${ARCH}_baseline_fold{1..5}.log"
