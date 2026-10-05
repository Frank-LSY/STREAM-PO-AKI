#!/usr/bin/env bash
# Usage: eval.sh ARCH EVAL_TYPE [GPU]
# EVAL_TYPE: postop6h | clinical | fixedgrid
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ARCH="${1:?ARCH required (transformer|lstm)}"
EVAL_TYPE="${2:?EVAL_TYPE required (postop6h|clinical|fixedgrid)}"
GPU="${3:-0}"

case "${EVAL_TYPE}" in
  postop6h)
    for fold in {1..5}; do
      run_event_stream_eval_fold "${ARCH}" postop6h "${fold}" "${GPU}" \
        --postop_hour_bin 6 --macro_min_n 10 --export_csv --print_table --cls_threshold 0.5
    done
    ;;
  clinical)
    for fold in {1..5}; do
      run_event_stream_eval_fold "${ARCH}" clinical "${fold}" "${GPU}" \
        --alert_thresholds "0.5,0.7,0.9" \
        --plot_dca --plot_calibration \
        --calib_jsonl "$(fold_train_jsonl "${fold}")"
    done
    ;;
  fixedgrid)
    for fold in {1..5}; do
      run_event_stream_eval_fold "${ARCH}" fixedgrid "${fold}" "${GPU}" \
        --bucket_size 6 --export_csv --print_table
    done
    ;;
  *)
    echo "Unknown EVAL_TYPE: ${EVAL_TYPE}" >&2
    exit 1
    ;;
esac
