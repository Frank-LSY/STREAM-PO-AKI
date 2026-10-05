#!/usr/bin/env bash
# Usage: train_all_folds.sh ARCH [epochs] [batch_size] [extra train.py args...]
# 1 GPU: sequential; multiple GPUs: parallel (one job per fold).
# Per-fold logs: logs/train_timeseries_${ARCH}_fold{1..5}.log
# Launcher messages only go to stdout (safe to nohup to a small *_launcher.log).
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ARCH="${1:?ARCH required (transformer|lstm|grud)}"
EPOCHS="${2:-${BUCKET_TRAIN_EPOCHS:-100}}"
BATCH_SIZE="${3:-}"
if [[ -z "${BATCH_SIZE}" ]]; then
  case "${ARCH}" in
    lstm) BATCH_SIZE="${BUCKET_LSTM_BATCH_SIZE}" ;;
    grud) BATCH_SIZE="${BUCKET_GRUD_BATCH_SIZE}" ;;
    *) BATCH_SIZE="${BUCKET_TRANSFORMER_BATCH_SIZE}" ;;
  esac
fi
# Remaining positional args are forwarded verbatim to train.py.
shift $(( $# > 3 ? 3 : $# ))
EXTRA_ARGS=("$@")

echo "ARCH=${ARCH} epochs=${EPOCHS} batch_size=${BATCH_SIZE} extra=${EXTRA_ARGS[*]-}"
echo "Per-fold training logs: ${LOG_DIR}/train_timeseries_${ARCH}_fold{1..5}.log"

run_fold() {
  local fold="$1" device="$2" log="$3"
  train_timeseries_fold "${ARCH}" "${fold}" "${EPOCHS}" "${BATCH_SIZE}" "${device}" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "${log}" 2>&1
}

# TS_WAIT_JOBS=1 blocks until all folds finish, so callers can chain archs serially.
# Concurrent archs contend for host memory and stall each other, so chain rather than co-launch.
run_all_folds_with_gpu_policy "train_timeseries_${ARCH}" run_fold "${TS_WAIT_JOBS:-0}"
