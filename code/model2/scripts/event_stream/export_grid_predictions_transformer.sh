#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

MAX_EP="${1:-0}"
OUT_ROOT="${2:-${RESULT_DIR}/event_stream/transformer/baseline_grid_predictions}"
GPU="${3:-0}"

for fold in {1..5}; do
  out="${OUT_ROOT}/fold${fold}_test.jsonl"
  extra=()
  if [[ "${MAX_EP}" != "0" ]]; then
    extra+=(--max_episodes "${MAX_EP}")
  fi
  log="$(log_file "export_baseline_grid_fold${fold}.log")"
  nohup "${PYTHON}" "${EVENT_STREAM_DIR}/export_baseline_grid_predictions.py" \
    --checkpoint "$(event_stream_best_ckpt transformer "${fold}")" \
    --episode_path "${TEST_JSONL}" \
    --output_jsonl "${out}" \
    --device "$(device_cuda "${GPU}")" \
    --bucket_size 6 \
    --bucket_end 720 \
    "${extra[@]}" > "${log}" 2>&1 &
  echo "Launched fold ${fold} -> ${out} (log ${log})"
done
