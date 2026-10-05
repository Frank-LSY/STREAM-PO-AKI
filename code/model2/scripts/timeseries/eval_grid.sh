#!/usr/bin/env bash
# Grid eval for timeseries models. Usage: eval_grid.sh ARCH [gpu]
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ARCH="${1:?ARCH required (transformer|lstm)}"
GPU="${2:-0}"
bind_gpu "${GPU}" >/dev/null
require_ts6h_data

for fold in {1..5}; do
  ckpt="$(timeseries_best_ckpt "${ARCH}" "${fold}")"
  out="$(timeseries_grid_eval_dir "${ARCH}" "${fold}")"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skip fold ${fold}: missing ${ckpt}" >&2
    continue
  fi
  echo "Timeseries ${ARCH} grid eval fold ${fold} -> ${out}"
  "${PYTHON}" "${TIMESERIES_DIR}/evaluate_bucket_grid.py" \
    --checkpoint "${ckpt}" \
    --test_npz "${TEST_NPZ}" \
    --output_dir "${out}" \
    --batch_size "${BUCKET_EVAL_BATCH_SIZE}" \
    --device cuda:0 \
    --export_csv
done
