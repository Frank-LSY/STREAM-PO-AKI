#!/usr/bin/env bash
# Fixed 6 h patient-window grid eval (0–720 h) for the full model and every ablation, 5 folds.
# Usage: run_fixedgrid_ablation.sh [LANES_PER_GPU] [NUM_GPUS]
set -euo pipefail

LANES_PER_GPU="${1:-2}"
NUM_GPUS="${2:-8}"
ROOT="${PERIOP_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
PY=python
TEST_JSONL=${ROOT}/data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl
LOG_DIR=${ROOT}/result/event_stream/ablation/logs_fixedgrid
mkdir -p "${LOG_DIR}"

RUNS=(
  "Full:${ROOT}/result/event_stream/transformer"
  "A1:${ROOT}/result/event_stream/ablation/A1_no_time_encoding"
  "A2:${ROOT}/result/event_stream/ablation/A2_causal_attention"
  "A3:${ROOT}/result/event_stream/ablation/A3_procedure_as_base"
  "A4:${ROOT}/result/event_stream/ablation/A4_no_baseline_loss"
  "A5:${ROOT}/result/event_stream/ablation/A5_no_mem"
  "I1:${ROOT}/result/event_stream/ablation/I1_no_med_events"
  "I2:${ROOT}/result/event_stream/ablation/I2_no_lab_events"
  "I3:${ROOT}/result/event_stream/ablation/I3_no_conditions"
  "I4:${ROOT}/result/event_stream/ablation/I4_generic_surgery"
  "A3I4:${ROOT}/result/event_stream/ablation/A3I4_procedure_as_base_generic_surgery"
)

JOBS=()
for run in "${RUNS[@]}"; do
  for k in 1 2 3 4 5; do
    JOBS+=("${run}:${k}")
  done
done

n_lanes=$((LANES_PER_GPU * NUM_GPUS))
for ((lane = 0; lane < n_lanes; lane++)); do
  gpu=$((lane % NUM_GPUS))
  (
    for ((j = lane; j < ${#JOBS[@]}; j += n_lanes)); do
      IFS=: read -r id dir k <<< "${JOBS[$j]}"
      out="${dir}/eval_fixedgrid_fold${k}_test"
      log="${LOG_DIR}/${id}_fold${k}.log"
      if [[ -f "${out}/fixed_time_grid_metrics.json" && "${out}/fixed_time_grid_metrics.json" -nt "${dir}/fold${k}/best_model.pt" && "${FORCE:-0}" != 1 ]]; then
        echo "skip ${id} fold ${k}" >> "${log}"
        continue
      fi
      cd "${ROOT}/code/model2"
      PYTHONPATH=. "${PY}" event_stream/evaluate_fixed_time_grid.py \
        --checkpoint "${dir}/fold${k}/best_model.pt" \
        --episode_path "${TEST_JSONL}" \
        --output_dir "${out}" \
        --device "cuda:${gpu}" \
        --no_progress --export_csv > "${log}" 2>&1 \
        && echo "done ${id} fold ${k}" >> "${LOG_DIR}/progress.txt" \
        || echo "FAIL ${id} fold ${k}" >> "${LOG_DIR}/progress.txt"
    done
  ) &
done
wait
echo "all fixed-grid jobs finished" >> "${LOG_DIR}/progress.txt"
