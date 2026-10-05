#!/usr/bin/env bash
# Test-set baseline + postoperative 6h-bucket eval for all event-stream ablations.
#
# Usage:
#   ABLATION_GPUS=0,1,2,3,4,5,6,7 bash scripts/event_stream/eval_ablations.sh
#   bash scripts/event_stream/eval_ablations.sh A1 A3 I3   # subset by short id
#
# Outputs under result/event_stream/ablation/<id>/:
#   eval_baseline_fold{k}_test/baseline_metrics.json
#   eval_postop6h_fold{k}_test/metrics.json
#   eval_baseline_5fold_summary.csv
#   eval_postop6h_5fold_summary.csv
# Cross-ablation comparison:
#   result/event_stream/ablation/ablation_test_comparison.csv
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ABLATION_ROOT="${RESULT_DIR}/event_stream/ablation"
mkdir -p "${ABLATION_ROOT}" "${LOG_DIR}/ablation_eval"

declare -A ABLATION_DIRS=(
  [A1]=A1_no_time_encoding
  [A2]=A2_causal_attention
  [A3]=A3_procedure_as_base
  [A4]=A4_no_baseline_loss
  [A5]=A5_no_mem
  [I1]=I1_no_med_events
  [I2]=I2_no_lab_events
  [I3]=I3_no_conditions
  [I4]=I4_generic_surgery
  [A3I4]=A3I4_procedure_as_base_generic_surgery
)

ALL_SHORT=(A1 A2 A3 A4 A5 I1 I2 I3 I4 A3I4)

resolve_ids() {
  local -a out=()
  local a
  if [[ $# -eq 0 || "${1}" == "all" ]]; then
    out=("${ALL_SHORT[@]}")
  else
    for a in "$@"; do
      case "${a}" in
        A1|A2|A3|A4|A5|I1|I2|I3|I4|A3I4) ;;
        A1_no_time_encoding) a=A1 ;;
        A2_causal_attention) a=A2 ;;
        A3_procedure_as_base) a=A3 ;;
        A4_no_baseline_loss) a=A4 ;;
        A5_no_mem) a=A5 ;;
        I1_no_med_events|I1_no_med_exposure) a=I1 ;;
        I2_no_lab_events|I2_no_lab_values) a=I2 ;;
        I3_no_conditions) a=I3 ;;
        I4_generic_surgery) a=I4 ;;
        A3I4_procedure_as_base_generic_surgery|X1) a=A3I4 ;;
        *)
          echo "Unknown ablation id: ${a}" >&2
          exit 1
          ;;
      esac
      out+=("${a}")
    done
  fi
  printf '%s\n' "${out[@]}"
}

run_one() {
  local short="$1" fold="$2" gpu="$3"
  local abl="${ABLATION_DIRS[${short}]}"
  local ckpt="${ABLATION_ROOT}/${abl}/fold${fold}/best_model.pt"
  local base_out="${ABLATION_ROOT}/${abl}/eval_baseline_fold${fold}_test"
  local postop_out="${ABLATION_ROOT}/${abl}/eval_postop6h_fold${fold}_test"
  local log="${LOG_DIR}/ablation_eval/${abl}_fold${fold}.log"

  if [[ ! -f "${ckpt}" ]]; then
    echo "[skip-missing] ${abl} fold${fold}: no best_model.pt"
    return 1
  fi
  if [[ -f "${base_out}/baseline_metrics.json" && -f "${postop_out}/metrics.json" ]]; then
    echo "[skip] ${abl} fold${fold}: eval already present"
    return 0
  fi

  echo "[$(date -Is)] START ${abl} fold=${fold} gpu=${gpu}"
  {
    echo "=== ${abl} fold${fold} baseline ==="
    if [[ ! -f "${base_out}/baseline_metrics.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${EVENT_STREAM_DIR}/evaluate_baseline.py" \
        --checkpoint "${ckpt}" \
        --episode_path "${TEST_JSONL}" \
        --output_dir "${base_out}" \
        --device cuda:0 \
        --cls_threshold 0.5 \
        --batch_size 8 \
        --print_table
    fi
    echo "=== ${abl} fold${fold} postop6h ==="
    if [[ ! -f "${postop_out}/metrics.json" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${EVENT_STREAM_DIR}/evaluate_postop_hour_buckets.py" \
        --checkpoint "${ckpt}" \
        --episode_path "${TEST_JSONL}" \
        --output_dir "${postop_out}" \
        --device cuda:0 \
        --postop_hour_bin 6 \
        --macro_min_n 10 \
        --cls_threshold 0.5 \
        --batch_size 8 \
        --export_csv \
        --print_table
    fi
  } > "${log}" 2>&1
  local ec=$?
  echo "[$(date -Is)] DONE ${abl} fold=${fold} ec=${ec}"
  return "${ec}"
}

aggregate_one() {
  local short="$1"
  local abl="${ABLATION_DIRS[${short}]}"
  local root="${ABLATION_ROOT}/${abl}"
  local base_csv="${root}/eval_baseline_5fold_summary.csv"
  local postop_csv="${root}/eval_postop6h_5fold_summary.csv"

  "${PYTHON}" "${COMMON_DIR}/aggregate_baseline_folds.py" \
    --result_root "${root}" \
    --pattern "eval_baseline_fold{fold}_test" \
    --label_pattern "${abl}_baseline_fold{fold}" \
    --metrics_json baseline_metrics.json \
    --metrics_key metrics \
    --output_csv "${base_csv}"

  "${PYTHON}" "${COMMON_DIR}/aggregate_baseline_folds.py" \
    --result_root "${root}" \
    --pattern "eval_postop6h_fold{fold}_test" \
    --label_pattern "${abl}_postop6h_fold{fold}" \
    --metrics_json metrics.json \
    --metrics_key metrics.online_all_valid \
    --extra_metric macro_online_hour_bucket_auprc=summary.macro_online_hour_bucket_auprc \
    --extra_metric baseline_auprc=summary.baseline_auprc_node \
    --extra_metric final_valid_auprc=summary.final_valid_auprc \
    --output_csv "${postop_csv}"
}

mapfile -t SELECTED < <(resolve_ids "$@")
JOBS=()
for short in "${SELECTED[@]}"; do
  for fold in 1 2 3 4 5; do
    JOBS+=("${short}:${fold}")
  done
done

IFS=',' read -ra GPUS <<< "${ABLATION_GPUS:-0,1,2,3,4,5,6,7}"
echo "Ablations: ${SELECTED[*]}"
echo "Jobs: ${#JOBS[@]}  GPUs: ${GPUS[*]}"
echo "Root: ${ABLATION_ROOT}"

declare -A GPU_PID=()
declare -A GPU_JOB=()
FAIL=0
SLOT_GPU=""

wait_one_slot() {
  local gpu pid
  SLOT_GPU=""
  while true; do
    for gpu in "${GPUS[@]}"; do
      pid="${GPU_PID[${gpu}]:-}"
      if [[ -z "${pid}" ]]; then
        SLOT_GPU="${gpu}"
        return 0
      fi
      if ! kill -0 "${pid}" 2>/dev/null; then
        if ! wait "${pid}"; then
          FAIL=$((FAIL + 1))
        fi
        echo "[$(date -Is)] free gpu=${gpu} finished ${GPU_JOB[${gpu}]:-} (fail=${FAIL})"
        unset "GPU_PID[${gpu}]"
        unset "GPU_JOB[${gpu}]"
        SLOT_GPU="${gpu}"
        return 0
      fi
    done
    sleep 20
  done
}

for spec in "${JOBS[@]}"; do
  short="${spec%%:*}"
  fold="${spec##*:}"
  abl="${ABLATION_DIRS[${short}]}"
  if [[ -f "${ABLATION_ROOT}/${abl}/eval_baseline_fold${fold}_test/baseline_metrics.json" \
     && -f "${ABLATION_ROOT}/${abl}/eval_postop6h_fold${fold}_test/metrics.json" ]]; then
    echo "[skip] ${abl} fold${fold}"
    continue
  fi
  wait_one_slot
  gpu="${SLOT_GPU}"
  (
    run_one "${short}" "${fold}" "${gpu}"
  ) &
  GPU_PID[${gpu}]=$!
  GPU_JOB[${gpu}]="${abl}:fold${fold}"
done

for gpu in "${GPUS[@]}"; do
  pid="${GPU_PID[${gpu}]:-}"
  if [[ -n "${pid}" ]]; then
    if ! wait "${pid}"; then
      FAIL=$((FAIL + 1))
    fi
  fi
done

echo "[$(date -Is)] eval jobs finished fail=${FAIL}; aggregating..."

for short in "${SELECTED[@]}"; do
  aggregate_one "${short}" || true
done

"${PYTHON}" "${MODEL2_ROOT}/scripts/event_stream/aggregate_ablation_test.py"

echo "[$(date -Is)] ALL ablation eval finished fail=${FAIL}"
exit "${FAIL}"
