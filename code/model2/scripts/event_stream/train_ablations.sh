#!/usr/bin/env bash
# Train event-stream Transformer ablations (5-fold).
#
# Usage:
#   bash scripts/event_stream/train_ablations.sh                 # P1 defaults
#   bash scripts/event_stream/train_ablations.sh A1 A2 A4 A5    # by id
#   ABLATION_GPUS=0,6,7 bash scripts/event_stream/train_ablations.sh all
#
# Outputs: result/event_stream/ablation/<ablation_id>/fold{k}/
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ABLATION_ROOT="${RESULT_DIR}/event_stream/ablation"
mkdir -p "${ABLATION_ROOT}" "${LOG_DIR}/ablation"

# Shared training hyperparams (match production full model).
COMMON_ARGS=(
  --folds_dir "${DATA_DIR}/"
  --test_jsonl "${TEST_JSONL}"
  --epochs 100 --lr 1e-4 --lr_step_size 5 --lr_gamma 0.7
  --early_stop_patience 10 --early_stop_min_delta 0.00001
  --dropout 0.2
  --batch_size 16
  --cls_pos_weight 1
  --lambda_online 1.0 --lambda_base 0.3 --lambda_mem 0.01
  --mem_mask_ratio 0.15
  --best_selection_lambda 0.9
  --use_time_encoding 1
  --attention_mode block
  --baseline_token dedicated
  --num_med_exposure 0
)

# ablation_id -> extra CLI args (as a string evaluated later)
declare -A ABLATION_EXTRA=(
  # Architecture / training
  [A1_no_time_encoding]="--use_time_encoding 0"
  [A2_causal_attention]="--attention_mode causal"
  [A4_no_baseline_loss]="--lambda_base 0 --best_selection_lambda 1.0"
  [A5_no_mem]="--lambda_mem 0"
  [A3_procedure_as_base]="--baseline_token procedure"
  # Input modalities (full event removal; med-exposure vector is off in production)
  [I1_no_med_events]="--drop_meds"
  [I2_no_lab_events]="--drop_labs"
  [I3_no_conditions]="--drop_conditions"
  [I4_generic_surgery]="--generic_surgery"
  # Interaction — A3 × I4 (procedure as baseline + generic surgery)
  [A3I4_procedure_as_base_generic_surgery]="--baseline_token procedure --generic_surgery"
)

P1_IDS=(A1_no_time_encoding A2_causal_attention A4_no_baseline_loss A5_no_mem)
P2_IDS=(A3_procedure_as_base)
P3_IDS=(I1_no_med_events I2_no_lab_events I3_no_conditions I4_generic_surgery)
X_IDS=(A3I4_procedure_as_base_generic_surgery)
ALL_IDS=("${P1_IDS[@]}" "${P2_IDS[@]}" "${P3_IDS[@]}" "${X_IDS[@]}")

resolve_ids() {
  local -a out=()
  local a
  if [[ $# -eq 0 || "${1}" == "p1" ]]; then
    out=("${P1_IDS[@]}")
  elif [[ "${1}" == "p2" ]]; then
    out=("${P2_IDS[@]}")
  elif [[ "${1}" == "p3" ]]; then
    out=("${P3_IDS[@]}")
  elif [[ "${1}" == "x" || "${1}" == "interaction" ]]; then
    out=("${X_IDS[@]}")
  elif [[ "${1}" == "all" ]]; then
    out=("${ALL_IDS[@]}")
  else
    for a in "$@"; do
      # allow short forms A1 / I1 / A3I4
      case "${a}" in
        A1) a=A1_no_time_encoding ;;
        A2) a=A2_causal_attention ;;
        A3) a=A3_procedure_as_base ;;
        A4) a=A4_no_baseline_loss ;;
        A5) a=A5_no_mem ;;
        I1) a=I1_no_med_events ;;
        I2) a=I2_no_lab_events ;;
        I3) a=I3_no_conditions ;;
        I4) a=I4_generic_surgery ;;
        A3I4|X1) a=A3I4_procedure_as_base_generic_surgery ;;
        # legacy aliases (old partial ablations)
        I1_no_med_exposure) a=I1_no_med_events ;;
        I2_no_lab_values) a=I2_no_lab_events ;;
      esac
      if [[ -z "${ABLATION_EXTRA[${a}]+x}" ]]; then
        echo "Unknown ablation id: ${a}" >&2
        exit 1
      fi
      out+=("${a}")
    done
  fi
  printf '%s\n' "${out[@]}"
}

run_one() {
  local abl="$1" fold="$2" gpu="$3"
  local save="${ABLATION_ROOT}/${abl}/fold${fold}"
  local log="${LOG_DIR}/ablation/${abl}_fold${fold}.log"
  mkdir -p "${save}"
  if [[ -f "${save}/best_model.pt" ]]; then
    echo "[skip] ${abl} fold${fold}: best_model.pt exists"
    return 0
  fi
  # shellcheck disable=SC2206
  local extra=( ${ABLATION_EXTRA[${abl}]} )
  echo "[$(date -Is)] START ${abl} fold=${fold} gpu=${gpu} extra=${extra[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${EVENT_STREAM_DIR}/train_online.py" \
    --save_dir "${save}" \
    --val_fold "${fold}" \
    --device cuda:0 \
    --wandb_project model2_ablation \
    --wandb_run_name "${abl}_fold${fold}" \
    "${COMMON_ARGS[@]}" \
    "${extra[@]}" \
    > "${log}" 2>&1
  local ec=$?
  echo "[$(date -Is)] DONE ${abl} fold=${fold} ec=${ec}"
  return "${ec}"
}

# Build job queue: ablation × fold
mapfile -t SELECTED < <(resolve_ids "$@")
JOBS=()
for abl in "${SELECTED[@]}"; do
  for fold in 1 2 3 4 5; do
    JOBS+=("${abl}:${fold}")
  done
done

IFS=',' read -ra GPUS <<< "${ABLATION_GPUS:-0,6,7}"
echo "Ablations: ${SELECTED[*]}"
echo "Jobs: ${#JOBS[@]}  GPUs: ${GPUS[*]}"
echo "Root: ${ABLATION_ROOT}"

# Simple GPU worker pool.
# IMPORTANT: do not capture wait_one_slot via $(...) — free-slot logs must not
# pollute CUDA_VISIBLE_DEVICES (that previously made CUDA "unavailable").
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
    sleep 30
  done
}

for spec in "${JOBS[@]}"; do
  abl="${spec%%:*}"
  fold="${spec##*:}"
  if [[ -f "${ABLATION_ROOT}/${abl}/fold${fold}/best_model.pt" ]]; then
    echo "[skip] ${abl} fold${fold}"
    continue
  fi
  wait_one_slot
  gpu="${SLOT_GPU}"
  (
    run_one "${abl}" "${fold}" "${gpu}"
  ) &
  GPU_PID[${gpu}]=$!
  GPU_JOB[${gpu}]="${abl}:fold${fold}"
done

# Drain
for gpu in "${GPUS[@]}"; do
  pid="${GPU_PID[${gpu}]:-}"
  if [[ -n "${pid}" ]]; then
    if ! wait "${pid}"; then
      FAIL=$((FAIL + 1))
    fi
  fi
done

echo "[$(date -Is)] ALL ablation jobs finished fail=${FAIL}"
exit "${FAIL}"
