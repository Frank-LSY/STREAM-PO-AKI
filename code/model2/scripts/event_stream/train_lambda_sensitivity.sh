#!/usr/bin/env bash
# Phase-1 training-λ sensitivity for event-stream Transformer.
#
# Fix checkpoint selection at λ_sel=0.9, sweep:
#   λ_base ∈ {0.1, 0.3, 0.5}
#   λ_mem  ∈ {0, 0.005, 0.01, 0.02}
#
# Uses the current embedding defaults (num_med_exposure=0, age MLP, hour-only time, …).
#
# Usage:
#   SENS_GPUS=1,2,3,4,5,6,7 bash scripts/event_stream/train_lambda_sensitivity.sh
#   SENS_FOLDS=1 bash scripts/event_stream/train_lambda_sensitivity.sh   # fold1 only
#
# Outputs: result/event_stream/sensitivity/lambda_grid/lb{base}_lm{mem}/fold{k}/
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

ROOT="${RESULT_DIR}/event_stream/sensitivity/lambda_grid"
mkdir -p "${ROOT}" "${LOG_DIR}/sensitivity"

COMMON_ARGS=(
  --folds_dir "${DATA_DIR}/"
  --test_jsonl "${TEST_JSONL}"
  --epochs 100 --lr 1e-4 --lr_step_size 5 --lr_gamma 0.7
  --early_stop_patience 10 --early_stop_min_delta 0.00001
  --dropout 0.2
  --batch_size 16
  --cls_pos_weight 1
  --lambda_online 1.0
  --mem_mask_ratio 0.15
  --best_selection_lambda 0.9
  --use_time_encoding 1
  --attention_mode block
  --baseline_token dedicated
  --num_med_exposure 0
)

BASES=(0.1 0.3 0.5)
MEMS=(0 0.005 0.01 0.02)

if [[ -n "${SENS_FOLDS:-}" ]]; then
  IFS=',' read -ra FOLDS <<< "${SENS_FOLDS}"
else
  FOLDS=(1 2 3 4 5)
fi

cfg_id() {
  local b="$1" m="$2"
  # lb0.3_lm0.01 / lb0.1_lm0
  printf 'lb%s_lm%s' "${b}" "${m}"
}

# Fold-major order: finish all λ configs on fold1 first (fast Phase-1 signal),
# then folds 2–5. Useful while only GPUs 6–7 are free during A3I4.
JOBS=()
for fold in "${FOLDS[@]}"; do
  for b in "${BASES[@]}"; do
    for m in "${MEMS[@]}"; do
      id="$(cfg_id "${b}" "${m}")"
      JOBS+=("${id}:${b}:${m}:${fold}")
    done
  done
done

IFS=',' read -ra GPUS <<< "${SENS_GPUS:-1,2,3,4,5,6,7}"
echo "Configs: ${#BASES[@]} bases × ${#MEMS[@]} mem × ${#FOLDS[@]} folds = $(( ${#BASES[@]} * ${#MEMS[@]} * ${#FOLDS[@]} )) jobs listed (${#JOBS[@]} queued)"
echo "GPUs: ${GPUS[*]}"
echo "Root: ${ROOT}"
echo "Selection λ fixed at 0.9 (online) / 0.1 (baseline)"

run_one() {
  local id="$1" base="$2" mem="$3" fold="$4" gpu="$5"
  local save="${ROOT}/${id}/fold${fold}"
  local log="${LOG_DIR}/sensitivity/${id}_fold${fold}.log"
  mkdir -p "${save}"
  if [[ -f "${save}/best_model.pt" ]]; then
    echo "[skip] ${id} fold${fold}: best_model.pt exists"
    return 0
  fi
  echo "[$(date -Is)] START ${id} fold=${fold} gpu=${gpu} λ_base=${base} λ_mem=${mem}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u "${EVENT_STREAM_DIR}/train_online.py" \
    --save_dir "${save}" \
    --val_fold "${fold}" \
    --device cuda:0 \
    --wandb_project model2_lambda_sensitivity \
    --wandb_run_name "${id}_fold${fold}" \
    "${COMMON_ARGS[@]}" \
    --lambda_base "${base}" \
    --lambda_mem "${mem}" \
    > "${log}" 2>&1
  local ec=$?
  echo "[$(date -Is)] DONE ${id} fold=${fold} ec=${ec}"
  return "${ec}"
}

declare -A GPU_PID=()
declare -A GPU_JOB=()
FAIL=0
SLOT_GPU=""

# True if physical GPU is idle enough for a new train (and not already held by us).
# Lets GPUs 6–7 start while A3I4 still occupies 1–5; picks up 1–5 when they free.
gpu_physically_idle() {
  local gpu="$1"
  local free total util threshold
  IFS=',' read -r free total util _ < <(
    nvidia-smi --query-gpu=memory.free,memory.total,utilization.gpu \
      --format=csv,noheader,nounits -i "${gpu}" 2>/dev/null | tr -d ' '
  )
  [[ -z "${free:-}" || -z "${total:-}" ]] && return 1
  # Need a nearly-empty card (~75 GiB free on 80 GiB GPUs) so we do not
  # share with A3I4 / other jobs that still hold a partial footprint.
  threshold="${SENS_MIN_FREE_MB:-75000}"
  [[ "${free}" -ge "${threshold}" ]]
}

wait_one_slot() {
  local gpu pid
  SLOT_GPU=""
  while true; do
    for gpu in "${GPUS[@]}"; do
      pid="${GPU_PID[${gpu}]:-}"
      if [[ -n "${pid}" ]]; then
        if ! kill -0 "${pid}" 2>/dev/null; then
          if ! wait "${pid}"; then
            FAIL=$((FAIL + 1))
          fi
          echo "[$(date -Is)] free gpu=${gpu} finished ${GPU_JOB[${gpu}]:-} (fail=${FAIL})"
          unset "GPU_PID[${gpu}]"
          unset "GPU_JOB[${gpu}]"
          # fall through to idle check below
        else
          continue
        fi
      fi
      if gpu_physically_idle "${gpu}"; then
        SLOT_GPU="${gpu}"
        return 0
      fi
    done
    sleep 30
  done
}

for spec in "${JOBS[@]}"; do
  IFS=':' read -r id base mem fold <<< "${spec}"
  if [[ -f "${ROOT}/${id}/fold${fold}/best_model.pt" ]]; then
    echo "[skip] ${id} fold${fold}"
    continue
  fi
  wait_one_slot
  gpu="${SLOT_GPU}"
  (
    run_one "${id}" "${base}" "${mem}" "${fold}" "${gpu}"
  ) &
  GPU_PID[${gpu}]=$!
  GPU_JOB[${gpu}]="${id}:fold${fold}"
done

for gpu in "${GPUS[@]}"; do
  pid="${GPU_PID[${gpu}]:-}"
  if [[ -n "${pid}" ]]; then
    if ! wait "${pid}"; then
      FAIL=$((FAIL + 1))
    fi
  fi
done

echo "[$(date -Is)] ALL lambda-sensitivity train jobs finished fail=${FAIL}"
# Phase-2 rescoring (no retrain)
"${PYTHON}" "${MODEL2_ROOT}/scripts/event_stream/rescore_selection_sensitivity.py" \
  --root "${ROOT}" \
  --folds "${FOLDS[*]}" \
  --out "${RESULT_DIR}/event_stream/sensitivity" \
  || true
echo "[$(date -Is)] rescore done; see result/event_stream/sensitivity/"
exit "${FAIL}"
