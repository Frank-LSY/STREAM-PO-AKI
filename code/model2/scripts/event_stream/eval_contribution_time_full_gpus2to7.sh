#!/usr/bin/env bash
# Full-cohort grouped-deletion contribution on GPUs 2-7 (6-way shard parallelism).
# Keeps GPU0/GPU1 free (jupyter / timeseries sequential train).
#
# Queue: 5 folds × 6 shards = 30 jobs; at most 6 concurrent (one per GPU).
# When all shards finish, runs aggregate_contribution_5fold_full.py.
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

GPUS=(2 3 4 5 6 7)
NUM_SHARDS="${#GPUS[@]}"
MAX_EP="${MAX_EPISODES:-0}"
OUT_ROOT="${RESULT_DIR}/event_stream/transformer/contribution_5fold_full"
REGION_MAP="${MODEL2_ROOT}/../../data/procedure_region_map.json"
MASTER="$(log_file evaluate_event_contribution_time_full_launch.log)"
PY_SCRIPT="${EVENT_STREAM_DIR}/evaluate_event_contribution_time.py"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

exec > >(tee -a "${MASTER}") 2>&1

echo "======== [$(ts)] LAUNCH grouped-deletion contribution (full test) ========"
echo "GPUs=${GPUS[*]} num_shards=${NUM_SHARDS} max_episodes=${MAX_EP}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "TEST_JSONL=${TEST_JSONL}"
echo "REGION_MAP=${REGION_MAP}"

# Preserve previous full run if present and not already backed up.
if [[ -d "${OUT_ROOT}/fold1" && ! -d "${OUT_ROOT}_backup_pre_relaunch" ]]; then
  echo "[$(ts)] Moving existing ${OUT_ROOT} -> ${OUT_ROOT}_backup_pre_relaunch"
  mv "${OUT_ROOT}" "${OUT_ROOT}_backup_pre_relaunch"
fi
mkdir -p "${OUT_ROOT}"

declare -a QUEUE=()
for fold in 1 2 3 4 5; do
  ckpt="$(event_stream_best_ckpt transformer "${fold}")"
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing checkpoint: ${ckpt}" >&2
    exit 1
  fi
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    QUEUE+=("${fold}:${shard}")
  done
done

echo "[$(ts)] Queue size=${#QUEUE[@]} (5 folds × ${NUM_SHARDS} shards)"

# gpu_slot_pid[i] = pid running on GPUS[i], or empty
declare -a SLOT_PID=()
declare -a SLOT_JOB=()
for ((i = 0; i < NUM_SHARDS; i++)); do
  SLOT_PID[i]=""
  SLOT_JOB[i]=""
done

launch_one() {
  local fold="$1" shard="$2" slot="$3"
  local gpu="${GPUS[$slot]}"
  local out="${OUT_ROOT}/fold${fold}/shard${shard}"
  local log
  log="$(log_file "evaluate_event_contribution_time_fold${fold}_shard${shard}.log")"
  mkdir -p "${out}"
  echo "[$(ts)] START fold=${fold} shard=${shard}/${NUM_SHARDS} gpu=${gpu} -> ${log}"
  (
    unset CUDA_VISIBLE_DEVICES
    "${PYTHON}" "${PY_SCRIPT}" \
      --checkpoint "$(event_stream_best_ckpt transformer "${fold}")" \
      --episode_path "${TEST_JSONL}" \
      --output_dir "${out}" \
      --device "cuda:${gpu}" \
      --max_episodes "${MAX_EP}" \
      --max_query_steps 25 \
      --lookback_hours 72 \
      --max_groups 0 \
      --min_count 10 \
      --min_patients 10 \
      --max_query_hour 168 \
      --top_k 50 \
      --procedure_region_map "${REGION_MAP}" \
      --num_shards "${NUM_SHARDS}" \
      --shard_index "${shard}"
  ) > "${log}" 2>&1 &
  SLOT_PID[$slot]=$!
  SLOT_JOB[$slot]="fold${fold}/shard${shard}"
}

q_idx=0
fail=0

# Fill all slots first
while [[ "${q_idx}" -lt "${#QUEUE[@]}" ]]; do
  # free finished slots
  for ((slot = 0; slot < NUM_SHARDS; slot++)); do
    pid="${SLOT_PID[$slot]:-}"
    if [[ -n "${pid}" ]]; then
      if ! kill -0 "${pid}" 2>/dev/null; then
        if wait "${pid}"; then
          echo "[$(ts)] DONE ${SLOT_JOB[$slot]} (gpu=${GPUS[$slot]})"
        else
          echo "[$(ts)] FAILED ${SLOT_JOB[$slot]} (gpu=${GPUS[$slot]}) rc=$?"
          fail=1
        fi
        SLOT_PID[$slot]=""
        SLOT_JOB[$slot]=""
      fi
    fi
  done

  # launch into empty slots
  launched=0
  for ((slot = 0; slot < NUM_SHARDS; slot++)); do
    if [[ -z "${SLOT_PID[$slot]:-}" && "${q_idx}" -lt "${#QUEUE[@]}" ]]; then
      IFS=':' read -r fold shard <<< "${QUEUE[$q_idx]}"
      q_idx=$((q_idx + 1))
      launch_one "${fold}" "${shard}" "${slot}"
      launched=1
    fi
  done

  # if all slots busy, wait a bit
  busy=0
  for ((slot = 0; slot < NUM_SHARDS; slot++)); do
    [[ -n "${SLOT_PID[$slot]:-}" ]] && busy=1
  done
  if [[ "${busy}" -eq 1 && "${launched}" -eq 0 ]]; then
    sleep 30
  elif [[ "${q_idx}" -ge "${#QUEUE[@]}" ]]; then
    break
  fi
done

# Drain remaining
for ((slot = 0; slot < NUM_SHARDS; slot++)); do
  pid="${SLOT_PID[$slot]:-}"
  if [[ -n "${pid}" ]]; then
    if wait "${pid}"; then
      echo "[$(ts)] DONE ${SLOT_JOB[$slot]} (gpu=${GPUS[$slot]})"
    else
      echo "[$(ts)] FAILED ${SLOT_JOB[$slot]} (gpu=${GPUS[$slot]}) rc=$?"
      fail=1
    fi
  fi
done

if [[ "${fail}" -ne 0 ]]; then
  echo "[$(ts)] One or more shards failed — skip aggregate. Check logs/evaluate_event_contribution_time_fold*_shard*.log"
  exit 1
fi

echo "[$(ts)] All shards done — aggregating (bootstrap min_patients=25, n_boot=2000)"
AGG_LOG="$(log_file evaluate_event_contribution_time_full_aggregate.log)"
"${PYTHON}" "${EVENT_STREAM_DIR}/aggregate_contribution_5fold_full.py" \
  --root "${OUT_ROOT}" \
  --n_boot 2000 \
  --min_patients 25 \
  > "${AGG_LOG}" 2>&1
echo "[$(ts)] Aggregate done -> ${AGG_LOG}"
echo "Outputs: ${OUT_ROOT}/crossfold_{baseline,online}.csv"
echo "======== [$(ts)] ALL DONE ========"
