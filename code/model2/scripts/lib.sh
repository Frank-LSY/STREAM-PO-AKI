#!/usr/bin/env bash
# Reusable helpers for model2 shell scripts.
# Usage (from scripts/<category>/foo.sh):
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
#   init_model2_script "${BASH_SOURCE[0]}"

_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${_LIB_DIR}/common.sh"

init_model2_script() {
  local src="${1:?init_model2_script: missing BASH_SOURCE}"
  SCRIPT_DIR="$(cd "$(dirname "${src}")" && pwd)"
  cd "${MODEL2_ROOT}"
  mkdir -p "${LOG_DIR}"
}

for_each_fold() {
  local fold
  for fold in {1..5}; do
    "$@" "${fold}"
  done
}

# With CUDA_VISIBLE_DEVICES set, use cuda:0 inside the process.
bind_gpu() {
  export CUDA_VISIBLE_DEVICES="${1:-0}"
  echo "cuda:0"
}

device_cuda() {
  echo "cuda:${1:-0}"
}

gpu_for_fold() {
  # Map fold 1..5 -> cuda:0..4 (no CUDA_VISIBLE_DEVICES remapping).
  echo "cuda:$(( $1 - 1 ))"
}

gpu_for_fold() {
  # Map fold 1..5 -> cuda:0..4 (no CUDA_VISIBLE_DEVICES remapping).
  echo "cuda:$(( $1 - 1 ))"
}

# Populated by refresh_idle_gpus(); physical GPU indices with enough free memory.
_AVAILABLE_GPU_INDICES=()

# Idle if free >= max(GPU_MIN_FREE_MB, GPU_MIN_FREE_RATIO * total) and util <= GPU_MAX_UTIL.
refresh_idle_gpus() {
  _AVAILABLE_GPU_INDICES=()
  if ! command -v nvidia-smi &>/dev/null; then
    return 0
  fi

  local min_free_mb="${GPU_MIN_FREE_MB:-8192}"
  local min_free_ratio="${GPU_MIN_FREE_RATIO:-0.5}"
  local max_util="${GPU_MAX_UTIL:-20}"
  local -a candidates=()
  local idx free total util threshold

  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra candidates <<< "${CUDA_VISIBLE_DEVICES}"
  else
    mapfile -t candidates < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
  fi

  for idx in "${candidates[@]}"; do
    idx="${idx// /}"
    [[ -z "${idx}" ]] && continue
    IFS=',' read -r free total util _ < <(
      nvidia-smi --query-gpu=memory.free,memory.total,utilization.gpu \
        --format=csv,noheader,nounits -i "${idx}" 2>/dev/null | tr -d ' '
    )
    [[ -z "${free:-}" || -z "${total:-}" || -z "${util:-}" ]] && continue
    threshold="$(awk -v total="${total}" -v min_mb="${min_free_mb}" -v ratio="${min_free_ratio}" \
      'BEGIN { t = total * ratio; print int((t > min_mb) ? t : min_mb) }')"
    if [[ "${free}" -ge "${threshold}" && "${util}" -le "${max_util}" ]]; then
      _AVAILABLE_GPU_INDICES+=("${idx}")
    fi
  done
}

count_available_gpus() {
  refresh_idle_gpus
  echo "${#_AVAILABLE_GPU_INDICES[@]}"
}

device_for_fold() {
  # device_for_fold FOLD — map to an idle physical GPU (cuda:N).
  local fold="$1"
  local n="${#_AVAILABLE_GPU_INDICES[@]}"
  if [[ "${n}" -lt 1 ]]; then
    echo "cuda:0"
    return 0
  fi
  echo "cuda:${_AVAILABLE_GPU_INDICES[$(( (fold - 1) % n ))]}"
}

_idle_gpu_summary() {
  local min_free_mb="${GPU_MIN_FREE_MB:-8192}"
  local min_free_ratio="${GPU_MIN_FREE_RATIO:-0.5}"
  local max_util="${GPU_MAX_UTIL:-20}"
  if [[ "${#_AVAILABLE_GPU_INDICES[@]}" -eq 0 ]]; then
    printf 'none (need free >= max(%s MiB, %s * total) and util <= %s%%)' \
      "${min_free_mb}" "${min_free_ratio}" "${max_util}"
    return 0
  fi
  printf 'cuda:%s' "${_AVAILABLE_GPU_INDICES[0]}"
  local i
  for ((i = 1; i < ${#_AVAILABLE_GPU_INDICES[@]}; i++)); do
    printf ', cuda:%s' "${_AVAILABLE_GPU_INDICES[$i]}"
  done
}

# Run folds 1–5 sequentially on one idle GPU, or in parallel across idle GPUs.
# RUNNER: function (fold, device, log_path) — should redirect stdout/stderr to log_path.
# Optional 3rd arg WAIT_JOBS=1: wait for parallel jobs (default 0 = launch+disown, for long trains).
run_all_folds_with_gpu_policy() {
  local log_prefix="$1"
  local runner="$2"
  local wait_jobs="${3:-0}"
  local num_gpus fold log device
  local -a pids=()

  refresh_idle_gpus
  num_gpus="${#_AVAILABLE_GPU_INDICES[@]}"
  if [[ "${num_gpus}" -lt 1 ]]; then
    echo "No idle GPU found. Adjust GPU_MIN_FREE_MB / GPU_MIN_FREE_RATIO / GPU_MAX_UTIL or free a GPU." >&2
    echo "Criteria: $(_idle_gpu_summary)" >&2
    exit 1
  fi

  echo "Idle GPU(s): $(_idle_gpu_summary) (${num_gpus} available)."

  # TS_FORCE_SEQUENTIAL=1: one fold at a time. Else parallel across idle GPUs.
  # TS_FOLD_STAGGER_SEC: delay between parallel fold launches (spreads npz open).
  if [[ "${TS_FORCE_SEQUENTIAL:-0}" == "1" || "${num_gpus}" -le 1 ]]; then
    device="cuda:${_AVAILABLE_GPU_INDICES[0]}"
    echo "Running folds 1-5 sequentially on ${device}."
    for fold in {1..5}; do
      log="$(log_file "${log_prefix}_fold${fold}.log")"
      echo "Fold ${fold} -> ${log}"
      "${runner}" "${fold}" "${device}" "${log}"
    done
    echo "Finished 5 folds sequentially."
  else
    local stagger_sec="${TS_FOLD_STAGGER_SEC:-0}"
    echo "Launching folds 1-5 in parallel across ${num_gpus} idle GPU(s) (stagger=${stagger_sec}s)."
    for fold in {1..5}; do
      log="$(log_file "${log_prefix}_fold${fold}.log")"
      device="$(device_for_fold "${fold}")"
      echo "Fold ${fold} on ${device} -> ${log}"
      "${runner}" "${fold}" "${device}" "${log}" &
      pids+=($!)
      if [[ "${wait_jobs}" != "1" ]]; then
        disown
      fi
      if [[ "${stagger_sec}" -gt 0 && "${fold}" -lt 5 ]]; then
        sleep "${stagger_sec}"
      fi
    done
    if [[ "${wait_jobs}" == "1" ]]; then
      local pid rc=0
      for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
          rc=1
        fi
      done
      if [[ "${rc}" -ne 0 ]]; then
        echo "One or more fold jobs failed. Check logs/${log_prefix}_fold{1..5}.log" >&2
        exit 1
      fi
      echo "Finished 5 folds in parallel."
    else
      echo "Launched 5 jobs. Tail: logs/${log_prefix}_fold{1..5}.log"
    fi
  fi
}

require_ts6h_data() {
  if [[ ! -f "${TEST_NPZ}" ]]; then
    echo "Building ${TS6H_DATA_DIR} ..."
    bash "${MODEL2_ROOT}/scripts/timeseries/build_dataset.sh"
  fi
}

require_file() {
  local path="$1" msg="${2:-Missing file: ${path}}"
  if [[ ! -e "${path}" ]]; then
    echo "${msg}" >&2
    return 1
  fi
}

best_ckpt() {
  echo "${1%/}/best_model.pt"
}

event_stream_best_ckpt() {
  best_ckpt "$(event_stream_ckpt_dir "${1}" "${2}")"
}

timeseries_best_ckpt() {
  best_ckpt "$(timeseries_ckpt_dir "${1}" "${2}")"
}

aggregate_fold_summaries() {
  local pattern="$1" label_pattern="$2" output_csv="$3"
  shift 3
  "${PYTHON}" "${COMMON_DIR}/aggregate_baseline_folds.py" \
    --result_root "${RESULT_DIR}" \
    --pattern "${pattern}" \
    --label_pattern "${label_pattern}" \
    --output_csv "${output_csv}" \
    "$@"
}

# Print extra args for baseline calibration (empty if plot_cal=0).
baseline_calibration_args() {
  local fold="$1" plot_cal="${2:-1}"
  if [[ "${plot_cal}" == "1" ]]; then
    printf '%s\n' \
      --plot_calibration \
      --calib_jsonl "$(fold_train_jsonl "${fold}")" \
      --plot_temperature_scaled \
      --export_calibration_csv
  fi
}

run_event_stream_baseline_eval() {
  # run_event_stream_baseline_eval ARCH FOLD GPU [plot_cal]
  local arch="$1" fold="$2" gpu="${3:-0}" plot_cal="${4:-1}"
  local out ckpt
  out="$(event_stream_eval_dir "${arch}" baseline "${fold}")"
  ckpt="$(event_stream_best_ckpt "${arch}" "${fold}")"
  echo "Event-stream ${arch} baseline eval fold ${fold} -> ${out}"
  local -a extra=()
  if [[ "${plot_cal}" == "1" ]]; then
    extra=(
      --plot_calibration
      --calib_jsonl "$(fold_train_jsonl "${fold}")"
      --plot_temperature_scaled
      --export_calibration_csv
    )
  fi
  "${PYTHON}" "${EVENT_STREAM_DIR}/evaluate_baseline.py" \
    --checkpoint "${ckpt}" \
    --episode_path "${TEST_JSONL}" \
    --output_dir "${out}" \
    --device "$(device_cuda "${gpu}")" \
    --cls_threshold 0.5 \
    --print_table \
    "${extra[@]}"
}

run_timeseries_baseline_eval() {
  # run_timeseries_baseline_eval ARCH FOLD
  local arch="$1" fold="$2"
  local ckpt out
  ckpt="$(timeseries_best_ckpt "${arch}" "${fold}")"
  out="$(timeseries_eval_dir "${arch}" "${fold}")"
  echo "Timeseries ${arch} baseline eval fold ${fold} -> ${out}"
  "${PYTHON}" "${TIMESERIES_DIR}/evaluate_bucket_baseline.py" \
    --checkpoint "${ckpt}" \
    --test_npz "${TEST_NPZ}" \
    --output_dir "${out}" \
    --episode_path "${TEST_JSONL}" \
    --batch_size "${BUCKET_EVAL_BATCH_SIZE}" \
    --device cuda:0 \
    --print_table
}

run_timeseries_postop6h_eval() {
  # run_timeseries_postop6h_eval ARCH FOLD
  local arch="$1" fold="$2"
  local ckpt out
  ckpt="$(timeseries_best_ckpt "${arch}" "${fold}")"
  out="$(timeseries_postop6h_eval_dir "${arch}" "${fold}")"
  echo "Timeseries ${arch} postop6h eval fold ${fold} -> ${out}"
  "${PYTHON}" "${TIMESERIES_DIR}/evaluate_bucket_postop6h.py" \
    --checkpoint "${ckpt}" \
    --test_npz "${TEST_NPZ}" \
    --output_dir "${out}" \
    --episode_path "${TEST_JSONL}" \
    --batch_size "${BUCKET_EVAL_BATCH_SIZE}" \
    --device cuda:0 \
    --postop_hour_bin 6 \
    --macro_min_n 10 \
    --export_csv \
    --print_table
}

aggregate_timeseries_baseline_summary() {
  local arch="$1"
  local pattern summary
  case "${arch}" in
    lstm)
      pattern="timeseries/lstm/eval_fold{fold}_test"
      summary="${RESULT_DIR}/bucket_lstm_eval_nocross_5fold_summary.csv"
      ;;
    grud)
      pattern="timeseries/grud/eval_fold{fold}_test"
      summary="${RESULT_DIR}/bucket_grud_eval_nocross_5fold_summary.csv"
      ;;
    *)
      pattern="timeseries/transformer/eval_fold{fold}_test"
      summary="${RESULT_DIR}/bucket_eval_nocross_5fold_summary.csv"
      ;;
  esac
  aggregate_fold_summaries "${pattern}" "${pattern}" "${summary}"
  echo "Wrote ${summary}"
}

aggregate_timeseries_postop6h_summary() {
  local arch="$1"
  local postop_pattern postop_summary
  case "${arch}" in
    lstm)
      postop_pattern="timeseries/lstm/eval_postop6h_fold{fold}_test"
      postop_summary="${RESULT_DIR}/bucket_lstm_postop6h_5fold_summary.csv"
      ;;
    grud)
      postop_pattern="timeseries/grud/eval_postop6h_fold{fold}_test"
      postop_summary="${RESULT_DIR}/bucket_grud_postop6h_5fold_summary.csv"
      ;;
    *)
      postop_pattern="timeseries/transformer/eval_postop6h_fold{fold}_test"
      postop_summary="${RESULT_DIR}/bucket_postop6h_5fold_summary.csv"
      ;;
  esac
  aggregate_fold_summaries "${postop_pattern}" "${postop_pattern}" "${postop_summary}" \
    --metrics_json postop_hour_bucket_metrics.json
  echo "Wrote ${postop_summary}"
}

aggregate_event_stream_fixedgrid_summary() {
  # Same 6h postoperative grid as aggregate_timeseries_postop6h_summary, so the two CSVs line up.
  local arch="${1:-transformer}"
  local pattern summary
  pattern="event_stream/${arch}/eval_fixedgrid_fold{fold}_test"
  summary="${RESULT_DIR}/event_stream_${arch}_fixedgrid_postop6h_5fold_summary.csv"
  aggregate_fold_summaries "${pattern}" "${pattern}" "${summary}" \
    --metrics_json fixed_time_grid_metrics.json \
    --metrics_key pooled_grid \
    --extra_metric macro_online_hour_bucket_auprc=summary.macro_bucket_auprc
  echo "Wrote ${summary}"
}

train_timeseries_fold() {
  # train_timeseries_fold ARCH FOLD EPOCHS BATCH_SIZE [DEVICE] [extra args...]
  local arch="$1" fold="$2" epochs="$3" batch_size="$4" device="${5:-cuda:0}"
  shift 5
  local save
  save="$(timeseries_ckpt_dir "${arch}" "${fold}")"
  echo "Train timeseries ${arch} fold ${fold} -> ${save} (batch=${batch_size}, device=${device})"
  "${PYTHON}" "${TIMESERIES_DIR}/${arch}/train.py" \
    --data_root "${TS6H_DATA_DIR}" \
    --val_fold "${fold}" \
    --save_dir "${save}" \
    --test_jsonl "${TEST_JSONL}" \
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --num_workers "${BUCKET_TRAIN_NUM_WORKERS}" \
    --device "${device}" \
    --lr_gamma 0.7 \
    --dropout 0.2 \
    --lambda_online 1.0 --lambda_base 0.3 \
    --early_stop_patience 10 --early_stop_min_delta 0.00001 \
    --best_selection_lambda 0.9 \
    --wandb_project "${WANDB_PROJECT_TS6H:-model2_ts6h}" \
    --wandb_run_name "ts6h_${arch}_fold${fold}" \
    "$@"
}

train_event_stream_fold() {
  # train_event_stream_fold ARCH FOLD SAVE_DIR EPOCHS BATCH_SIZE DEVICE [extra...]
  local arch="$1" fold="$2" save_dir="$3" epochs="$4" batch_size="$5" device="$6"
  shift 6
  local train_py="${EVENT_STREAM_DIR}/train_online.py"
  if [[ "${arch}" != "transformer" ]]; then
    train_py="${EVENT_STREAM_DIR}/${arch}/train.py"
  fi
  echo "Train event-stream ${arch} fold ${fold} -> ${save_dir}"
  "${PYTHON}" "${train_py}" \
    --save_dir "${save_dir}" \
    --folds_dir "${DATA_DIR}/" \
    --val_fold "${fold}" \
    --test_jsonl "${TEST_JSONL}" \
    --epochs "${epochs}" \
    --batch_size "${batch_size}" \
    --device "${device}" \
    --lr 1e-4 --lr_step_size 5 --lr_gamma 0.7 \
    --early_stop_patience 10 --early_stop_min_delta 0.00001 \
    --dropout 0.2 \
    --cls_pos_weight 1 \
    --lambda_online 1.0 --lambda_base 0.3 --lambda_mem 0.01 \
    --mem_mask_ratio 0.15 \
    --best_selection_lambda 0.9 \
    "$@"
}

launch_nohup_per_fold() {
  # launch_nohup_per_fold LOG_PREFIX CMD...
  local log_prefix="$1"
  shift
  local fold log
  for fold in {1..5}; do
    log="$(log_file "${log_prefix}_${fold}.log")"
    echo "Launch fold ${fold} -> ${log}"
    nohup "$@" > "${log}" 2>&1 &
  done
}

run_event_stream_eval_fold() {
  # run_event_stream_eval_fold ARCH EVAL_TYPE FOLD GPU [extra python args...]
  local arch="$1" eval_type="$2" fold="$3" gpu="${4:-0}"
  shift 4
  local module ckpt out
  ckpt="$(event_stream_best_ckpt "${arch}" "${fold}")"
  out="$(event_stream_eval_dir "${arch}" "${eval_type}" "${fold}")"
  case "${eval_type}" in
    postop6h) module="evaluate_postop_hour_buckets.py" ;;
    clinical) module="evaluate_clinical.py" ;;
    fixedgrid) module="evaluate_fixed_time_grid.py" ;;
    *)
      echo "Unknown event_stream eval_type: ${eval_type}" >&2
      return 1
      ;;
  esac
  if [[ ! -f "${ckpt}" ]]; then
    echo "Skip fold ${fold}: missing ${ckpt}" >&2
    return 0
  fi
  echo "Event-stream ${arch} ${eval_type} eval fold ${fold} -> ${out}"
  "${PYTHON}" "${EVENT_STREAM_DIR}/${module}" \
    --checkpoint "${ckpt}" \
    --episode_path "${TEST_JSONL}" \
    --output_dir "${out}" \
    --device "$(device_cuda "${gpu}")" \
    "$@"
}

train_eval_timeseries() {
  # train_eval_timeseries ARCH [epochs] [batch_size]
  local arch="$1"
  local epochs="${2:-${BUCKET_TRAIN_EPOCHS:-100}}"
  local batch_size="${3:-}"
  if [[ -z "${batch_size}" ]]; then
    if [[ "${arch}" == "lstm" ]]; then
      batch_size="${BUCKET_LSTM_BATCH_SIZE}"
    elif [[ "${arch}" == "grud" ]]; then
      batch_size="${BUCKET_GRUD_BATCH_SIZE}"
    else
      batch_size="${BUCKET_TRANSFORMER_BATCH_SIZE}"
    fi
  fi
  require_ts6h_data
  local fold ckpt log
  for fold in {1..5}; do
    ckpt="$(timeseries_best_ckpt "${arch}" "${fold}")"
    log="$(log_file "train_eval_timeseries_${arch}_fold${fold}.log")"
    echo "Fold ${fold} train+eval -> ${log}"
    {
      if [[ ! -f "${ckpt}" ]]; then
        train_timeseries_fold "${arch}" "${fold}" "${epochs}" "${batch_size}"
      else
        echo "Skip train; found ${ckpt}"
      fi
      run_timeseries_baseline_eval "${arch}" "${fold}"
      run_timeseries_postop6h_eval "${arch}" "${fold}"
    } > "${log}" 2>&1
  done
  aggregate_timeseries_baseline_summary "${arch}"
  aggregate_timeseries_postop6h_summary "${arch}"
}

train_eval_event_stream() {
  # train_eval_event_stream ARCH [epochs] [batch_size] [gpu]
  local arch="$1"
  local epochs="${2:-20}"
  local batch_size="${3:-16}"
  local gpu="${4:-0}"
  local device
  device="$(bind_gpu "${gpu}")"
  local fold ckpt save
  for fold in {1..5}; do
    ckpt="$(event_stream_best_ckpt "${arch}" "${fold}")"
    save="$(event_stream_ckpt_dir "${arch}" "${fold}")"
    if [[ ! -f "${ckpt}" ]]; then
      echo "Train event-stream ${arch} fold ${fold}"
      train_event_stream_fold "${arch}" "${fold}" "${save}" "${epochs}" "${batch_size}" "${device}" \
        --early_stop_patience 3
    fi
    run_event_stream_baseline_eval "${arch}" "${fold}" "${gpu}" 0
  done
  local pattern summary label
  if [[ "${arch}" == "lstm" ]]; then
    pattern="baseline_eval_lstm_nocross_fold{fold}_test"
    label="${pattern}"
    summary="${RESULT_DIR}/baseline_eval_lstm_nocross_5fold_summary.csv"
  elif [[ "${arch}" == "gru" ]]; then
    pattern="baseline_eval_gru_nocross_fold{fold}_test"
    label="${pattern}"
    summary="${RESULT_DIR}/baseline_eval_gru_nocross_5fold_summary.csv"
  else
    pattern="baseline_eval_nocross_fold{fold}_test"
    label="${pattern}"
    summary="${RESULT_DIR}/baseline_eval_nocross_5fold_summary.csv"
  fi
  aggregate_fold_summaries "${pattern}" "${label}" "${summary}"
  echo "Wrote ${summary}"
}
