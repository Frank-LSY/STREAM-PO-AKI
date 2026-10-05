# Shared paths for model2 shell scripts. Source from scripts/**.sh
MODEL2_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${MODEL2_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
LOG_DIR="${MODEL2_ROOT}/logs"
COMMON_DIR="${MODEL2_ROOT}/common"

# Code layout: static | timeseries | event_stream
STATIC_DIR="${MODEL2_ROOT}/static"
TIMESERIES_DIR="${MODEL2_ROOT}/timeseries"
EVENT_STREAM_DIR="${MODEL2_ROOT}/event_stream"

# Backward-compat symlinks (blockmask -> event_stream, etc.)
BLOCKMASK_DIR="${EVENT_STREAM_DIR}"
BUCKET_TS6H_DIR="${TIMESERIES_DIR}"
XGBOOST_DIR="${STATIC_DIR}/xgboost"

# Data (export for Python subprocesses; see common/fold_data.default_data_dir)
# deduplicate4: same protocol as deduplicate3, but episodes with SEX|0 removed before split
export DATA_DIR="${MODEL2_ROOT}/../../data/five_folds_deduplicate4"
export XGB_DATA_DIR="${MODEL2_ROOT}/../../data/five_folds_xgboost4"
export TS6H_DATA_DIR="${MODEL2_ROOT}/../../data/five_folds_ts6h4"
TEST_NPZ="${TS6H_DATA_DIR}/test.npz"
RESULT_DIR="${MODEL2_ROOT}/../../result"
TEST_JSONL="${DATA_DIR}/7d_window_clm.mixed.test.jsonl"
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Canonical result paths: result/{category}/{arch}/fold{k}/...
# Run scripts/migrate_result_layout.sh once to symlink legacy dirs.
# ---------------------------------------------------------------------------

static_ckpt_dir() {
  # arch: logistic | xgboost | lightgbm
  echo "${RESULT_DIR}/static/${1}/fold${2}"
}

static_eval_dir() {
  echo "${RESULT_DIR}/static/${1}/eval_fold${2}_test"
}

timeseries_ckpt_dir() {
  # arch: grud | lstm | transformer
  echo "${RESULT_DIR}/timeseries/${1}/fold${2}"
}

timeseries_eval_dir() {
  echo "${RESULT_DIR}/timeseries/${1}/eval_fold${2}_test"
}

timeseries_postop6h_eval_dir() {
  echo "${RESULT_DIR}/timeseries/${1}/eval_postop6h_fold${2}_test"
}

timeseries_grid_eval_dir() {
  echo "${RESULT_DIR}/timeseries/${1}/grid_fold${2}_test"
}

event_stream_ckpt_dir() {
  # arch: gru | lstm | transformer
  echo "${RESULT_DIR}/event_stream/${1}/fold${2}"
}

event_stream_eval_dir() {
  # eval_type: baseline | postop6h | clinical | fixedgrid | attribution | ...
  echo "${RESULT_DIR}/event_stream/${1}/eval_${2}_fold${3}_test"
}

fold_train_jsonl() {
  echo "${DATA_DIR}/7d_window_clm.mixed.fold${1}.jsonl"
}

log_file() {
  echo "${LOG_DIR}/${1}"
}

# ---------------------------------------------------------------------------
# Legacy helpers (delegate to canonical paths after layout migration)
# ---------------------------------------------------------------------------

xgboost_model_dir() { static_ckpt_dir xgboost "${1}"; }
xgboost_eval_dir() { static_eval_dir xgboost "${1}"; }

bucket_ckpt_dir() { timeseries_ckpt_dir transformer "${1}"; }
bucket_baseline_eval_dir() { timeseries_eval_dir transformer "${1}"; }
bucket_lstm_ckpt_dir() { timeseries_ckpt_dir lstm "${1}"; }
bucket_lstm_baseline_eval_dir() { timeseries_eval_dir lstm "${1}"; }
bucket_lstm_grid_eval_dir() { timeseries_grid_eval_dir lstm "${1}"; }

checkpoint_nocross() {
  echo "$(event_stream_ckpt_dir transformer "${1}")/best_model.pt"
}

checkpoint_nocross_lstm() {
  echo "$(event_stream_ckpt_dir lstm "${1}")/best_model.pt"
}

blockmask_lstm_ckpt_dir() { event_stream_ckpt_dir lstm "${1}"; }
blockmask_lstm_baseline_eval_dir() { event_stream_eval_dir lstm baseline "${1}"; }
blockmask_lstm_postop_6h_eval_dir() { event_stream_eval_dir lstm postop6h "${1}"; }
blockmask_lstm_clinical_eval_dir() { event_stream_eval_dir lstm clinical "${1}"; }
blockmask_lstm_fixed_grid_eval_dir() { event_stream_eval_dir lstm fixedgrid "${1}"; }

# 6h bucket training throughput (H100 80GB defaults; override via env).
BUCKET_LSTM_BATCH_SIZE="${BUCKET_LSTM_BATCH_SIZE:-2048}"
BUCKET_GRUD_BATCH_SIZE="${BUCKET_GRUD_BATCH_SIZE:-2048}"
BUCKET_TRANSFORMER_BATCH_SIZE="${BUCKET_TRANSFORMER_BATCH_SIZE:-2048}"
# Modest DataLoader workers for TS6H (override per launch).
BUCKET_TRAIN_NUM_WORKERS="${BUCKET_TRAIN_NUM_WORKERS:-0}"
# Align with event-stream training budget (override via TS_EPOCHS / CLI).
BUCKET_TRAIN_EPOCHS="${BUCKET_TRAIN_EPOCHS:-100}"
BUCKET_EVAL_BATCH_SIZE="${BUCKET_EVAL_BATCH_SIZE:-2048}"
