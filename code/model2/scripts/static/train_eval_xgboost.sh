#!/usr/bin/env bash
# Train/eval static XGBoost on 5 folds.
set -euo pipefail
# shellcheck source=../lib.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../lib.sh"
init_model2_script "${BASH_SOURCE[0]}"

for fold in {1..5}; do
  model_out="$(static_ckpt_dir xgboost "${fold}")"
  eval_out="$(static_eval_dir xgboost "${fold}")"
  echo "XGBoost fold ${fold} -> ${model_out} | eval ${eval_out}"
  "${PYTHON}" "${STATIC_DIR}/xgboost/train_xgboost.py" \
    --data_root "${XGB_DATA_DIR}" \
    --val_fold "${fold}" \
    --model_dir "${model_out}" \
    --eval_output_dir "${eval_out}" \
    --episode_path "${TEST_JSONL}" \
    --cls_threshold 0.5 \
    --print_table
  echo "Done: ${eval_out}/baseline_metrics.json"
done

aggregate_fold_summaries \
  "static/xgboost/eval_fold{fold}_test" \
  "xgboost_eval_nocross_fold{fold}_test" \
  "${RESULT_DIR}/static/xgboost/eval_nocross_5fold_summary.csv"
echo "Wrote ${RESULT_DIR}/static/xgboost/eval_nocross_5fold_summary.csv"
