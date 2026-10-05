# Logistic regression (static baseline)

Uses the **same tabular features** as XGBoost (`five_folds_xgboost` parquet from `build_xgboost_dataset.py`).

**Train + test baseline eval (5 folds):**

```bash
# Build features once (if not already):
bash static_build_dataset_xgboost.sh

bash static_train_eval_logistic.sh
```

**Result paths:**
- Checkpoints: `result/static/logistic/fold{k}/model.joblib`
- Eval: `result/static/logistic/eval_fold{k}_test/baseline_metrics.json`
- Summary: `result/logistic_eval_nocross_5fold_summary.csv`

**Single fold:**

```bash
python static/logistic/train_logistic.py \
  --data_root ../../data/five_folds_xgboost \
  --val_fold 1 \
  --print_table
```
