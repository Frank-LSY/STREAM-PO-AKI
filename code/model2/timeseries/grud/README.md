# GRU-D (6h time-series)

**Status:** implemented — causal GRU-D over `five_folds_ts6h` NPZ grids.

## Train

```bash
bash timeseries_build_dataset.sh
bash timeseries_train_all_folds_grud.sh      # 5 folds, GPU policy
bash timeseries_train_eval_grud.sh           # train (if needed) + baseline + postop6h eval
```

## Evaluate

```bash
bash timeseries_eval_baseline_grud.sh        # [BASE] bucket baseline
bash timeseries_eval_postop6h_grud.sh        # online preds by 6h postop hour bin
```

**Result path:** `result/timeseries/grud/fold{k}/`
