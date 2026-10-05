# GRU (event-stream)

Segmented BiGRU encoder mirroring `event_stream/lstm/` layout.

**Train (5 folds, background):**

```bash
bash event_stream_train_all_folds_gru.sh
# logs: logs/train_event_stream_gru_fold{1..5}.log
```

**Result path:** `result/event_stream/gru/fold{k}/`
