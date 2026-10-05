"""Fixed 6h time-grid models (shared NPZ + per-architecture subpackages).

Subpackages:
- ``timeseries.transformer`` — block-mask Transformer on 149-step grid
- ``timeseries.lstm`` — segmented BiLSTM + causal postop LSTM
- ``timeseries.grud`` — GRU-D (missing-value decay GRU)

Shared: ``dataset``, ``timeseries_6h_common``, ``train_loop``, ``checkpoint``, ``evaluate_bucket_*``.
"""
