"""Load 6h bucket Transformer, LSTM, or GRU-D checkpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from timeseries.grud.model import PerioperativeBucketGRUD
from timeseries.lstm.model import PerioperativeBucketLSTM
from timeseries.transformer.model import PerioperativeBucketTransformer


def infer_arch(train_args: Dict[str, Any]) -> str:
    arch = str(train_args.get("arch", "transformer")).lower()
    if arch in {"lstm", "bucket_lstm", "bucket_lstm_ts6h"}:
        return "lstm"
    if arch in {"grud", "gru_d", "gru-d", "bucket_grud"}:
        return "grud"
    return "transformer"


def build_bucket_model(
    arch: str,
    n_features: int,
    train_args: Dict[str, Any],
    layout: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    d_model = int(train_args.get("d_model", 128))
    num_layers = int(train_args.get("num_layers", 2))
    dropout = float(train_args.get("dropout", 0.1))
    if arch == "lstm":
        return PerioperativeBucketLSTM(
            input_dim=n_features,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
        )
    if arch == "grud":
        return PerioperativeBucketGRUD(
            input_dim=n_features,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            layout=layout,
            obs_mode=str(train_args.get("grud_obs_mode", "legacy")),
        )
    return PerioperativeBucketTransformer(
        input_dim=n_features,
        d_model=d_model,
        nhead=int(train_args.get("nhead", 4)),
        num_layers=num_layers,
        dim_feedforward=int(train_args.get("ffn_dim", 256)),
        dropout=dropout,
    )


def load_bucket_checkpoint(
    checkpoint: str | Path,
    device: torch.device | None = None,
) -> Tuple[nn.Module, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    train_args = dict(ckpt.get("args") or {})
    layout = dict(ckpt.get("feature_layout") or {})
    n_features = int(ckpt.get("n_features") or layout.get("n_features"))
    arch = infer_arch(train_args)
    model = build_bucket_model(arch, n_features, train_args, layout=layout or None)
    model.load_state_dict(ckpt["model_state_dict"])
    if layout and hasattr(model, "set_layout"):
        model.set_layout(layout)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, ckpt, train_args, layout
