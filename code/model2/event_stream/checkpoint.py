"""Build event-stream Transformer or LSTM from checkpoint args."""
from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from event_stream.gru.model import PerioperativeBlockMaskGRU
from event_stream.lstm.model import PerioperativeBlockMaskLSTM
from event_stream.transformer.model import PerioperativeBlockMaskModel


def infer_arch(train_args: Dict[str, Any]) -> str:
    arch = str(train_args.get("arch", "transformer")).lower()
    if arch in {"lstm", "blockmask_lstm", "event_lstm"}:
        return "lstm"
    if arch in {"gru", "blockmask_gru", "event_gru"}:
        return "gru"
    return "transformer"


def build_blockmask_model(
    arch: str,
    vocab_size: int,
    train_args: Dict[str, Any],
    num_med_exposure: int,
) -> nn.Module:
    d_model = int(train_args.get("d_model", 128))
    num_layers = int(train_args.get("num_layers", 2))
    dropout = float(train_args.get("dropout", 0.1))
    if arch == "lstm":
        return PerioperativeBlockMaskLSTM(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            num_med_exposure=num_med_exposure,
        )
    if arch == "gru":
        return PerioperativeBlockMaskGRU(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            dropout=dropout,
            num_med_exposure=num_med_exposure,
        )
    return PerioperativeBlockMaskModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=int(train_args.get("nhead", 4)),
        num_layers=num_layers,
        dim_feedforward=int(train_args.get("ffn_dim", 256)),
        dropout=dropout,
        num_med_exposure=num_med_exposure,
        use_time_encoding=bool(train_args.get("use_time_encoding", True)),
        attention_mode=str(train_args.get("attention_mode", "block")),
    )


build_event_stream_model = build_blockmask_model
