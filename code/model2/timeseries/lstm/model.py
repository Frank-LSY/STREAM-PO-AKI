"""6h bucket sequence + causal unidirectional LSTM over the full grid."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from timeseries.timeseries_6h_common import SEQ_LEN


class PerioperativeBucketLSTM(nn.Module):
    """Causal bucket LSTM: one unidirectional LSTM over all 149 time steps.

    Position t only depends on buckets 0..t. Baseline loss uses hidden at [BASE];
    online loss uses hidden at valid postoperative buckets (unchanged masks in train_loop).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_emb = nn.Embedding(SEQ_LEN, d_model)
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout,
        )
        self.online_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.baseline_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        pos = torch.arange(SEQ_LEN, dtype=torch.long)
        self.register_buffer("_positions", pos, persistent=False)

    def forward(
        self,
        bucket_features: torch.Tensor,
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask, base_mask
        B, L, _ = bucket_features.shape
        pos = self._positions[:L].unsqueeze(0).expand(B, -1)
        x = self.input_proj(bucket_features) + self.pos_emb(pos)
        h, _ = self.lstm(x)
        return {
            "token_logits": self.online_head(h).squeeze(-1),
            "baseline_logits_all": self.baseline_head(h).squeeze(-1),
            "hidden": h,
        }
