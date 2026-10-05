"""6h bucket sequence + block-masked Transformer (attention mirrors the event-stream model)."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from common.block_encoder import BlockMaskedTransformerEncoder, build_periop_block_attention_mask
from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, NUM_PREOP_BUCKETS, SEQ_LEN


def build_static_periop_block_mask(
    seq_len: int = SEQ_LEN,
    base_idx: int = BASE_BUCKET_IDX,
) -> torch.Tensor:
    """[L, L] block mask for the fixed 6h grid, where every step is valid.

    Preoperative steps attend bidirectionally among themselves; the [BASE] step attends
    to the full preoperative history plus itself; postoperative step i attends only to
    [BASE] and postoperative steps <= i. True means blocked.

    Equivalent to ``build_periop_block_attention_mask`` for this layout, but avoids its
    per-sample Python loop.
    """
    mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask[:base_idx, :base_idx] = False
    mask[base_idx, : base_idx + 1] = False
    n_post = seq_len - base_idx - 1
    if n_post > 0:
        mask[base_idx + 1 :, base_idx] = False
        mask[base_idx + 1 :, base_idx + 1 :] = torch.triu(
            torch.ones(n_post, n_post, dtype=torch.bool), diagonal=1
        )
    return mask


class PerioperativeBucketTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
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
        self.pre_ln = nn.LayerNorm(d_model)
        self.encoder = BlockMaskedTransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.online_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.baseline_head = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))

        pos = torch.arange(SEQ_LEN, dtype=torch.long)
        self.register_buffer("_positions", pos, persistent=False)
        self.register_buffer("_block_mask", build_static_periop_block_mask(), persistent=False)

    def _attention_block_mask(
        self,
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> torch.Tensor:
        B, L = attention_mask.shape
        standard_grid = (
            L == SEQ_LEN
            and bool(attention_mask.all())
            and bool((base_mask.argmax(dim=1) == BASE_BUCKET_IDX).all())
        )
        if standard_grid:
            return self._block_mask.unsqueeze(0).expand(B, -1, -1)
        return build_periop_block_attention_mask(attention_mask, base_mask)

    def forward(
        self,
        bucket_features: torch.Tensor,
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, L, _ = bucket_features.shape
        pos = self._positions[:L].unsqueeze(0).expand(B, -1)
        x = self.input_proj(bucket_features) + self.pos_emb(pos)
        x = self.pre_ln(x)
        block_mask = self._attention_block_mask(attention_mask, base_mask)
        h = self.encoder(x, block_mask)
        return {
            "token_logits": self.online_head(h).squeeze(-1),
            "baseline_logits_all": self.baseline_head(h).squeeze(-1),
            "hidden": h,
            "attention_block_mask": block_mask,
        }


def build_bucket_masks(batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Fixed [B, SEQ_LEN] masks for standard grid layout."""
    preop = torch.zeros(batch_size, SEQ_LEN, device=device)
    base = torch.zeros(batch_size, SEQ_LEN, device=device)
    postop = torch.zeros(batch_size, SEQ_LEN, device=device)
    preop[:, :NUM_PREOP_BUCKETS] = 1.0
    base[:, BASE_BUCKET_IDX] = 1.0
    postop[:, BASE_BUCKET_IDX + 1 :] = 1.0
    attn = torch.ones(batch_size, SEQ_LEN, device=device)
    return {
        "attention_mask": attn,
        "base_mask": base,
        "preop_mask": preop,
        "postop_mask": postop,
    }
