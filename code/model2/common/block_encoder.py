"""Block-masked Transformer encoder shared by event-stream and bucket models."""
from __future__ import annotations

import torch
import torch.nn as nn


class BlockMaskedTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 256, dropout: float = 0.1):
        super().__init__()
        self.nhead = int(nhead)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        expanded_mask = attn_mask.unsqueeze(1).expand(B, self.nhead, L, L)
        expanded_mask = expanded_mask.reshape(B * self.nhead, L, L)
        attn_out, _ = self.self_attn(
            x, x, x,
            attn_mask=expanded_mask,
            need_weights=False,
        )
        x = self.norm1(x + self.dropout1(attn_out))
        ff = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm2(x + self.dropout2(ff))
        return x


class BlockMaskedTransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            BlockMaskedTransformerLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x


def build_causal_attention_mask(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Lower-triangular causal mask for fixed or padded sequences.

    Query at index i may attend to keys j <= i (among valid positions).
    Returns bool mask [B, L, L] where True means attention is blocked.
    """
    future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    mask = future.unsqueeze(0).expand(batch_size, -1, -1).clone()
    if attention_mask is None:
        return mask
    valid = attention_mask > 0
    for b in range(batch_size):
        invalid = ~valid[b]
        if invalid.any():
            mask[b, invalid, :] = True
            mask[b, :, invalid] = True
        valid_idx = torch.nonzero(valid[b], as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            mask[b, :, 0] = False
            continue
        first_valid = int(valid_idx[0].item())
        mask[b, invalid, first_valid] = False
    return mask


def build_periop_block_attention_mask(
    attention_mask: torch.Tensor,
    base_mask: torch.Tensor,
) -> torch.Tensor:
    """Build per-sample structured block attention mask.

    Sequence layout is expected to be:
        preoperative tokens, [BASE], postoperative tokens, padding

    Allowed attention:
    - Preoperative query -> all preoperative keys only.
    - [BASE] query -> preoperative keys + [BASE].
    - Postoperative query i -> [BASE] + postoperative keys up to i.
    - Padding query -> key 0 only, to avoid all-masked rows; padding outputs are ignored.

    Returns bool mask [B, L, L], where True means attention is blocked.
    """
    device = attention_mask.device
    B, L = attention_mask.shape
    valid = attention_mask > 0
    base_bool = base_mask > 0

    pos = torch.arange(L, device=device)
    q = pos.view(1, L, 1)
    k = pos.view(1, 1, L)
    vq = valid[:, :, None]
    vk = valid[:, None, :]
    has_valid = valid.any(dim=1).view(B, 1, 1)
    has_base = base_bool.any(dim=1).view(B, 1, 1)
    bi = base_bool.int().argmax(dim=1).view(B, 1, 1)
    first_valid = valid.int().argmax(dim=1).view(B, 1, 1)

    pre_k = vk & (k < bi)
    allow_base = (
        (vq & (q < bi) & pre_k)
        | ((q == bi) & (pre_k | (k == bi)))
        | (vq & (q > bi) & ((k == bi) | (vk & (k > bi) & (k <= q))))
    )
    allow_nobase = vq & vk & (k <= q)
    allow = torch.where(has_base, allow_base, allow_nobase)
    allow = allow | ((~vq) & (k == first_valid))
    allow = torch.where(has_valid, allow, (k == 0).expand(B, L, L))
    return ~allow
