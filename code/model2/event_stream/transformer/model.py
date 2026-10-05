from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from common.med_exposure import NUM_EVENT_TYPES
from common.block_encoder import (
    BlockMaskedTransformerEncoder,
    build_causal_attention_mask,
    build_periop_block_attention_mask,
)


class PerioperativeBlockMaskModel(nn.Module):
    """Unified perioperative Transformer with structured block attention.

    Outputs:
    - baseline_logits_all: risk logits at all tokens; [BASE] position is used for baseline loss.
    - token_logits: online risk logits at all tokens; postoperative positions are used for online loss.
    - mem_logits: masked event-code reconstruction logits at all tokens.
    """

    def __init__(
        self,
        vocab_size: int,
        num_modalities: int = 6,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        num_med_exposure: int = 0,
        use_time_encoding: bool = True,
        attention_mode: str = "block",
    ):
        super().__init__()
        self.num_med_exposure = int(num_med_exposure)
        self.use_time_encoding = bool(use_time_encoding)
        mode = str(attention_mode).lower().strip()
        if mode not in {"block", "causal"}:
            raise ValueError(f"attention_mode must be 'block' or 'causal', got {attention_mode!r}")
        self.attention_mode = mode
        self.code_emb = nn.Embedding(vocab_size, d_model)
        self.modality_emb = nn.Embedding(num_modalities, d_model)
        # Medication start/end boundary; id 0 = none (non-med / pad), frozen as zeros.
        self.event_type_emb = nn.Embedding(NUM_EVENT_TYPES, d_model, padding_idx=0)
        # Continuous time: hour / 24 only (no day channel).
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        # Lab (and other numeric) values; contribution gated by value_mask.
        self.value_mlp = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        # AGE token only: z-scored age + mask, separate from value_mlp.
        self.age_mlp = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.med_exposure_proj: Optional[nn.Linear]
        if self.num_med_exposure > 0:
            self.med_exposure_proj = nn.Linear(self.num_med_exposure, d_model)
        else:
            self.med_exposure_proj = None
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
        self.mem_head = nn.Linear(d_model, vocab_size)

    def embed_tokens(
        self,
        code_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        time_days: torch.Tensor,
        time_hours: torch.Tensor,
        values: torch.Tensor,
        value_masks: torch.Tensor,
        med_day_feats: Optional[torch.Tensor] = None,
        event_type_ids: Optional[torch.Tensor] = None,
        age_values: Optional[torch.Tensor] = None,
        age_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del time_days  # hour-only time encoding
        val_feat = torch.stack([values.clamp(min=-1e4, max=1e4), value_masks], dim=-1)
        x = self.code_emb(code_ids) + self.modality_emb(modality_ids)
        # Gate value_mlp so tokens without numeric values contribute exactly 0.
        x = x + self.value_mlp(val_feat) * value_masks.unsqueeze(-1)
        if age_values is None:
            age_values = torch.zeros_like(values)
        if age_masks is None:
            age_masks = torch.zeros_like(value_masks)
        age_feat = torch.stack([age_values.clamp(min=-1e4, max=1e4), age_masks], dim=-1)
        x = x + self.age_mlp(age_feat) * age_masks.unsqueeze(-1)
        if event_type_ids is None:
            event_type_ids = torch.zeros_like(code_ids)
        x = x + self.event_type_emb(event_type_ids)
        if self.use_time_encoding:
            time_feat = (time_hours / 24.0).unsqueeze(-1)
            x = x + self.time_mlp(time_feat)
        if self.med_exposure_proj is not None:
            if med_day_feats is None:
                raise ValueError("med_day_feats is required when num_med_exposure > 0")
            x = x + self.med_exposure_proj(med_day_feats)
        return self.pre_ln(x)

    def forward(
        self,
        code_ids: torch.Tensor,
        modality_ids: torch.Tensor,
        time_days: torch.Tensor,
        time_hours: torch.Tensor,
        values: torch.Tensor,
        value_masks: torch.Tensor,
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
        med_day_feats: Optional[torch.Tensor] = None,
        event_type_ids: Optional[torch.Tensor] = None,
        age_values: Optional[torch.Tensor] = None,
        age_masks: Optional[torch.Tensor] = None,
        preop_mask: Optional[torch.Tensor] = None,
        postop_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del preop_mask, postop_mask
        x = self.embed_tokens(
            code_ids=code_ids,
            modality_ids=modality_ids,
            time_days=time_days,
            time_hours=time_hours,
            values=values,
            value_masks=value_masks,
            med_day_feats=med_day_feats,
            event_type_ids=event_type_ids,
            age_values=age_values,
            age_masks=age_masks,
        )
        if self.attention_mode == "causal":
            B, L = attention_mask.shape
            block_mask = build_causal_attention_mask(B, L, attention_mask.device, attention_mask)
        else:
            block_mask = build_periop_block_attention_mask(attention_mask, base_mask)
        h = self.encoder(x, block_mask)
        online_logits = self.online_head(h).squeeze(-1)
        baseline_logits_all = self.baseline_head(h).squeeze(-1)
        mem_logits = self.mem_head(h)
        return {
            "token_logits": online_logits,
            "baseline_logits_all": baseline_logits_all,
            "mem_logits": mem_logits,
            "hidden": h,
            "attention_block_mask": block_mask,
        }


OnlinePeriopEventModel = PerioperativeBlockMaskModel
