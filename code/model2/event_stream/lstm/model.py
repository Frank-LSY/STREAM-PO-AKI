"""Event-stream segmented LSTM (preop BiLSTM, BASE fusion, postop causal LSTM)."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from common.med_exposure import NUM_EVENT_TYPES


class PerioperativeBlockMaskLSTM(nn.Module):
    """LSTM event-stream encoder mirroring block-mask information flow.

    Sequences follow ``preop | [BASE] | postop | padding`` (left-aligned per batch).
    Encoding uses batched ``pack_padded_sequence`` over variable-length segments.
    """

    def __init__(
        self,
        vocab_size: int,
        num_modalities: int = 6,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_med_exposure: int = 0,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_med_exposure = int(num_med_exposure)
        self.code_emb = nn.Embedding(vocab_size, d_model)
        self.modality_emb = nn.Embedding(num_modalities, d_model)
        self.event_type_emb = nn.Embedding(NUM_EVENT_TYPES, d_model, padding_idx=0)
        self.time_mlp = nn.Sequential(nn.Linear(1, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.value_mlp = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.age_mlp = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        self.med_exposure_proj: Optional[nn.Linear]
        if self.num_med_exposure > 0:
            self.med_exposure_proj = nn.Linear(self.num_med_exposure, d_model)
        else:
            self.med_exposure_proj = None
        self.pre_ln = nn.LayerNorm(d_model)

        preop_hidden = d_model // 2
        lstm_dropout = float(dropout) if int(num_layers) > 1 else 0.0
        self.preop_lstm = nn.LSTM(
            d_model,
            preop_hidden,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.base_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.postop_lstm = nn.LSTM(
            d_model,
            d_model,
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=False,
            dropout=lstm_dropout,
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
        del time_days
        val_feat = torch.stack([values.clamp(min=-1e4, max=1e4), value_masks], dim=-1)
        if event_type_ids is None:
            event_type_ids = torch.zeros_like(code_ids)
        if age_values is None:
            age_values = torch.zeros_like(values)
        if age_masks is None:
            age_masks = torch.zeros_like(value_masks)
        age_feat = torch.stack([age_values.clamp(min=-1e4, max=1e4), age_masks], dim=-1)
        x = (
            self.code_emb(code_ids)
            + self.modality_emb(modality_ids)
            + self.event_type_emb(event_type_ids)
            + self.value_mlp(val_feat) * value_masks.unsqueeze(-1)
            + self.age_mlp(age_feat) * age_masks.unsqueeze(-1)
            + self.time_mlp((time_hours / 24.0).unsqueeze(-1))
        )
        if self.med_exposure_proj is not None:
            if med_day_feats is None:
                raise ValueError("med_day_feats is required when num_med_exposure > 0")
            x = x + self.med_exposure_proj(med_day_feats)
        return self.pre_ln(x)

    @staticmethod
    def _segment_lengths(
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (base_idx, preop_lens, postop_lens) for left-aligned preop|BASE|postop layout."""
        B, L = attention_mask.shape
        device = attention_mask.device
        seq_lens = attention_mask.sum(dim=1).long()
        has_base = base_mask.sum(dim=1) > 0
        base_idx = base_mask.argmax(dim=-1)
        base_idx = torch.where(has_base, base_idx, seq_lens.clamp(max=L - 1))
        preop_lens = base_idx
        postop_lens = (seq_lens - base_idx - 1).clamp(min=0)
        return base_idx, preop_lens, postop_lens

    def _run_packed_lstm(
        self,
        lstm: nn.LSTM,
        seq: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Run LSTM on left-aligned ``seq`` [B, T, D] with per-row ``lengths`` (may be 0)."""
        B, T, D = seq.shape
        if T == 0:
            hidden = lstm.hidden_size * (2 if lstm.bidirectional else 1)
            return seq.new_zeros(B, 0, hidden)

        lengths = lengths.long()
        safe_lens = lengths.clamp(min=1)
        packed = pack_padded_sequence(seq, safe_lens.cpu(), batch_first=True, enforce_sorted=False)
        out_packed, _ = lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=T)
        empty = lengths == 0
        if empty.any():
            out = out.clone()
            out[empty] = 0.0
        return out

    def _encode_segments(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
        preop_mask: torch.Tensor,
        postop_mask: torch.Tensor,
    ) -> torch.Tensor:
        del preop_mask, postop_mask  # layout inferred from base_mask + attention_mask.
        B, L, D = x.shape
        h = x.new_zeros(B, L, D)
        base_idx, preop_lens, postop_lens = self._segment_lengths(attention_mask, base_mask)
        batch_arange = torch.arange(B, device=x.device)

        max_preop = int(preop_lens.max().item())
        if max_preop > 0:
            preop_x = x[:, :max_preop, :]
            preop_h = self._run_packed_lstm(self.preop_lstm, preop_x, preop_lens)
            h[:, :max_preop, :] = preop_h
            last_idx = (preop_lens - 1).clamp(min=0)
            preop_last = preop_h[batch_arange, last_idx]
            preop_last = preop_last * (preop_lens > 0).unsqueeze(-1).to(preop_last.dtype)
        else:
            preop_last = x.new_zeros(B, D)

        base_x = x[batch_arange, base_idx.clamp(max=L - 1)]
        base_h = self.base_fusion(torch.cat([preop_last, base_x], dim=-1))
        h[batch_arange, base_idx.clamp(max=L - 1)] = base_h

        max_postop = int(postop_lens.max().item())
        if max_postop > 0:
            offsets = torch.arange(max_postop, device=x.device).view(1, max_postop)
            src_idx = base_idx.unsqueeze(1) + 1 + offsets
            valid_post = offsets < postop_lens.unsqueeze(1)
            src_idx = src_idx.clamp(max=L - 1)
            postop_x = x.gather(1, src_idx.unsqueeze(-1).expand(B, max_postop, D))
            postop_x = postop_x * valid_post.unsqueeze(-1).to(postop_x.dtype)

            postop_in = x.new_zeros(B, 1 + max_postop, D)
            postop_in[:, 0, :] = base_h
            postop_in[:, 1 : 1 + max_postop, :] = postop_x

            in_lens = postop_lens + 1
            postop_out = self._run_packed_lstm(self.postop_lstm, postop_in, in_lens)
            postop_h = postop_out[:, 1 : 1 + max_postop, :]
            postop_h = postop_h * valid_post.unsqueeze(-1).to(postop_h.dtype)
            h.scatter_(1, src_idx.unsqueeze(-1).expand(B, max_postop, D), postop_h)

        return h

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
        if preop_mask is None or postop_mask is None:
            valid = attention_mask > 0
            if postop_mask is None:
                postop_mask = valid.new_zeros(valid.shape)
            if preop_mask is None:
                preop_mask = valid & (base_mask <= 0) & (postop_mask <= 0)
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
        h = self._encode_segments(x, attention_mask, base_mask, preop_mask, postop_mask)
        online_logits = self.online_head(h).squeeze(-1)
        baseline_logits_all = self.baseline_head(h).squeeze(-1)
        mem_logits = self.mem_head(h)
        return {
            "token_logits": online_logits,
            "baseline_logits_all": baseline_logits_all,
            "mem_logits": mem_logits,
            "hidden": h,
        }
