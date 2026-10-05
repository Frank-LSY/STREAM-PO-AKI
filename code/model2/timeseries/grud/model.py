"""6h bucket GRU-D: causal sequence model with missing-value decay."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from timeseries.grud.grud_cell import GRUDCell
from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, SEQ_LEN, bucket_hour_left_for_index, obs_feature_indices


def _slice_indices(layout: Dict[str, Any], key: str) -> tuple[int, int]:
    sl = layout["slices"][key]
    return int(sl[0]), int(sl[1])


def build_observation_mask(x: torch.Tensor, layout: Dict[str, Any]) -> torch.Tensor:
    """Per-feature observation mask [B, L, F] for GRU-D (1 = new observation in bucket)."""
    B, L, _ = x.shape
    device = x.device
    mask = torch.zeros_like(x)

    ccs0, ccs1 = _slice_indices(layout, "ccs")
    ccs = x[:, :, ccs0:ccs1]
    if ccs.size(-1) > 0:
        ccs_obs = torch.zeros_like(ccs)
        ccs_obs[:, 0] = (ccs[:, 0] > 0).float()
        if L > 1:
            ccs_obs[:, 1:] = ((ccs[:, 1:] > 0) & (ccs[:, 1:] != ccs[:, :-1])).float()
        mask[:, :, ccs0:ccs1] = ccs_obs

    lab0, lab1 = _slice_indices(layout, "lab")
    n_lab = (lab1 - lab0) // 2
    for i in range(n_lab):
        miss_idx = lab0 + 2 * i + 1
        mask[:, :, miss_idx] = (x[:, :, miss_idx] > 0).float()

    med0, med1 = _slice_indices(layout, "med")
    med = x[:, :, med0:med1]
    if med.size(-1) > 0:
        med_obs = torch.zeros_like(med)
        med_obs[:, 0] = (med[:, 0] > 0).float()
        if L > 1:
            med_obs[:, 1:] = ((med[:, 1:] > 0) & (med[:, 1:] != med[:, :-1])).float()
        mask[:, :, med0:med1] = med_obs

    he0, _ = _slice_indices(layout, "has_event")
    has_ev = (x[:, :, he0] > 0).float()

    for key in ("age", "sex", "race", "procedure"):
        a, b = _slice_indices(layout, key)
        mask[:, BASE_BUCKET_IDX : BASE_BUCKET_IDX + 1, a:b] = x[
            :, BASE_BUCKET_IDX : BASE_BUCKET_IDX + 1, a:b
        ].clamp(min=0.0, max=1.0)

    mask = torch.maximum(mask, has_ev.unsqueeze(-1).expand_as(mask))
    return mask.clamp(0.0, 1.0)


def build_raw_observation_mask(x: torch.Tensor, obs: torch.Tensor, layout: Dict[str, Any]) -> torch.Tensor:
    """[B, L, F] mask from pre-LOCF observations: CCS / lab values / medications use ``obs``
    (1 = observed in the bucket); all other features (lab indicators, has_event, static) are always observed."""
    mask = torch.ones_like(x)
    idx = torch.as_tensor(obs_feature_indices(layout), device=x.device, dtype=torch.long)
    mask[:, :, idx] = obs.to(x.dtype)
    return mask


class PerioperativeBucketGRUD(nn.Module):
    """Causal GRU-D over fixed 6h buckets; same baseline/online heads as LSTM/Transformer.

    obs_mode="raw": per-feature decay driven by pre-LOCF observation masks (``obs_mask`` input),
    elapsed time in days, and the decayed hidden state in the GRU update.
    obs_mode="legacy": original mask derived from LOCF features (kept for old checkpoints).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        layout: Optional[Dict[str, Any]] = None,
        obs_mode: str = "legacy",
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)
        self.layout = layout
        if obs_mode not in ("legacy", "raw"):
            raise ValueError(f"obs_mode must be legacy|raw, got {obs_mode}")
        self.obs_mode = obs_mode
        self.uses_obs_mask = obs_mode == "raw"

        self.grud = GRUDCell(self.input_dim, d_model, dropout=dropout, decay_hidden_update=self.uses_obs_mask)
        refine_layers = max(0, self.num_layers - 1)
        lstm_dropout = float(dropout) if refine_layers > 1 else 0.0
        self.refine: Optional[nn.GRU] = None
        if refine_layers > 0:
            self.refine = nn.GRU(
                d_model,
                d_model,
                num_layers=refine_layers,
                batch_first=True,
                bidirectional=False,
                dropout=lstm_dropout,
            )

        self.online_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.baseline_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )

        hours = torch.tensor(
            [bucket_hour_left_for_index(i) for i in range(SEQ_LEN)],
            dtype=torch.float32,
        )
        self.register_buffer("_bucket_hours", hours, persistent=False)

    def set_layout(self, layout: Dict[str, Any]) -> None:
        self.layout = layout

    def _bucket_times(self, length: int, device: torch.device) -> torch.Tensor:
        return self._bucket_hours[:length].to(device)

    def forward(
        self,
        bucket_features: torch.Tensor,
        attention_mask: torch.Tensor,
        base_mask: torch.Tensor,
        obs_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del attention_mask, base_mask
        if self.layout is None:
            raise RuntimeError("PerioperativeBucketGRUD requires feature_layout (call set_layout or pass at init).")

        B, L, _ = bucket_features.shape
        device = bucket_features.device
        times = self._bucket_times(L, device).unsqueeze(0).expand(B, -1)
        if self.uses_obs_mask:
            if obs_mask is None:
                raise RuntimeError("obs_mode='raw' requires obs_mask (load dataset with load_obs=True).")
            times = times / 24.0
            obs_mask = build_raw_observation_mask(bucket_features, obs_mask, self.layout)
        else:
            obs_mask = build_observation_mask(bucket_features, self.layout)

        h = torch.zeros(B, self.d_model, device=device)
        x_keep = torch.zeros(B, self.input_dim, device=device)
        s_prev = times[:, 0:1].expand(-1, self.input_dim)

        hidden_steps: List[torch.Tensor] = []
        for t in range(L):
            tt = times[:, t : t + 1]
            h, x_keep, s_prev = self.grud(
                bucket_features[:, t],
                obs_mask[:, t],
                tt,
                h,
                x_keep,
                s_prev,
            )
            hidden_steps.append(h)

        h_seq = torch.stack(hidden_steps, dim=1)
        if self.refine is not None:
            h_seq, _ = self.refine(h_seq)

        return {
            "token_logits": self.online_head(h_seq).squeeze(-1),
            "baseline_logits_all": self.baseline_head(h_seq).squeeze(-1),
            "hidden": h_seq,
        }
