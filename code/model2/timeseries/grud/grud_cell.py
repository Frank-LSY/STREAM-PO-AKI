"""GRU-D cell (Che et al.) — input/hidden decay for irregular observations."""
from __future__ import annotations

import torch
import torch.nn as nn


def _exp_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(-torch.relu(x))


class GRUDCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.0, decay_hidden_update: bool = False):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.dropout = float(dropout)
        # Che et al.: the decayed hidden state replaces h_{t-1} in the update; False keeps legacy checkpoints.
        self.decay_hidden_update = bool(decay_hidden_update)

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)
        self.masking_kernel = nn.Linear(input_size, hidden_size * 3)

        self.input_decay_kernel = nn.Parameter(torch.zeros(input_size))
        self.input_decay_bias = nn.Parameter(torch.zeros(input_size))
        self.hidden_decay_kernel = nn.Parameter(torch.zeros(input_size, hidden_size))
        self.hidden_decay_bias = nn.Parameter(torch.zeros(hidden_size))
        if self.decay_hidden_update:
            # All-zero init puts relu at 0 where its gradient is 0, so the decays would never train.
            # Elapsed time is in days: gamma_x ~ exp(-0.025 * days), gamma_h starts near 0.8.
            nn.init.uniform_(self.input_decay_kernel, 0.0, 0.05)
            nn.init.uniform_(self.hidden_decay_kernel, 0.0, 1e-4)

    def forward(
        self,
        x: torch.Tensor,
        m: torch.Tensor,
        times: torch.Tensor,
        h_prev: torch.Tensor,
        x_keep_prev: torch.Tensor,
        s_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step. Shapes: x,m,x_keep_prev,s_prev [B,F]; times [B,1]; h_prev [B,H]."""
        input_d = times - s_prev
        input_1m = 1.0 - m.float()

        gamma_di = _exp_relu(input_d * self.input_decay_kernel + self.input_decay_bias)
        gamma_dh = _exp_relu(torch.mm(input_d, self.hidden_decay_kernel) + self.hidden_decay_bias)

        x_keep = torch.where(m.bool(), x, x_keep_prev)
        x_decayed = torch.where(m.bool(), x, gamma_di * x_keep)
        h_decayed = gamma_dh * h_prev

        m_contribution = self.masking_kernel(input_1m)
        h_size = self.hidden_size
        z = torch.sigmoid(
            self.W_z(torch.cat([x_decayed, h_decayed], dim=-1)) + m_contribution[:, :h_size]
        )
        r = torch.sigmoid(
            self.W_r(torch.cat([x_decayed, h_decayed], dim=-1))
            + m_contribution[:, h_size : 2 * h_size]
        )
        h_tilde = torch.tanh(
            self.W_h(torch.cat([x_decayed, r * h_decayed], dim=-1))
            + m_contribution[:, 2 * h_size :]
        )
        h_new = z * (h_decayed if self.decay_hidden_update else h_prev) + (1.0 - z) * h_tilde
        s_new = torch.where(m.bool(), times.expand(-1, self.input_size), s_prev)
        return h_new, x_keep, s_new
