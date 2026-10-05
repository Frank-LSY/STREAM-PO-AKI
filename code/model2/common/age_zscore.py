"""Train-set z-score for the AGE static token (separate from lab value path)."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np

EPS = 1e-6


def fit_age_zscore_from_episodes(
    episodes: List[Dict[str, Any]],
    train_indices: Iterable[int],
) -> Dict[str, Any]:
    vals: List[float] = []
    idx_set = {int(i) for i in train_indices}
    for i in idx_set:
        ep = episodes[i]
        for ev in ep.get("events", []) or []:
            if str(ev.get("code", "")) != "AGE":
                continue
            v = ev.get("value")
            if v is None:
                continue
            vals.append(float(v))
        for tok in ep.get("static_tokens", []) or []:
            if str(tok.get("code", "")) != "AGE":
                continue
            v = tok.get("value")
            if v is None:
                continue
            vals.append(float(v))
    if vals:
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < EPS or not np.isfinite(std):
            std = 1.0
    else:
        mean, std = 0.0, 1.0
    return {"mean": mean, "std": std, "version": 1}


def apply_age_zscore(raw: float, age_stats: Optional[Dict[str, Any]]) -> float:
    if age_stats is None:
        return float(raw)
    m = float(age_stats.get("mean", 0.0))
    s = float(age_stats.get("std", 1.0))
    if s < EPS or not np.isfinite(s):
        s = 1.0
    z = (float(raw) - m) / s
    return float(z) if np.isfinite(z) else 0.0
