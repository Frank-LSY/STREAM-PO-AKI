from __future__ import annotations
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional
import numpy as np

EPS = 1e-6


def fit_lab_zscore_from_episodes(
    episodes: List[Dict[str, Any]],
    train_indices: Iterable[int],
) -> Dict[str, Any]:
    idx_set = {int(i) for i in train_indices}
    by_code: Dict[str, List[float]] = defaultdict(list)
    all_vals: List[float] = []
    for i in idx_set:
        ep = episodes[i]
        for ev in ep.get("events", []):
            if str(ev.get("modality", "")).lower() != "lab":
                continue
            v = ev.get("value")
            if v is None:
                continue
            code = str(ev.get("code", "<UNK_LAB>"))
            fv = float(v)
            by_code[code].append(fv)
            all_vals.append(fv)
    per_code: Dict[str, Dict[str, float]] = {}
    for code, vals in by_code.items():
        arr = np.asarray(vals, dtype=np.float64)
        mean = float(arr.mean())
        std = float(arr.std())
        if std < EPS or not np.isfinite(std):
            std = 1.0
        per_code[code] = {"mean": mean, "std": std}
    if all_vals:
        arr = np.asarray(all_vals, dtype=np.float64)
        g_mean = float(arr.mean())
        g_std = float(arr.std())
        if g_std < EPS or not np.isfinite(g_std):
            g_std = 1.0
    else:
        g_mean, g_std = 0.0, 1.0
    return {"per_code": per_code, "global_mean": g_mean, "global_std": g_std, "version": 1}


def apply_lab_zscore(raw: float, lab_code: str, lab_stats: Optional[Dict[str, Any]]) -> float:
    if lab_stats is None:
        return float(raw)
    pc = lab_stats.get("per_code") or {}
    if lab_code in pc:
        m = float(pc[lab_code]["mean"])
        s = float(pc[lab_code]["std"])
    else:
        m = float(lab_stats.get("global_mean", 0.0))
        s = float(lab_stats.get("global_std", 1.0))
    if s < EPS or not np.isfinite(s):
        s = 1.0
    z = (float(raw) - m) / s
    return float(z) if np.isfinite(z) else 0.0
