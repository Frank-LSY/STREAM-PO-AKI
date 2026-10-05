#!/usr/bin/env python3
"""Rank AKI+/AKI− case-study candidates from the 5-fold sequential screen.

AKI+: low baseline, crosses 0.5 well before PO-AKI diagnosis, rise spread over
several arrivals, all folds agree. AKI−: baseline above 0.5, falls below 0.5
early and stays low through 7 d, fall spread over several arrivals.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
SCREEN = ROOT / "result/event_stream/transformer/case_screen"
PROC_NAMES = ROOT / "data/snomed_procedure_names.json"
UNREADABLE = {"MED_OTHERS", "COND_OTHERS", "LAB_OTHERS"}


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _load():
    eps = pd.read_parquet(SCREEN / "fold1/episodes.parquet")[
        ["person_id", "label_aki", "aki_hours", "procedure_code", "n_postop"]
    ]
    toks, zb = [], []
    for f in range(1, 6):
        t = pd.read_parquet(SCREEN / f"fold{f}/tokens.parquet")
        t["fold"] = f
        toks.append(t)
        e = pd.read_parquet(SCREEN / f"fold{f}/episodes.parquet")[["person_id", "z_base"]]
        e["fold"] = f
        zb.append(e)
    return eps, pd.concat(toks, ignore_index=True), pd.concat(zb, ignore_index=True)


def _dominance(dz: np.ndarray, sign: int) -> float:
    d = dz[np.sign(dz) == sign] * sign
    return float(d.max() / d.sum()) if d.size and d.sum() > 0 else 1.0


def _pos_feats(g: pd.DataFrame, z0: float, aki_h: float) -> dict | None:
    w = g[g.time_hour <= aki_h + 0.05]
    if len(w) < 12:
        return None
    p = _sig(w.logit.to_numpy())
    p0 = float(_sig(z0))
    above = np.where(p >= 0.5)[0]
    if p0 >= 0.5 or not above.size:
        return None
    ic = int(above[0])
    t_cross = float(w.time_hour.iloc[ic])
    return {
        "p0": p0,
        "p_end": float(p[-1]),
        "t_cross": t_cross,
        "lead_h": aki_h - t_cross,
        "stay_above": float((p[ic:] >= 0.5).mean()),
        "dom": _dominance(w.delta_logit.to_numpy(), +1),
        "n": len(w),
    }


def _neg_feats(g: pd.DataFrame, z0: float, horizon: float = 48.0) -> dict | None:
    w = g[g.time_hour <= horizon]
    if len(w) < 12:
        return None
    p = _sig(w.logit.to_numpy())
    p0 = float(_sig(z0))
    below = np.where(p < 0.5)[0]
    if p0 < 0.5 or not below.size:
        return None
    ic = int(below[0])
    late = g[(g.time_hour > horizon)]
    p_late_max = float(_sig(late.logit.to_numpy()).max()) if len(late) else 0.0
    return {
        "p0": p0,
        "p_end": float(p[-1]),
        "t_cross": float(w.time_hour.iloc[ic]),
        "stay_below": float((p[ic:] < 0.5).mean()),
        "p_late_max": p_late_max,
        "dom": _dominance(w.delta_logit.to_numpy(), -1),
        "n": len(w),
    }


def _drivers(g: pd.DataFrame, t_max: float, sign: int, k: int = 5) -> list[str]:
    w = g[(g.time_hour <= t_max) & (np.sign(g.delta_logit) == sign)]
    w = w[~w.raw_code.isin(UNREADABLE)]
    w = w.reindex(w.delta_logit.abs().sort_values(ascending=False).index).head(k)
    return [f"{r.modality[0].upper()}:{r.raw_code}@{r.time_hour:.1f}h({r.delta_logit:+.2f})" for r in w.itertuples()]


def main() -> None:
    eps, toks, zb = _load()
    names = json.loads(PROC_NAMES.read_text()) if PROC_NAMES.exists() else {}
    zb = zb.set_index(["person_id", "fold"]).z_base
    toks = toks.sort_values(["fold", "person_id", "step"])
    groups = {k: g for k, g in toks.groupby(["person_id", "fold"], sort=False)}

    pos_rows, neg_rows = [], []
    for r in eps.itertuples():
        pid = int(r.person_id)
        if r.label_aki == 1:
            if r.aki_hours is None or not np.isfinite(r.aki_hours) or not (6.0 <= r.aki_hours <= 48.0):
                continue
            feats = []
            for f in range(1, 6):
                g = groups.get((pid, f))
                x = None if g is None else _pos_feats(g, float(zb[(pid, f)]), float(r.aki_hours))
                if x is None:
                    break
                feats.append(x)
            if len(feats) < 5:
                continue
            df = pd.DataFrame(feats)
            m = df.mean()
            if not (
                0.08 <= df.p0.max() and df.p0.max() < 0.45
                and df.lead_h.min() >= 3.0
                and df.stay_above.min() >= 0.75
                and 0.6 <= m.p_end <= 0.997
                and m.dom <= 0.6
                and 12 <= m.n <= 150
            ):
                continue
            g1 = groups[(pid, 1)]
            pos_rows.append({
                "person_id": pid, "aki_h": r.aki_hours, "proc": names.get(r.procedure_code, r.procedure_code),
                **{k: round(float(v), 3) for k, v in m.items()},
                "lead_min": round(float(df.lead_h.min()), 2),
                "score": m.lead_h / max(r.aki_hours, 1) + (1 - m.dom) + 0.5 * (0.5 - m.p0),
                "drivers": _drivers(g1, float(r.aki_hours), +1),
            })
        else:
            feats = []
            for f in range(1, 6):
                g = groups.get((pid, f))
                x = None if g is None else _neg_feats(g, float(zb[(pid, f)]))
                if x is None:
                    break
                feats.append(x)
            if len(feats) < 5:
                continue
            df = pd.DataFrame(feats)
            m = df.mean()
            if not (
                df.p0.min() >= 0.55
                and m.p_end <= 0.25
                and df.stay_below.min() >= 0.85
                and df.p_late_max.max() < 0.5
                and 2.0 <= m.t_cross <= 36.0
                and m.dom <= 0.6
                and 12 <= m.n <= 150
            ):
                continue
            g1 = groups[(pid, 1)]
            neg_rows.append({
                "person_id": pid, "proc": names.get(r.procedure_code, r.procedure_code),
                **{k: round(float(v), 3) for k, v in m.items()},
                "score": (m.p0 - m.p_end) + (1 - m.dom),
                "drivers": _drivers(g1, 48.0, -1),
            })

    pos = pd.DataFrame(pos_rows, columns=["person_id", "score"] if not pos_rows else None)
    neg = pd.DataFrame(neg_rows, columns=["person_id", "score"] if not neg_rows else None)
    pos = pos.sort_values("score", ascending=False)
    neg = neg.sort_values("score", ascending=False)
    pos.to_csv(SCREEN / "candidates_pos.csv", index=False)
    neg.to_csv(SCREEN / "candidates_neg.csv", index=False)
    pd.set_option("display.width", 250, "display.max_colwidth", 200)
    print(f"AKI+ candidates: {len(pos)}")
    print(pos.head(15).to_string(index=False))
    print(f"\nAKI− candidates: {len(neg)}")
    print(neg.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
