#!/usr/bin/env python3
"""Pool per-fold event-arrival code summaries with sign agreement.

Each fold's ``mean_delta_logit`` is the patient-pooled mean of order-robust
arrival attributions ``a_i`` (sequential Δz when |G_t|=1; Shapley-MC φ̂ when
|G_t|>1). Cross-fold pooling is patient-weighted across folds.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "result/event_stream/transformer/event_arrival_pooled"
MIN_PATIENTS = 50


def main() -> None:
    frames = []
    fold_meta = {}
    for f in range(1, 6):
        fold_dir = ROOT / f"result/event_stream/transformer/event_arrival_fold{f}"
        p = fold_dir / "event_arrival_by_code.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        t = pd.read_csv(p)
        t["fold"] = f
        frames.append(t)
        sp = fold_dir / "event_arrival_summary.json"
        if sp.exists():
            fold_meta[f] = json.loads(sp.read_text())
    df = pd.concat(frames, ignore_index=True)

    rows = []
    for (mod, code), g in df.groupby(["modality", "raw_code"]):
        if len(g) < 5:
            continue
        means = g.mean_delta_logit.to_numpy(dtype=float)
        ns = g.n_patients.to_numpy(dtype=float)
        # patient-weighted mean across folds
        w = ns / ns.sum()
        pooled = float(np.sum(w * means))
        signs = np.sign(means)
        agree = int(np.all(signs == signs[0]) and signs[0] != 0)
        rows.append(
            {
                "modality": mod,
                "raw_code": code,
                "n_folds": int(len(g)),
                "n_patients_max": int(ns.max()),
                "n_patients_mean": float(ns.mean()),
                "mean_delta_logit": pooled,
                "fold_min": float(means.min()),
                "fold_max": float(means.max()),
                "fold_sd": float(means.std(ddof=1)) if len(means) > 1 else 0.0,
                "folds_agreeing_sign": agree,
                "mean_abs_delta_logit": float(
                    np.sum(w * g.mean_abs_delta_logit.to_numpy(dtype=float))
                ),
            }
        )
    out = pd.DataFrame(rows)
    out = out[out.n_patients_max >= MIN_PATIENTS].copy()
    out = out.sort_values("mean_delta_logit", key=lambda s: s.abs(), ascending=False)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / "event_arrival_by_code_pooled.csv", index=False)
    rep = out[out.folds_agreeing_sign == 1]
    rep.to_csv(OUT / "event_arrival_by_code_replicated.csv", index=False)
    summary = {
        "attribution": "order-robust hybrid a_i (Δz if |G_t|=1; φ̂ if |G_t|>1)",
        "n_pooled_codes": int(len(out)),
        "n_replicated": int(len(rep)),
        "min_patients": MIN_PATIENTS,
        "fold_summaries": {str(k): v for k, v in fold_meta.items()},
    }
    (OUT / "event_arrival_pooled_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"pooled codes={len(out)}  replicated_sign={len(rep)}  -> {OUT}")
    print(rep.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
