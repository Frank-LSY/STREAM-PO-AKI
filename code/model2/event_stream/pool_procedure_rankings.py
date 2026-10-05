#!/usr/bin/env python3
"""Pool per-fold procedure rankings via mean absolute logit, then delta vs median."""
from pathlib import Path
import pandas as pd
from common.paths import PROJECT_ROOT
ROOT = PROJECT_ROOT
frames = []
for f in range(1, 6):
    p = ROOT / f"result/event_stream/transformer/procedure_effects_fold{f}/procedure_counterfactual_ranking.csv"
    t = pd.read_csv(p)
    t["fold"] = f
    frames.append(t)
df = pd.concat(frames, ignore_index=True)
g = df.groupby("procedure_code", as_index=False).agg(
    mean_logit=("mean_logit", "mean"),
    n_panel_patients=("n_panel_patients", "max"),
    n_episodes_actually_this_procedure=("n_episodes_actually_this_procedure", "first"),
    fold_sd_logit=("mean_logit", "std"),
)
g = g.sort_values("mean_logit").reset_index(drop=True)
ref = g.iloc[len(g) // 2]["procedure_code"]
ref_logit = float(g.loc[g.procedure_code == ref, "mean_logit"].iloc[0])
g["mean_delta_logit_vs_reference"] = g["mean_logit"] - ref_logit
g["std_delta_logit_vs_reference"] = g["fold_sd_logit"].fillna(0.0)
g = g.sort_values("mean_delta_logit_vs_reference", ascending=False)
out = ROOT / "result/event_stream/transformer/procedure_effects_pooled"
out.mkdir(parents=True, exist_ok=True)
cols = [
    "procedure_code", "n_episodes_actually_this_procedure", "n_panel_patients",
    "mean_logit", "mean_delta_logit_vs_reference", "std_delta_logit_vs_reference",
]
g[cols].to_csv(out / "procedure_counterfactual_ranking.csv", index=False)
(out / "reference.txt").write_text(ref + "\n")
print(f"pooled n={len(g)} reference={ref} -> {out}")
