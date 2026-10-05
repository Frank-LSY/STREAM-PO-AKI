#!/usr/bin/env python3
"""Per-fold paired ablation comparison vs the full model (test set + validation selection score)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
ABL = ROOT / "result/event_stream/ablation"
FULL = ROOT / "result/event_stream/transformer"

IDS = [
    ("Full", FULL),
    ("A1", ABL / "A1_no_time_encoding"),
    ("A2", ABL / "A2_causal_attention"),
    ("A3", ABL / "A3_procedure_as_base"),
    ("A4", ABL / "A4_no_baseline_loss"),
    ("A5", ABL / "A5_no_mem"),
    ("I1", ABL / "I1_no_med_events"),
    ("I2", ABL / "I2_no_lab_events"),
    ("I3", ABL / "I3_no_conditions"),
    ("I4", ABL / "I4_generic_surgery"),
    ("A3I4", ABL / "A3I4_procedure_as_base_generic_surgery"),
]
METRICS = [
    "val_score", "base_auroc", "base_auprc", "macro_auprc", "online_auroc", "online_auprc",
    "grid_pooled_auroc", "grid_pooled_auprc", "grid_macro_auroc", "grid_macro_auprc",
]


def fold_row(root: Path, k: int) -> dict:
    b = json.loads((root / f"eval_baseline_fold{k}_test/baseline_metrics.json").read_text())["metrics"]
    p = json.loads((root / f"eval_postop6h_fold{k}_test/metrics.json").read_text())
    oa = p["metrics"]["online_all_valid"]
    es = json.loads((root / f"fold{k}/early_stop.json").read_text())
    # Patient-window grid (one prediction per episode per 6 h window, 0–720 h), same as Table 2.
    g = json.loads((root / f"eval_fixedgrid_fold{k}_test/fixed_time_grid_metrics.json").read_text())
    bk = [v for v in g["by_bucket_hour"].values()
          if v.get("auroc") is not None and np.isfinite(v["auroc"]) and np.isfinite(v["auprc"])]
    return {
        "fold": k,
        "val_score": float(es["best_score"]),
        "epochs": int(es.get("stopped_at_epoch", -1)),
        "base_auroc": float(b["auroc"]),
        "base_auprc": float(b["auprc"]),
        "macro_auprc": float(p["summary"]["macro_online_hour_bucket_auprc"]),
        "online_auroc": float(oa["auroc"]),
        "online_auprc": float(oa["auprc"]),
        "online_n": int(oa.get("n", -1)),
        "grid_pooled_auroc": float(g["pooled_grid"]["auroc"]),
        "grid_pooled_auprc": float(g["pooled_grid"]["auprc"]),
        "grid_macro_auroc": float(np.mean([v["auroc"] for v in bk])),
        "grid_macro_auprc": float(np.mean([v["auprc"] for v in bk])),
        "grid_n_windows": len(bk),
    }


def main() -> None:
    per_fold = []
    for sid, root in IDS:
        for k in range(1, 6):
            r = fold_row(root, k)
            r["id"] = sid
            per_fold.append(r)
    df = pd.DataFrame(per_fold)
    df.to_csv(ABL / "ablation_per_fold.csv", index=False)

    full = df[df.id == "Full"].set_index("fold")
    rows = []
    for sid, _ in IDS:
        s = df[df.id == sid].set_index("fold")
        row = {"id": sid, "epochs_mean": s.epochs.mean()}
        for m in METRICS:
            row[f"{m}_mean"] = s[m].mean()
            row[f"{m}_sd"] = s[m].std(ddof=1)
            if sid != "Full":
                d = s[m] - full[m]
                row[f"{m}_delta"] = d.mean()
                row[f"{m}_folds_worse"] = int((d < 0).sum())
                row[f"{m}_p"] = float(stats.ttest_rel(s[m], full[m]).pvalue)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(ABL / "ablation_paired_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 80)
    print(out.round(4).T.to_string())


if __name__ == "__main__":
    main()
