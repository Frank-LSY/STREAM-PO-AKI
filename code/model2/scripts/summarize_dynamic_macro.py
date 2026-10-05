"""Macro-averaged postoperative dynamic performance (Table 2 / Supplementary Table S5).

Each metric is computed within each of the 120 six-hour windows (0-720 h), averaged with equal
weight per window, then summarized as mean (min, max) across the five fold models.
"""
import json

import numpy as np
import pandas as pd

from common.paths import PROJECT_ROOT

RES = PROJECT_ROOT / "result"
METRICS = ["auroc", "auprc", "accuracy", "specificity", "sensitivity", "precision", "f1"]
MODELS = [
    ("LSTM", lambda k: RES / f"timeseries/lstm/eval_postop6h_fold{k}_test/postop_hour_bucket_metrics.json"),
    ("GRU-D", lambda k: RES / f"timeseries/grud/eval_postop6h_fold{k}_test/postop_hour_bucket_metrics.json"),
    ("Time-series Transformer",
     lambda k: RES / f"timeseries/transformer/eval_postop6h_fold{k}_test/postop_hour_bucket_metrics.json"),
    ("STREAM", lambda k: RES / f"event_stream/transformer/eval_fixedgrid_fold{k}_test/fixed_time_grid_metrics.json"),
]


def windows(path):
    d = json.loads(path.read_text())
    if "by_bucket_hour" in d:
        return d["by_bucket_hour"]
    return d["full_metrics"]["by_postop_hour_bucket"]


def window_mean(by_window, key):
    vals = [d.get(key) for d in by_window.values()]
    if key in ("auroc", "auprc"):
        return float(np.mean([v for v in vals if v is not None]))
    # Threshold metrics undefined in a window (no predicted positives) count as 0, matching zero_division=0.
    return float(np.mean([0.0 if v is None else v for v in vals]))


def main():
    rows = []
    for name, path in MODELS:
        per_fold = [windows(path(k)) for k in range(1, 6)]
        row = {"model": name}
        for m in METRICS:
            v = np.array([window_mean(w, m) for w in per_fold]) * 100
            row[m] = f"{v.mean():.2f} ({v.min():.2f}, {v.max():.2f})"
            row.update({f"{m}_fold{k + 1}": round(float(x), 4) for k, x in enumerate(v)})
        rows.append(row)
    out = pd.DataFrame(rows)
    dst = RES / "dynamic_macro_5fold_summary.csv"
    out.to_csv(dst, index=False)
    print(out[["model"] + METRICS].to_string(index=False))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
