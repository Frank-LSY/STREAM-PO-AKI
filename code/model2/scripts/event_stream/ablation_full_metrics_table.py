"""Supplementary ablation table: complete threshold metrics, mean (min, max) over 5 folds, in %."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import PROJECT_ROOT

RES = (PROJECT_ROOT / "result/event_stream")
ABL = RES / "ablation"
RUNS = [
    ("Full model", "STREAM (full model)", RES / "transformer"),
    ("Architecture", "No time encoding (A1)", ABL / "A1_no_time_encoding"),
    ("Architecture", "Causal attention (A2)", ABL / "A2_causal_attention"),
    ("Architecture", "Procedure token as baseline (A3)", ABL / "A3_procedure_as_base"),
    ("Training objective", "No baseline loss (A4)", ABL / "A4_no_baseline_loss"),
    ("Training objective", "No MEM loss (A5)", ABL / "A5_no_mem"),
    ("Input modality", "No medication events (I1)", ABL / "I1_no_med_events"),
    ("Input modality", "No laboratory events (I2)", ABL / "I2_no_lab_events"),
    ("Input modality", "No condition events (I3)", ABL / "I3_no_conditions"),
    ("Input modality", "Generic surgery token (I4)", ABL / "I4_generic_surgery"),
    ("Interaction", "Procedure-as-baseline + generic surgery (A3+I4)", ABL / "A3I4_procedure_as_base_generic_surgery"),
]
THR = ["accuracy", "specificity", "sensitivity", "precision", "f1"]
STYLE = "sd" if "--sd" in sys.argv else "range"


def fmt(vals):
    # F1 is undefined (None) when a fold predicts no positives; count it as 0 (zero_division=0).
    v = np.nan_to_num(np.asarray([np.nan if x is None else x for x in vals], dtype=float), nan=0.0) * 100
    if STYLE == "sd":
        return f"{v.mean():.2f} ± {v.std(ddof=1):.2f}"
    return f"{v.mean():.2f} ({v.min():.2f}, {v.max():.2f})"


def macro(by_bucket, key):
    return float(np.nanmean([d[key] for d in by_bucket.values() if d.get(key) is not None]))


def macro_zero(by_bucket, key):
    # Threshold metrics undefined in a window (no predicted positives) count as 0, matching zero_division=0.
    return float(np.mean([0.0 if d.get(key) is None else d[key] for d in by_bucket.values()]))


def main():
    base_rows, grid_rows, macro_rows = [], [], []
    for cat, name, root in RUNS:
        b = [json.loads((root / f"eval_baseline_fold{k}_test/baseline_metrics.json").read_text())["metrics"]
             for k in range(1, 6)]
        g = [json.loads((root / f"eval_fixedgrid_fold{k}_test/fixed_time_grid_metrics.json").read_text())
             for k in range(1, 6)]
        base_rows.append({"Category": cat, "Ablation": name,
                          **{m.upper() if m.startswith("au") else m.capitalize(): fmt([x[m] for x in b])
                             for m in ["auroc", "auprc"] + THR}})
        p = [x["pooled_grid"] for x in g]
        grid_rows.append({"Category": cat, "Ablation": name,
                          "Pooled AUROC": fmt([x["auroc"] for x in p]),
                          "Pooled AUPRC": fmt([x["auprc"] for x in p]),
                          "Macro AUROC": fmt([macro(x["by_bucket_hour"], "auroc") for x in g]),
                          "Macro AUPRC": fmt([macro(x["by_bucket_hour"], "auprc") for x in g]),
                          **{m.capitalize(): fmt([x[m] for x in p]) for m in THR}})
        macro_rows.append({"Category": cat, "Ablation": name,
                           "AUROC": fmt([macro(x["by_bucket_hour"], "auroc") for x in g]),
                           "AUPRC": fmt([macro(x["by_bucket_hour"], "auprc") for x in g]),
                           **{m.capitalize(): fmt([macro_zero(x["by_bucket_hour"], m) for x in g]) for m in THR}})
    a = pd.DataFrame(base_rows).rename(columns={"F1": "F1"})
    c = pd.DataFrame(grid_rows)
    d = pd.DataFrame(macro_rows)
    sfx = "_sd" if STYLE == "sd" else ""
    a.to_csv(ABL / f"ablation_full_metrics_baseline{sfx}.csv", index=False)
    c.to_csv(ABL / f"ablation_full_metrics_dynamic_grid{sfx}.csv", index=False)
    d.to_csv(ABL / f"ablation_full_metrics_dynamic_macro{sfx}.csv", index=False)
    pd.set_option("display.width", 400, "display.max_columns", 20)
    print(a.to_string(index=False)); print(); print(c.to_string(index=False)); print(); print(d.to_string(index=False))


if __name__ == "__main__":
    main()
