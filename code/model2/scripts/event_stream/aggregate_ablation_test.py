#!/usr/bin/env python3
"""Build cross-ablation test comparison CSV from per-ablation 5-fold summaries."""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
ABL = ROOT / "result/event_stream/ablation"
FULL = ROOT / "result/event_stream/transformer"

IDS = [
    ("Full", FULL, True),
    ("A1", ABL / "A1_no_time_encoding", False),
    ("A2", ABL / "A2_causal_attention", False),
    ("A3", ABL / "A3_procedure_as_base", False),
    ("A4", ABL / "A4_no_baseline_loss", False),
    ("A5", ABL / "A5_no_mem", False),
    ("I1", ABL / "I1_no_med_events", False),
    ("I2", ABL / "I2_no_lab_events", False),
    ("I3", ABL / "I3_no_conditions", False),
    ("I4", ABL / "I4_generic_surgery", False),
    ("A3I4", ABL / "A3I4_procedure_as_base_generic_surgery", False),
]


def mean_from_summary(csv_path: Path, metric: str):
    if not csv_path.exists():
        return None
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row.get("metric") == metric:
                v = row.get("mean", "")
                try:
                    return float(v)
                except Exception:
                    return None
    return None


def full_postop_means():
    macros, auprcs, aurocs, f1s, sens, specs = [], [], [], [], [], []
    for k in range(1, 6):
        mj = FULL / f"eval_postop6h_fold{k}_test" / "metrics.json"
        if not mj.exists():
            continue
        d = json.loads(mj.read_text())
        macros.append(float(d["summary"]["macro_online_hour_bucket_auprc"]))
        oa = d["metrics"]["online_all_valid"]
        auprcs.append(float(oa["auprc"]))
        aurocs.append(float(oa["auroc"]))
        f1s.append(float(oa["f1"]))
        sens.append(float(oa["sensitivity"]))
        specs.append(float(oa["specificity"]))
    if not macros:
        return {}
    return {
        "postop_macro_hour_bucket_auprc": st.mean(macros),
        "postop_online_all_auprc": st.mean(auprcs),
        "postop_online_all_auroc": st.mean(aurocs),
        "postop_f1": st.mean(f1s),
        "postop_sensitivity": st.mean(sens),
        "postop_specificity": st.mean(specs),
    }


def main() -> None:
    rows = []
    full_post = full_postop_means()
    for short, base, is_full in IDS:
        if is_full:
            bcsv = base / "eval_baseline_nocross_5fold_summary.csv"
            row = {
                "id": short,
                "baseline_auprc": mean_from_summary(bcsv, "auprc"),
                "baseline_auroc": mean_from_summary(bcsv, "auroc"),
                "baseline_f1": mean_from_summary(bcsv, "f1"),
                "baseline_sensitivity": mean_from_summary(bcsv, "sensitivity"),
                "baseline_specificity": mean_from_summary(bcsv, "specificity"),
                **full_post,
            }
        else:
            bcsv = base / "eval_baseline_5fold_summary.csv"
            pcsv = base / "eval_postop6h_5fold_summary.csv"
            row = {
                "id": short,
                "baseline_auprc": mean_from_summary(bcsv, "auprc"),
                "baseline_auroc": mean_from_summary(bcsv, "auroc"),
                "baseline_f1": mean_from_summary(bcsv, "f1"),
                "baseline_sensitivity": mean_from_summary(bcsv, "sensitivity"),
                "baseline_specificity": mean_from_summary(bcsv, "specificity"),
                "postop_online_all_auprc": mean_from_summary(pcsv, "auprc"),
                "postop_online_all_auroc": mean_from_summary(pcsv, "auroc"),
                "postop_macro_hour_bucket_auprc": mean_from_summary(
                    pcsv, "macro_online_hour_bucket_auprc"
                ),
                "postop_f1": mean_from_summary(pcsv, "f1"),
                "postop_sensitivity": mean_from_summary(pcsv, "sensitivity"),
                "postop_specificity": mean_from_summary(pcsv, "specificity"),
            }
        rows.append(row)

    fields = [
        "id",
        "baseline_auroc",
        "baseline_auprc",
        "baseline_f1",
        "baseline_sensitivity",
        "baseline_specificity",
        "postop_online_all_auroc",
        "postop_online_all_auprc",
        "postop_macro_hour_bucket_auprc",
        "postop_f1",
        "postop_sensitivity",
        "postop_specificity",
    ]
    out = ABL / "ablation_test_comparison.csv"
    ABL.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"Wrote {out}")
    for r in rows:
        print(
            f"{r['id']:<5} base_auprc={r.get('baseline_auprc')}  "
            f"macro={r.get('postop_macro_hour_bucket_auprc')}  "
            f"online_auprc={r.get('postop_online_all_auprc')}"
        )


if __name__ == "__main__":
    main()
