#!/usr/bin/env python3
"""Aggregate baseline_metrics.json across folds → CSV with per-fold, mean, min, max."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


METRIC_KEYS = [
    "n",
    "n_pos",
    "n_neg",
    "pos_rate",
    "pos_neg_ratio",
    "auroc",
    "auprc",
    "accuracy",
    "precision",
    "sensitivity",
    "recall",
    "specificity",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "macro_online_hour_bucket_auprc",
]


def dig(data: Any, dotted: str) -> Any:
    """Follow a dotted path through nested dicts, returning None if any hop is missing."""
    cur = data
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def load_fold_metrics(
    path: Path,
    metrics_json: str = "baseline_metrics.json",
    metrics_key: str = "metrics",
    extra: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    json_path = path / metrics_json if path.is_dir() else path
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    m = dict(dig(data, metrics_key) or {})
    cm = m.pop("confusion_matrix", None) or {}
    for k in ("tn", "fp", "fn", "tp"):
        if k not in m and k in cm:
            m[k] = cm[k]
    m.pop("confusion_matrix_2x2", None)
    for name, dotted in (extra or {}).items():
        v = dig(data, dotted)
        if v is not None:
            m[name] = v
    return m


def aggregate_folds(
    fold_dirs: List[Path],
    output_csv: Path,
    metrics_json: str = "baseline_metrics.json",
    fold_labels: Optional[List[str]] = None,
    metrics_key: str = "metrics",
    extra: Optional[Dict[str, str]] = None,
) -> None:
    rows_by_metric: Dict[str, List[Optional[float]]] = {k: [] for k in METRIC_KEYS}
    labels: List[str] = []

    for i, d in enumerate(fold_dirs):
        json_path = d / metrics_json
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing {json_path}")
        if fold_labels is not None and i < len(fold_labels):
            labels.append(fold_labels[i])
        else:
            labels.append(d.name)
        m = load_fold_metrics(d, metrics_json, metrics_key=metrics_key, extra=extra)
        for k in METRIC_KEYS:
            v = m.get(k)
            rows_by_metric[k].append(float(v) if v is not None else np.nan)

    out_rows: List[Dict[str, Any]] = []
    for metric in METRIC_KEYS:
        vals = np.asarray(rows_by_metric[metric], dtype=np.float64)
        finite = vals[np.isfinite(vals)]
        row: Dict[str, Any] = {"metric": metric}
        for i, label in enumerate(labels):
            v = rows_by_metric[metric][i]
            row[label] = "" if v is None or (isinstance(v, float) and not np.isfinite(v)) else v
        if len(finite) > 0:
            row["mean"] = float(np.mean(finite))
            row["min"] = float(np.min(finite))
            row["max"] = float(np.max(finite))
        else:
            row["mean"] = row["min"] = row["max"] = ""
        out_rows.append(row)

    fieldnames = ["metric"] + labels + ["mean", "min", "max"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {output_csv} ({len(out_rows)} metrics, {len(labels)} folds)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate baseline eval JSONs across folds.")
    p.add_argument(
        "--result_root",
        type=str,
        default="../../result",
        help="Directory containing baseline_eval_nocross_fold*_test folders.",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="baseline_eval_nocross_fold{fold}_test",
        help="Folder name pattern; {fold} = 1..5 (e.g. xgboost_eval_fold{fold}_test).",
    )
    p.add_argument(
        "--label_pattern",
        type=str,
        default=None,
        help="CSV column names; default = folder basename from --pattern.",
    )
    p.add_argument(
        "--metrics_json",
        type=str,
        default="baseline_metrics.json",
        help="Metrics JSON filename inside each fold directory.",
    )
    p.add_argument(
        "--metrics_key",
        type=str,
        default="metrics",
        help="Dotted path to the metrics block inside the JSON (e.g. pooled_grid).",
    )
    p.add_argument(
        "--extra_metric",
        action="append",
        default=[],
        metavar="NAME=DOTTED.PATH",
        help="Pull an additional scalar into row NAME; repeatable.",
    )
    p.add_argument("--folds", type=str, default="1,2,3,4,5")
    p.add_argument(
        "--output_csv",
        type=str,
        default="../../result/baseline_eval_nocross_5fold_summary.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.result_root).resolve()
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    fold_dirs = [root / args.pattern.format(fold=f) for f in folds]
    label_pattern = args.label_pattern or args.pattern
    column_labels = [label_pattern.format(fold=f) for f in folds]
    extra: Dict[str, str] = {}
    for spec in args.extra_metric:
        name, _, dotted = spec.partition("=")
        if not name or not dotted:
            raise SystemExit(f"--extra_metric expects NAME=DOTTED.PATH, got {spec!r}")
        extra[name] = dotted
    aggregate_folds(
        fold_dirs,
        Path(args.output_csv).resolve(),
        metrics_json=args.metrics_json,
        fold_labels=column_labels,
        metrics_key=args.metrics_key,
        extra=extra,
    )


if __name__ == "__main__":
    main()
