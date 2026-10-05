#!/usr/bin/env python3
"""Merge time-agnostic event Δp rankings across multiple fold eval outputs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from common.utils import save_json


def _load_ranked_events(summary_path: Path) -> List[Dict[str, Any]]:
    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)
    cohort = data.get("cohort_all") or {}
    ranked = cohort.get("all_ranked_events")
    if ranked:
        return list(ranked)
    # Backward-compat: merge top lists only.
    out: Dict[str, Dict[str, Any]] = {}
    for key in ("risk_increasing_events", "risk_decreasing_events"):
        for row in cohort.get(key) or []:
            ek = row.get("event_key") or f"{row.get('modality')}|{row.get('raw_code')}"
            out[ek] = row
    return list(out.values())


def merge_ranked_event_rows(fold_ranked: Sequence[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Weighted merge of per-fold aggregate rows by event_key."""
    pools_prob: Dict[str, List[float]] = {}
    pools_logit: Dict[str, List[float]] = {}
    weights: Dict[str, List[int]] = {}
    meta: Dict[str, Dict[str, str]] = {}

    for rows in fold_ranked:
        for row in rows:
            key = str(row.get("event_key") or f"{row.get('modality')}|{row.get('raw_code')}")
            n = int(row.get("n_measurements") or 0)
            if n <= 0:
                continue
            mean_dp = float(row["mean_delta_prob"])
            mean_dl = float(row.get("mean_delta_logit") or 0.0)
            pools_prob.setdefault(key, []).append(mean_dp)
            pools_logit.setdefault(key, []).append(mean_dl)
            weights.setdefault(key, []).append(n)
            meta[key] = {
                "modality": str(row.get("modality", "")),
                "raw_code": str(row.get("raw_code", "")),
                "event_key": key,
            }

    merged: List[Dict[str, Any]] = []
    for key, means in pools_prob.items():
        ws = weights[key]
        wsum = float(sum(ws))
        mean_dp = sum(m * w for m, w in zip(means, ws)) / wsum
        mean_dl = sum(m * w for m, w in zip(pools_logit[key], ws)) / wsum
        merged.append(
            {
                **meta[key],
                "n_measurements": int(round(wsum)),
                "n_folds_contributing": len(ws),
                "mean_delta_prob": float(mean_dp),
                "mean_delta_logit": float(mean_dl),
            }
        )
    return merged


def rank_merged_rows(
    merged_rows: List[Dict[str, Any]],
    *,
    min_count: int,
    top_k: int,
) -> Dict[str, Any]:
    ranked = [r for r in merged_rows if int(r.get("n_measurements") or 0) >= min_count]
    pos = sorted(
        [r for r in ranked if float(r["mean_delta_prob"]) > 0],
        key=lambda x: float(x["mean_delta_prob"]),
        reverse=True,
    )[:top_k]
    neg = sorted(
        [r for r in ranked if float(r["mean_delta_prob"]) < 0],
        key=lambda x: float(x["mean_delta_prob"]),
    )[:top_k]
    return {
        "min_count": min_count,
        "n_event_types_ranked": len(ranked),
        "risk_increasing_events": pos,
        "risk_decreasing_events": neg,
        "all_ranked_events": sorted(ranked, key=lambda x: abs(float(x["mean_delta_prob"])), reverse=True),
    }


def write_ranking_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = [
        "modality",
        "raw_code",
        "n_measurements",
        "n_folds_contributing",
        "mean_delta_prob",
        "mean_delta_logit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge event probability contribution summaries across folds.")
    p.add_argument(
        "--fold_dirs",
        type=str,
        nargs="+",
        required=True,
        help="Dirs containing event_probability_contribution_summary.json per fold.",
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_k_report", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_ranked: List[List[Dict[str, Any]]] = []
    fold_paths: List[str] = []
    for d in args.fold_dirs:
        summary = Path(d).resolve() / "event_probability_contribution_summary.json"
        if not summary.is_file():
            raise FileNotFoundError(f"Missing {summary}")
        fold_ranked.append(_load_ranked_events(summary))
        fold_paths.append(str(Path(d).resolve()))

    merged_rows = merge_ranked_event_rows(fold_ranked)
    cohort = rank_merged_rows(merged_rows, min_count=args.min_count, top_k=args.top_k)
    top_pos = cohort["risk_increasing_events"][: args.top_k_report]
    top_neg = cohort["risk_decreasing_events"][: args.top_k_report]

    report = {
        "analysis_type": "event_probability_contribution_time_agnostic_merged_folds",
        "method": "weighted_mean_delta_prob_across_folds",
        "disclaimer": (
            "Merged across fold-specific checkpoints on the same episode set. "
            "Positive mean Δp: event associated with higher AKI risk when present (occlusion attribution)."
        ),
        "fold_dirs": fold_paths,
        "min_count": args.min_count,
        "top_k": args.top_k,
        "top_k_report": args.top_k_report,
        "n_event_types_merged": len(merged_rows),
        "cohort_merged": cohort,
        "top_positive_contributors": top_pos,
        "top_negative_contributors": top_neg,
    }
    save_json(report, out_dir / "event_probability_contribution_merged_summary.json")
    write_ranking_csv(out_dir / f"top{args.top_k_report}_risk_increasing_events.csv", top_pos)
    write_ranking_csv(out_dir / f"top{args.top_k_report}_risk_decreasing_events.csv", top_neg)

    print("=== Merged event → AKI risk contribution (time-agnostic) ===")
    print(f"  folds: {len(fold_paths)}")
    print(f"  event types merged: {len(merged_rows)}")
    print(f"  top {args.top_k_report} risk-increasing:")
    for i, row in enumerate(top_pos, 1):
        print(
            f"    {i}. [{row['modality']}] {row['raw_code']}  "
            f"mean Δp={row['mean_delta_prob']:.6f}  n={row['n_measurements']}"
        )
    print(f"  top {args.top_k_report} risk-decreasing:")
    for i, row in enumerate(top_neg, 1):
        print(
            f"    {i}. [{row['modality']}] {row['raw_code']}  "
            f"mean Δp={row['mean_delta_prob']:.6f}  n={row['n_measurements']}"
        )
    print(f"Wrote {out_dir / 'event_probability_contribution_merged_summary.json'}")


if __name__ == "__main__":
    main()
