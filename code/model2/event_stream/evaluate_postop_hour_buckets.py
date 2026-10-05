#!/usr/bin/env python3
"""Evaluate model2 with postoperative hour buckets (default 6h, aligned with training validation).

Uses token-level online predictions and ``floor(time_hour / bin_width)`` on cumulative
postoperative ``time_hour`` (see ``utils.postop_continuous_hours``).

Writes:
  - metrics.json (full ``evaluate_online`` output + summary)
  - hour_bucket_metrics.csv (optional)
  - prints per-bucket table when ``--print_table``
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from event_stream.eval_common import run_online_evaluation
from common.utils import (
    macro_mean_postop_hour_bucket_auprc,
    save_json,
    validation_combined_selection_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate model2 online AKI metrics by postoperative hour bucket (default 6h)."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for metrics.json and optional CSV.",
    )
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--postop_hour_bin",
        type=float,
        default=6.0,
        help="Hour bucket width (same as training --postop_val_hour_bin). Default 6.",
    )
    p.add_argument("--day_bin_width", type=float, default=1.0)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument(
        "--macro_min_n",
        type=int,
        default=10,
        help="Min tokens per hour bucket for macro AUPRC (same default as training selection).",
    )
    p.add_argument(
        "--selection_lambda",
        type=float,
        default=None,
        help="λ for combined score; default reads checkpoint args best_selection_lambda or 0.5.",
    )
    p.add_argument("--export_csv", action="store_true", help="Write hour_bucket_metrics.csv.")
    p.add_argument("--print_table", action="store_true", help="Print hour-bucket table to stdout.")
    return p.parse_args()


def build_hour_bucket_table(
    metrics: Dict[str, Any],
    bin_width: float,
) -> List[Dict[str, Any]]:
    buckets = metrics.get("by_postop_hour_bucket") or {}
    rows: List[Dict[str, Any]] = []
    for key in sorted(buckets.keys(), key=lambda x: int(x)):
        b = int(key)
        stats = buckets[key]
        if not isinstance(stats, dict):
            continue
        h0 = b * bin_width
        h1 = (b + 1) * bin_width
        rows.append(
            {
                "bucket": b,
                "hour_start": h0,
                "hour_end": h1,
                "hour_range": f"[{h0:g},{h1:g})",
                "n": int(stats.get("n", 0)),
                "n_pos": stats.get("n_pos"),
                "n_neg": stats.get("n_neg"),
                "pos_rate": stats.get("pos_rate"),
                "pos_neg_ratio": stats.get("pos_neg_ratio"),
                "auprc": stats.get("auprc"),
                "auroc": stats.get("auroc"),
                "accuracy": stats.get("accuracy"),
                "precision": stats.get("precision"),
                "recall": stats.get("recall"),
                "f1": stats.get("f1"),
                "specificity": stats.get("specificity"),
            }
        )
    return rows


def enrich_metrics(
    metrics: Dict[str, Any],
    train_args: Dict[str, Any],
    checkpoint: str,
    episode_path: str,
    postop_hour_bin: float,
    macro_min_n: int,
    selection_lambda: Optional[float],
) -> Dict[str, Any]:
    lam = selection_lambda
    if lam is None:
        lam = float(train_args.get("best_selection_lambda", 0.5))
    macro = macro_mean_postop_hour_bucket_auprc(metrics, min_n=macro_min_n)
    sel_score, macro_f, base_f = validation_combined_selection_score(
        metrics, lam=lam, macro_min_n=macro_min_n
    )
    table = build_hour_bucket_table(metrics, postop_hour_bin)
    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "episode_path": str(Path(episode_path).resolve()),
        "time_axis": (
            "cumulative time_hour since surgery; "
            f"bucket = floor(time_hour / {postop_hour_bin:g})"
        ),
        "postop_hour_bin_width": float(postop_hour_bin),
        "macro_min_n": int(macro_min_n),
        "selection_lambda": float(lam),
        "summary": {
            "macro_online_hour_bucket_auprc": macro,
            "baseline_auprc": base_f,
            "validation_combined_selection_score": sel_score,
            "online_all_auprc": (metrics.get("online_all_valid") or {}).get("auprc"),
            "online_all_auroc": (metrics.get("online_all_valid") or {}).get("auroc"),
            "baseline_auprc_node": (metrics.get("baseline_node") or {}).get("auprc"),
            "final_valid_auprc": (metrics.get("final_valid_node") or {}).get("auprc"),
            "n_hour_buckets": len(table),
            "n_hour_buckets_in_macro": sum(
                1
                for r in table
                if r["n"] >= macro_min_n and r.get("auprc") is not None
            ),
        },
        "hour_bucket_table": table,
        "metrics": metrics,
    }


def print_hour_bucket_table(table: List[Dict[str, Any]], macro: Optional[float]) -> None:
    if not table:
        print("No postoperative hour buckets.")
        return
    header = (
        f"{'bucket':>6} {'hours':>14} {'n':>8} {'n+':>7} {'n-':>7} {'pos/neg':>8} "
        f"{'auprc':>8} {'auroc':>8} {'f1':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in table:
        auprc = r["auprc"]
        auroc = r["auroc"]
        f1 = r["f1"]
        ratio = r.get("pos_neg_ratio")
        ratio_s = f"{ratio:.3f}" if ratio is not None else "nan"
        print(
            f"{r['bucket']:>6} {r['hour_range']:>14} {r['n']:>8} "
            f"{r.get('n_pos', ''):>7} {r.get('n_neg', ''):>7} {ratio_s:>8} "
            f"{auprc if auprc is not None else 'nan':>8} "
            f"{auroc if auroc is not None else 'nan':>8} "
            f"{f1 if f1 is not None else 'nan':>8}"
        )
    if macro is not None:
        print(f"\nmacro_online_hour_bucket_auprc (equal weight per bucket, min_n applied in summary): {macro:.4f}")


def write_csv(table: List[Dict[str, Any]], path: Path) -> None:
    if not table:
        return
    fieldnames = list(table[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(table)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, train_args, _ = run_online_evaluation(
        checkpoint=args.checkpoint,
        episode_path=args.episode_path,
        device_str=args.device,
        batch_size=args.batch_size,
        day_bin_width=args.day_bin_width,
        postop_hour_bin=args.postop_hour_bin,
        cls_threshold=args.cls_threshold,
    )

    report = enrich_metrics(
        metrics,
        train_args,
        args.checkpoint,
        args.episode_path,
        args.postop_hour_bin,
        args.macro_min_n,
        args.selection_lambda,
    )

    json_path = out_dir / "metrics.json"
    save_json(report, json_path)

    if args.export_csv:
        write_csv(report["hour_bucket_table"], out_dir / "hour_bucket_metrics.csv")

    if args.print_table:
        print_hour_bucket_table(
            report["hour_bucket_table"],
            report["summary"].get("macro_online_hour_bucket_auprc"),
        )

    s = report["summary"]
    print(f"Wrote {json_path}")
    print(
        f"online_all auprc={s.get('online_all_auprc')} | "
        f"macro_6h_auprc={s.get('macro_online_hour_bucket_auprc')} | "
        f"baseline_auprc={s.get('baseline_auprc')} | "
        f"sel_score(λ={report['selection_lambda']})={s.get('validation_combined_selection_score')}"
    )


if __name__ == "__main__":
    main()
