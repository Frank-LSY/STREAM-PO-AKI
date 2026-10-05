#!/usr/bin/env python3
"""Fixed 6h postoperative time-grid test (fair comparison with bucket-aligned baselines)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.fixed_time_grid import (
    build_bucket_table,
    evaluate_fixed_time_grid,
    evaluate_fixed_time_grid_fast,
    plot_bucket_metric_curves,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model2 fixed postoperative time-grid evaluation.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device: cuda:N (e.g. cuda:0), cuda, cpu, or auto.",
    )
    p.add_argument("--bucket_size", type=float, default=6.0, help="Grid width in hours (default 6).")
    p.add_argument("--bucket_start", type=float, default=0.0, help="First bucket left edge (postop hours).")
    p.add_argument("--bucket_end", type=float, default=720.0, help="Grid horizon (default 30 days).")
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--macro_min_n", type=int, default=10, help="Min rows per bucket for macro AUPRC.")
    p.add_argument("--max_episodes", type=int, default=0, help="Cap episodes (0 = all).")
    p.add_argument(
        "--slow",
        action="store_true",
        help="Re-forward the truncated episode at every grid point. Exactly equivalent to the "
        "default batched path but ~2 orders of magnitude slower; kept for verification.",
    )
    p.add_argument("--batch_size", type=int, default=16, help="Episodes per forward (fast path).")
    p.add_argument("--no_progress", action="store_true")
    p.add_argument("--export_csv", action="store_true", help="Write bucket_metrics.csv.")
    p.add_argument("--print_table", action="store_true")
    p.add_argument(
        "--plot_metrics",
        action="store_true",
        help="Save line plots for each metric vs postoperative hour bucket.",
    )
    p.add_argument(
        "--no_carry_forward",
        action="store_true",
        help="Disable carry-forward: buckets without postop prediction stay NaN (legacy behavior).",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def print_bucket_table(table: List[Dict[str, Any]], macro: Any) -> None:
    if not table:
        print("No buckets.")
        return
    header = f"{'bucket':>6} {'hours':>14} {'n':>8} {'n+':>7} {'n-':>7} {'auprc':>8} {'auroc':>8}"
    print(header)
    print("-" * len(header))
    for r in table:
        print(
            f"{r['bucket_hour']:>6} {r['hour_range']:>14} {r['n']:>8} "
            f"{r.get('n_pos', ''):>7} {r.get('n_neg', ''):>7} "
            f"{r['auprc'] if r['auprc'] is not None else 'nan':>8} "
            f"{r['auroc'] if r['auroc'] is not None else 'nan':>8}"
        )
    if macro is not None:
        print(f"\nmacro_bucket_auprc: {macro:.4f}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    train_args = ckpt.get("args", {}) or {}
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)

    model = build_model_from_checkpoint(code_vocab, train_args, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    print(f"Using device: {device} (requested: {args.device})")
    model.to(device)
    model.eval()

    common_kwargs = dict(
        bucket_size=args.bucket_size,
        bucket_start=args.bucket_start,
        bucket_end=args.bucket_end,
        cls_threshold=args.cls_threshold,
        macro_min_n=args.macro_min_n,
        max_episodes=args.max_episodes,
        progress=not args.no_progress,
        carry_forward=not args.no_carry_forward,
        # Same featurization as training / evaluate_postop_hour_buckets (age z-score + input ablations).
        feature_kwargs=dict(
            age_stats=age_stats,
            baseline_token=str(train_args.get("baseline_token", "dedicated")),
            drop_conditions=bool(train_args.get("drop_conditions", False)),
            drop_lab_values=bool(train_args.get("drop_lab_values", False)),
            drop_labs=bool(train_args.get("drop_labs", False)),
            drop_meds=bool(train_args.get("drop_meds", False)),
            generic_surgery=bool(train_args.get("generic_surgery", False)),
        ),
    )
    runner = evaluate_fixed_time_grid if args.slow else evaluate_fixed_time_grid_fast
    if not args.slow:
        common_kwargs["batch_size"] = args.batch_size
    report = runner(
        episodes,
        model,
        code_vocab,
        lab_stats,
        num_med,
        device,
        **common_kwargs,
    )
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["episode_path"] = str(Path(args.episode_path).resolve())

    table = build_bucket_table(report.get("by_bucket_hour", {}), args.bucket_size)
    report["bucket_table"] = table

    save_json(report, out_dir / "fixed_time_grid_metrics.json")

    if args.export_csv and table:
        csv_path = out_dir / "fixed_time_grid_bucket_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print(f"Wrote {csv_path}")

    if args.print_table:
        print_bucket_table(table, report.get("summary", {}).get("macro_bucket_auprc"))

    if args.plot_metrics and table:
        plot_paths = plot_bucket_metric_curves(
            table, out_dir, dpi=args.dpi, bucket_size=args.bucket_size
        )
        report["plot_paths"] = plot_paths
        save_json(report, out_dir / "fixed_time_grid_metrics.json")
        for p in plot_paths:
            print(f"Wrote {p}")

    s = report.get("summary", {})
    print(f"Wrote {out_dir / 'fixed_time_grid_metrics.json'}")
    print(
        f"episodes={report.get('n_episodes')} grid_rows={report.get('n_grid_rows')} "
        f"finite={report.get('n_grid_rows_finite_prob')} | "
        f"pooled_auprc={s.get('pooled_grid_auprc')} macro_bucket_auprc={s.get('macro_bucket_auprc')} | "
        f"episode_last_auprc={s.get('episode_last_grid_auprc')}"
    )


if __name__ == "__main__":
    main()
