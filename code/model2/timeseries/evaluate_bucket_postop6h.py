#!/usr/bin/env python3
"""Evaluate 6h bucket model by postoperative hour bucket (default 6h bins)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from timeseries.checkpoint import infer_arch, load_bucket_checkpoint
from timeseries.dataset import Ts6hEpisodeDataset
from common.fold_data import default_test_jsonl
from common.utils import macro_mean_postop_hour_bucket_auprc, resolve_torch_device, save_json
from timeseries.utils_eval import evaluate_bucket_online


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bucket ts6h postoperative hour-bucket eval on test.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_npz", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--episode_path", type=str, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--postop_hour_bin", type=float, default=6.0)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--macro_min_n", type=int, default=10)
    p.add_argument("--export_csv", action="store_true")
    p.add_argument("--print_table", action="store_true")
    return p.parse_args()


def build_hour_bucket_table(metrics: Dict[str, Any], bin_width: float) -> List[Dict[str, Any]]:
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
                "hour_bucket": b,
                "hour_range": f"[{h0:g},{h1:g})",
                "n": stats.get("n"),
                "auroc": stats.get("auroc"),
                "auprc": stats.get("auprc"),
                "f1": stats.get("f1"),
                "sensitivity": stats.get("sensitivity"),
                "specificity": stats.get("specificity"),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    model, _ckpt, train_args, layout = load_bucket_checkpoint(args.checkpoint, device)
    if layout and hasattr(model, "set_layout"):
        model.set_layout(layout)
    arch = infer_arch(train_args)
    bin_width = float(args.postop_hour_bin or train_args.get("postop_val_hour_bin", 6.0))

    test_ds = Ts6hEpisodeDataset(args.test_npz, load_obs=bool(getattr(model, "uses_obs_mask", False)))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
    )
    metrics = evaluate_bucket_online(
        model,
        loader,
        device,
        cls_threshold=args.cls_threshold,
        postop_val_hour_bin_width=bin_width,
    )
    macro_auprc = macro_mean_postop_hour_bucket_auprc(
        metrics.get("by_postop_hour_bucket") or {},
        min_n=args.macro_min_n,
    )

    default_jsonl = default_test_jsonl()
    report: Dict[str, Any] = {
        "evaluation": "postoperative_hour_bucket_online_aki_risk",
        "prediction_unit": "online logits at each valid postoperative 6h bucket",
        "model_type": f"bucket_{arch}_ts6h",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path or default_jsonl).resolve()),
        "test_npz": str(Path(args.test_npz).resolve()),
        "postop_hour_bin": bin_width,
        "cls_threshold": float(args.cls_threshold),
        "macro_online_hour_bucket_auprc": macro_auprc,
        "metrics": {
            **dict(metrics.get("online_all_valid") or {}),
            "macro_online_hour_bucket_auprc": macro_auprc,
        },
        "full_metrics": metrics,
    }
    save_json(report, out_dir / "postop_hour_bucket_metrics.json")

    if args.export_csv:
        rows = build_hour_bucket_table(metrics, bin_width)
        csv_path = out_dir / "hour_bucket_metrics.csv"
        if rows:
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {csv_path}")

    if args.print_table:
        print(f"\n=== Bucket {arch} postop hour bins (width={bin_width}h, τ={args.cls_threshold}) ===")
        print(f"  macro AUPRC (min_n={args.macro_min_n}): {macro_auprc}")
        for row in build_hour_bucket_table(metrics, bin_width):
            print(
                f"  bin {row['hour_bucket']:>3} {row['hour_range']:>12}  "
                f"n={row['n']}  AUROC={row['auroc']}  AUPRC={row['auprc']}  F1={row['f1']}"
            )
    print(f"Wrote {out_dir / 'postop_hour_bucket_metrics.json'}")


if __name__ == "__main__":
    main()
