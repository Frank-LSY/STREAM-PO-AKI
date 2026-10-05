#!/usr/bin/env python3
"""Fixed 6h grid metrics from direct bucket predictions (no event truncation)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from timeseries.checkpoint import infer_arch, load_bucket_checkpoint
from timeseries.dataset import Ts6hEpisodeDataset
from timeseries.timeseries_6h_common import BUCKET_SIZE_H
from common.utils import _subset_classification_metrics, resolve_torch_device, save_json
from timeseries.utils_eval import collect_bucket_predictions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bucket grid eval on test NPZ.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_npz", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--bucket_size", type=float, default=6.0)
    p.add_argument("--macro_min_n", type=int, default=10)
    p.add_argument("--export_csv", action="store_true")
    return p.parse_args()


def macro_bucket_auprc(by_bucket: Dict[str, Dict[str, Any]], min_n: int) -> float | None:
    vals = []
    for b in sorted(by_bucket.keys(), key=lambda x: int(x)):
        m = by_bucket[b]
        if int(m.get("n", 0)) < min_n:
            continue
        a = m.get("auprc")
        if a is not None:
            vals.append(float(a))
    return float(np.mean(vals)) if vals else None


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    model, _ckpt, train_args, _layout = load_bucket_checkpoint(args.checkpoint, device)
    arch = infer_arch(train_args)

    test_ds = Ts6hEpisodeDataset(args.test_npz, load_obs=bool(getattr(model, "uses_obs_mask", False)))
    loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
    )
    arrs = collect_bucket_predictions(model, loader, device)
    probs, targets, hours = arrs["probs"], arrs["targets"], arrs["bucket_hours"]

    by_bucket: Dict[str, Dict[str, Any]] = {}
    w = float(args.bucket_size)
    if len(hours) > 0:
        buckets = np.floor(hours / w).astype(int)
        for b in sorted(np.unique(buckets)):
            mask = buckets == b
            by_bucket[str(int(b))] = _subset_classification_metrics(
                targets[mask], probs[mask], args.cls_threshold
            )

    pooled = _subset_classification_metrics(targets, probs, args.cls_threshold)
    macro = macro_bucket_auprc(by_bucket, args.macro_min_n)

    report = {
        "evaluation": "fixed_6h_bucket_grid_direct",
        "model_type": f"bucket_{arch}_ts6h",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "test_npz": str(Path(args.test_npz).resolve()),
        "bucket_size_hours": w,
        "by_bucket_hour": by_bucket,
        "summary": {
            "pooled_grid_auprc": pooled.get("auprc"),
            "pooled_grid_auroc": pooled.get("auroc"),
            "macro_bucket_auprc": macro,
            "n_predictions": int(len(probs)),
        },
        "baseline_node": _subset_classification_metrics(
            arrs["base_targets"], arrs["base_probs"], args.cls_threshold
        ),
    }
    save_json(report, out_dir / "bucket_grid_metrics.json")

    if args.export_csv:
        rows: List[Dict[str, Any]] = []
        for b, m in sorted(by_bucket.items(), key=lambda x: int(x[0])):
            bh = int(b) * int(w)
            rows.append(
                {
                    "bucket_hour": bh,
                    "hour_range": f"[{bh},{bh+int(w)})",
                    "n": m.get("n"),
                    "auroc": m.get("auroc"),
                    "auprc": m.get("auprc"),
                }
            )
        csv_path = out_dir / "bucket_grid_metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["bucket_hour"])
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"Wrote {csv_path}")

    print(
        f"pooled_auprc={report['summary']['pooled_grid_auprc']} "
        f"macro_bucket_auprc={macro} baseline_auprc={report['baseline_node'].get('auprc')}"
    )


if __name__ == "__main__":
    main()
