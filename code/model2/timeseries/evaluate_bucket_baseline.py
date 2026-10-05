#!/usr/bin/env python3
"""Evaluate 6h bucket Transformer baseline ([BASE] bucket) on test NPZ."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from timeseries.checkpoint import infer_arch, load_bucket_checkpoint
from timeseries.dataset import Ts6hEpisodeDataset
from common.fold_data import default_test_jsonl
from common.utils import resolve_torch_device, save_json
from timeseries.utils_eval import evaluate_bucket_online


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bucket Transformer baseline eval on test.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--test_npz", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--episode_path", type=str, default=None, help="Test JSONL for report parity.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--print_table", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_torch_device(args.device)
    model, ckpt, train_args, layout = load_bucket_checkpoint(args.checkpoint, device)
    if layout and hasattr(model, "set_layout"):
        model.set_layout(layout)
    arch = infer_arch(train_args)
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
        postop_val_hour_bin_width=float(train_args.get("postop_val_hour_bin", 6.0)),
    )
    baseline = dict(metrics.get("baseline_node") or {})

    default_jsonl = default_test_jsonl()
    report: Dict[str, Any] = {
        "evaluation": "baseline_preoperative_aki_risk",
        "prediction_unit": "one [BASE] 6h bucket per episode (episode-level AKI label)",
        "model_type": f"bucket_{arch}_ts6h",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path or default_jsonl).resolve()),
        "test_npz": str(Path(args.test_npz).resolve()),
        "cls_threshold": float(args.cls_threshold),
        "metrics": baseline,
    }
    save_json(report, out_dir / "baseline_metrics.json")

    if args.print_table:
        cm = baseline.get("confusion_matrix") or {}
        print(f"\n=== Bucket {arch} baseline (τ={args.cls_threshold}) ===")
        print(f"  AUROC={baseline.get('auroc')}  AUPRC={baseline.get('auprc')}  F1={baseline.get('f1')}")
        print(f"  sensitivity={baseline.get('sensitivity')}  specificity={baseline.get('specificity')}")
        print(f"  confusion (tn, fp, fn, tp)=({cm.get('tn')}, {cm.get('fp')}, {cm.get('fn')}, {cm.get('tp')})")
    print(f"Wrote {out_dir / 'baseline_metrics.json'}")


if __name__ == "__main__":
    main()
