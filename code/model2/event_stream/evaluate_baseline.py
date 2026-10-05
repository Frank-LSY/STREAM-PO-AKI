#!/usr/bin/env python3
"""Evaluate preoperative baseline AKI risk at the [BASE] token (one prediction per episode)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import load_checkpoint_bundle, run_online_evaluation
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate baseline ([BASE]) AKI risk prediction.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--cls_threshold", type=float, default=0.5, help="Threshold τ for classification metrics.")
    p.add_argument("--print_table", action="store_true", help="Print metrics table to stdout.")
    p.add_argument(
        "--plot_calibration",
        action="store_true",
        help="Save baseline calibration curve and baseline_calibration.json.",
    )
    p.add_argument(
        "--calib_jsonl",
        type=str,
        default=None,
        help="Validation JSONL for temperature scaling (e.g. same-fold train/val).",
    )
    p.add_argument("--calibration_n_bins", type=int, default=10)
    p.add_argument(
        "--plot_temperature_scaled",
        action="store_true",
        help="Overlay temperature-scaled curve on calibration plot when T != 1.",
    )
    p.add_argument("--export_calibration_csv", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def print_metrics_table(m: Dict[str, Any], tau: float) -> None:
    cm = m.get("confusion_matrix") or {}
    print(f"\n=== Baseline AKI risk ([BASE] token, episode-level, τ={tau}) ===")
    print(f"  n={m.get('n')}  n_pos={m.get('n_pos')}  n_neg={m.get('n_neg')}  pos_rate={m.get('pos_rate')}")
    print(f"  AUROC={m.get('auroc')}  AUPRC={m.get('auprc')}")
    print(
        f"  accuracy={m.get('accuracy')}  precision={m.get('precision')}  "
        f"sensitivity={m.get('sensitivity')}  recall={m.get('recall')}  "
        f"specificity={m.get('specificity')}  f1={m.get('f1')}"
    )
    print(
        f"  confusion (tn, fp, fn, tp)=({cm.get('tn')}, {cm.get('fp')}, {cm.get('fn')}, {cm.get('tp')})"
    )
    if m.get("confusion_matrix_2x2") is not None:
        print(f"  confusion_matrix_2x2 (rows=true 0,1): {m.get('confusion_matrix_2x2')}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics, train_args, model = run_online_evaluation(
        checkpoint=args.checkpoint,
        episode_path=args.episode_path,
        device_str=args.device,
        batch_size=args.batch_size,
        cls_threshold=args.cls_threshold,
        postop_hour_bin=6.0,
    )
    baseline = dict(metrics.get("baseline_node") or {})

    report: Dict[str, Any] = {
        "evaluation": "baseline_preoperative_aki_risk",
        "prediction_unit": "one [BASE] token per episode (episode-level AKI label)",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path).resolve()),
        "cls_threshold": float(args.cls_threshold),
        "train_args_snapshot": {
            "best_selection_lambda": train_args.get("best_selection_lambda"),
            "postop_val_hour_bin": train_args.get("postop_val_hour_bin"),
        },
        "metrics": baseline,
    }

    json_path = out_dir / "baseline_metrics.json"
    save_json(report, json_path)

    if args.print_table:
        print_metrics_table(baseline, args.cls_threshold)

    print(f"Wrote {json_path}")

    if args.plot_calibration:
        from event_stream.baseline_calibration import run_baseline_calibration

        _, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
        if num_med < 0:
            num_med = infer_num_med_exposure(load_episode_jsonl(args.episode_path))
        device = resolve_torch_device(args.device)
        run_baseline_calibration(
            model,
            args.episode_path,
            out_dir,
            code_vocab,
            lab_stats,
            num_med,
            args.batch_size,
            device,
            age_stats=age_stats,
            checkpoint=args.checkpoint,
            calib_jsonl=args.calib_jsonl,
            n_bins=args.calibration_n_bins,
            plot_temperature_scaled=args.plot_temperature_scaled,
            export_csv=args.export_calibration_csv,
            dpi=args.dpi,
        )


if __name__ == "__main__":
    main()
