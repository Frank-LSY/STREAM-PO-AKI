#!/usr/bin/env python3
"""Clinical utility evaluation for model2 online AKI risk.

Metrics: early warning before AKI onset, false-alert burden, PPV under alert budgets,
baseline vs online discrimination / NRI / decision curve, calibration.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.clinical_metrics import (
    EpisodeTrajectory,
    evaluate_clinical,
    fit_temperature_scaling,
    plot_calibration_figures,
    trajectory_from_episode_and_forward,
    _pool_online_from_trajectories,
)
from event_stream.eval_common import build_eval_dataloader, build_model_from_checkpoint, load_checkpoint_bundle
from common.utils import _forward_model, resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clinical evaluation for model2 AKI online risk.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--alert_thresholds",
        type=str,
        default="0.5,0.7,0.9",
        help="Comma-separated tau values for alert rules.",
    )
    p.add_argument(
        "--plot_dca",
        action="store_true",
        help="Save decision-curve plot (baseline vs online max).",
    )
    p.add_argument(
        "--plot_calibration",
        action="store_true",
        help="Save calibration reliability diagrams (overview, hour buckets, ECE curve).",
    )
    p.add_argument(
        "--postop_hour_bin",
        type=float,
        default=None,
        help="Hour bucket width for calibration strata (default: checkpoint postop_val_hour_bin or 6).",
    )
    p.add_argument("--calibration_n_bins", type=int, default=10)
    p.add_argument(
        "--calibration_min_bucket_n",
        type=int,
        default=200,
        help="Min postop tokens per hour bucket to include in bucket calibration plots.",
    )
    p.add_argument(
        "--calib_jsonl",
        type=str,
        default=None,
        help="Optional JSONL to fit temperature scaling (e.g. validation fold). If omitted, T=1.",
    )
    p.add_argument(
        "--fit_temperature_on_eval",
        action="store_true",
        help="Fit T on episode_path itself (not recommended; for debugging only).",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def collect_episode_trajectories(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    episodes: List[Dict[str, Any]],
) -> List[EpisodeTrajectory]:
    from event_stream.input_builder import episode_to_sequence_features

    model.eval()
    trajectories: List[EpisodeTrajectory] = []
    ep_idx = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = _forward_model(model, batch)
            token_probs = torch.sigmoid(outputs["token_logits"]).cpu().numpy()
            base_logits = outputs["baseline_logits_all"]
            B = token_probs.shape[0]
            for i in range(B):
                if ep_idx >= len(episodes):
                    break
                episode = episodes[ep_idx]
                feat = episode_to_sequence_features(
                    episode,
                    loader.dataset.code_vocab,  # type: ignore[attr-defined]
                    add_base=True,
                    lab_stats=getattr(loader.dataset, "lab_stats", None),  # type: ignore
                    num_med_exposure=getattr(loader.dataset, "num_med_exposure", 0),  # type: ignore
                )
                seq = feat["sequence"]
                valid_len = int(batch["attention_mask"][i].sum().item())
                probs_i = token_probs[i, :valid_len]
                base_mask = batch["base_mask"][i, :valid_len].cpu().numpy() > 0
                if base_mask.any():
                    bp = float(torch.sigmoid(base_logits[i, :valid_len][base_mask]).max().item())
                else:
                    bp = float(torch.sigmoid(base_logits[i, :valid_len]).max().item())
                trajectories.append(
                    trajectory_from_episode_and_forward(episode, probs_i, bp, seq)
                )
                ep_idx += 1
    return trajectories


def maybe_plot_dca(report: Dict[str, Any], out_dir: Path, dpi: int) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    dca = report.get("baseline_vs_online", {}).get("decision_curve", {})
    if not dca:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, style in (
        ("baseline_only", "C0--"),
        ("baseline_max_online", "C1-"),
    ):
        block = dca.get(name, {})
        th = block.get("thresholds", [])
        nb = block.get("net_benefit", [])
        if th and nb:
            ax.plot(th, nb, style, label=name.replace("_", " "))
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve (episode-level risk)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "decision_curve.png", dpi=dpi)
    plt.close(fig)


def print_clinical_summary(report: Dict[str, Any], primary_tau: float = 0.5) -> None:
    key = f"tau_{primary_tau:g}"
    block = report.get("by_alert_threshold", {}).get(key, {})
    ew = block.get("early_warning", {})
    fa = block.get("false_alert_burden", {})
    print("=== Early warning (AKI+ episodes) ===")
    print(f"  alert sensitivity before AKI (tau={primary_tau}): {ew.get('alert_sensitivity_before_aki')}")
    print(f"  median lead time (h): {ew.get('median_lead_time_hours')}")
    print(f"  IQR lead time (h): {ew.get('lead_time_iqr_hours')}")
    print(f"  proportion lead >= 24h: {ew.get('proportion_lead_ge_24h')}")
    print("=== False alert burden ===")
    print(f"  false alert episode rate (non-AKI): {fa.get('false_alert_episode_rate')}")
    print(f"  false alerts per 100 patient-days: {fa.get('false_alerts_per_100_patient_days')}")
    disc = report.get("baseline_vs_online", {}).get("discrimination", {})
    print("=== Discrimination (episode-level) ===")
    print(f"  baseline AUPRC: {disc.get('baseline_auprc')} | online max AUPRC: {disc.get('online_max_auprc')}")


def _resolve_postop_hour_bin(args: argparse.Namespace, train_args: Dict[str, Any]) -> float:
    if args.postop_hour_bin is not None and args.postop_hour_bin > 0:
        return float(args.postop_hour_bin)
    w = float(train_args.get("postop_val_hour_bin", 6.0))
    return w if w > 0 else 6.0


def _fit_temperature_from_trajectories(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    episodes: List[Dict[str, Any]],
) -> float:
    tr = collect_episode_trajectories(model, loader, device, episodes)
    _, probs, labels = _pool_online_from_trajectories(tr)
    base_probs = np.array([t.baseline_prob for t in tr], dtype=np.float64)
    base_y = np.array([t.label_aki for t in tr], dtype=np.int64)
    y = np.concatenate([labels, base_y])
    p = np.concatenate([probs, base_probs])
    return fit_temperature_scaling(y, p)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [float(x.strip()) for x in args.alert_thresholds.split(",") if x.strip()]

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    train_args = ckpt.get("args", {}) or {}
    postop_hour_bin = _resolve_postop_hour_bin(args, train_args)
    from event_stream.dataset import load_episode_jsonl

    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        from common.med_exposure import infer_num_med_exposure

        num_med = infer_num_med_exposure(episodes)
    loader = build_eval_dataloader(args.episode_path, code_vocab, lab_stats, num_med, args.batch_size, age_stats=age_stats)
    model = build_model_from_checkpoint(code_vocab, train_args, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device)

    temperature = 1.0
    temperature_fit_on: Optional[str] = None
    if args.calib_jsonl:
        calib_eps = load_episode_jsonl(args.calib_jsonl)
        if num_med < 0:
            from common.med_exposure import infer_num_med_exposure

            num_med = infer_num_med_exposure(calib_eps)
        calib_loader = build_eval_dataloader(args.calib_jsonl, code_vocab, lab_stats, num_med, args.batch_size, age_stats=age_stats)
        temperature = _fit_temperature_from_trajectories(model, calib_loader, device, calib_eps)
        temperature_fit_on = str(Path(args.calib_jsonl).resolve())
        print(f"Fitted temperature T={temperature:.4f} on {temperature_fit_on}")
    elif args.fit_temperature_on_eval:
        temperature = _fit_temperature_from_trajectories(model, loader, device, episodes)
        temperature_fit_on = str(Path(args.episode_path).resolve())
        print(f"Warning: fitted T={temperature:.4f} on eval set (same as episode_path).")

    trajectories = collect_episode_trajectories(model, loader, device, episodes)
    report = evaluate_clinical(
        trajectories,
        alert_thresholds=thresholds,
        calibration_n_bins=args.calibration_n_bins,
        postop_hour_bin_width=postop_hour_bin,
        calibration_min_bucket_n=args.calibration_min_bucket_n,
        temperature=temperature,
    )
    report["checkpoint"] = str(Path(args.checkpoint).resolve())
    report["episode_path"] = str(Path(args.episode_path).resolve())
    report["alert_thresholds"] = thresholds
    if temperature_fit_on:
        report["temperature_fit_jsonl"] = temperature_fit_on

    save_json(report, out_dir / "clinical_metrics.json")
    if args.plot_dca:
        maybe_plot_dca(report, out_dir, args.dpi)
    if args.plot_calibration:
        paths = plot_calibration_figures(report.get("calibration", {}), out_dir, dpi=args.dpi)
        for p in paths:
            print(f"Wrote {p}")

    print_clinical_summary(report, primary_tau=thresholds[0] if thresholds else 0.5)
    cal = report.get("calibration", {})
    pooled = (cal.get("online_all_postop_steps") or {}).get("uncalibrated") or {}
    print(
        f"=== Calibration (online all, uncalibrated) ===\n"
        f"  ECE={pooled.get('ece')} Brier={pooled.get('brier')} n={pooled.get('n')}"
    )
    print(f"Wrote {out_dir / 'clinical_metrics.json'}")


if __name__ == "__main__":
    main()
