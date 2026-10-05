#!/usr/bin/env python3
"""Cohort-level event contribution to predicted AKI probability (occlusion Δp).

Pools attributions by (modality, raw_code) without time-window stratification.
Not causal — model attribution under occlusion masking.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_attribution import (
    aggregate_time_agnostic_probability_contributions,
    probability_contributions_for_episode,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Event contribution to risk probability (occlusion Δp), time-agnostic cohort ranking."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_episodes", type=int, default=80, help="Cap episodes (0 = all).")
    p.add_argument("--max_preop_occlusions", type=int, default=60)
    p.add_argument("--max_postop_query_steps", type=int, default=25)
    p.add_argument("--max_events_per_query", type=int, default=60)
    p.add_argument("--lookback_hours", type=float, default=72.0)
    p.add_argument("--min_count", type=int, default=10, help="Min pooled measurements per event type.")
    p.add_argument("--top_k", type=int, default=50, help="Max events per direction in full ranking.")
    p.add_argument(
        "--top_k_report",
        type=int,
        default=5,
        help="Top-K positive/negative events written to dedicated CSV and summary fields.",
    )
    p.add_argument("--plot_top", action="store_true", help="Bar charts for top Δp events.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--save_per_episode_records", action="store_true", help="Write all flat records (large).")
    return p.parse_args()


def _write_ranking_csv(path: Path, rows: Sequence[Dict[str, Any]], direction: str) -> None:
    if not rows:
        return
    fields = [
        "modality",
        "raw_code",
        "n_measurements",
        "mean_delta_prob",
        "median_delta_prob",
        "mean_delta_logit",
        "fraction_positive",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  Wrote {path} ({direction}, n={len(rows)})")


def plot_top_events(
    agg: Dict[str, Any],
    out_path: Path,
    title: str,
    top_n: int = 20,
    dpi: int = 150,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    pos = agg.get("risk_increasing_events", [])[:top_n]
    neg = agg.get("risk_decreasing_events", [])[:top_n]
    if not pos and not neg:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4, 0.35 * max(len(pos), len(neg), 1))))

    def _bar(ax, rows, color, xlabel):
        if not rows:
            ax.axis("off")
            return
        labels = [f"{r['modality'][:3]}:{r['raw_code'][:28]}" for r in rows][::-1]
        vals = [r["mean_delta_prob"] for r in rows][::-1]
        ax.barh(range(len(labels)), vals, color=color, alpha=0.85)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel(xlabel)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.grid(True, axis="x", alpha=0.25)

    _bar(axes[0], pos, "C3", "Mean Δ probability (risk-increasing)")
    _bar(axes[1], neg, "C0", "Mean Δ probability (risk-decreasing)")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def build_report(
    all_records: List[Dict[str, Any]],
    checkpoint: str,
    episode_path: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    cohort_all = aggregate_time_agnostic_probability_contributions(
        all_records,
        min_count=settings["min_count"],
        top_k=settings["top_k"],
        label_aki=None,
    )
    cohort_aki_pos = aggregate_time_agnostic_probability_contributions(
        all_records,
        min_count=settings["min_count"],
        top_k=settings["top_k"],
        label_aki=1,
    )
    cohort_aki_neg = aggregate_time_agnostic_probability_contributions(
        all_records,
        min_count=settings["min_count"],
        top_k=settings["top_k"],
        label_aki=0,
    )

    n_ep = len({r["person_id"] for r in all_records})
    n_aki = len({r["person_id"] for r in all_records if int(r.get("label_aki", 0)) == 1})

    by_modality: Dict[str, List[float]] = {}
    for r in all_records:
        by_modality.setdefault(str(r.get("modality", "")), []).append(float(r["delta_prob"]))

    return {
        "analysis_type": "event_probability_contribution_time_agnostic",
        "method": "occlusion_masking_probability_difference",
        "disclaimer": (
            "Each measurement is Δp = p(with event) − p(event masked) at a baseline or online "
            "query step. Positive mean Δp means the event is associated with higher model risk "
            "when present. This is model attribution, not a causal effect. Time is not used "
            "in cohort aggregation (all query times and episodes pooled per event code)."
        ),
        "definitions": {
            "delta_prob": "sigmoid(z) − sigmoid(z_masked); positive → event increases predicted risk",
            "baseline_scope": "Effect on baseline token probability (preoperative events)",
            "online_scope": "Effect on online token probability at subsampled postop query steps",
            "grouping": "(modality, raw_code) pooled across episodes and query times",
        },
        "checkpoint": str(Path(checkpoint).resolve()),
        "episode_path": str(Path(episode_path).resolve()),
        "settings": settings,
        "n_episodes": n_ep,
        "n_aki_positive_episodes": n_aki,
        "n_occlusion_measurements": len(all_records),
        "mean_delta_prob_by_modality": {
            m: float(sum(v) / len(v)) if v else None for m, v in sorted(by_modality.items())
        },
        "cohort_all": {
            "risk_increasing_events": cohort_all["risk_increasing_events"],
            "risk_decreasing_events": cohort_all["risk_decreasing_events"],
            "all_ranked_events": cohort_all["all_ranked_events"],
            "n_event_types_ranked": cohort_all["n_event_types_ranked"],
        },
        "top_positive_contributors": cohort_all["risk_increasing_events"][: settings["top_k_report"]],
        "top_negative_contributors": cohort_all["risk_decreasing_events"][: settings["top_k_report"]],
        "cohort_aki_positive": {
            "risk_increasing_events": cohort_aki_pos["risk_increasing_events"],
            "risk_decreasing_events": cohort_aki_pos["risk_decreasing_events"],
            "n_event_types_ranked": cohort_aki_pos["n_event_types_ranked"],
        },
        "cohort_aki_negative": {
            "risk_increasing_events": cohort_aki_neg["risk_increasing_events"],
            "risk_decreasing_events": cohort_aki_neg["risk_decreasing_events"],
            "n_event_types_ranked": cohort_aki_neg["n_event_types_ranked"],
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    train_args = ckpt.get("args", {}) or {}
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]

    model = build_model_from_checkpoint(code_vocab, train_args, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()

    all_records: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes):
        print(f"Δp contribution {i + 1}/{len(episodes)} person_id={ep.get('person_id')}")
        recs = probability_contributions_for_episode(
            model,
            ep,
            code_vocab,
            lab_stats,
            num_med,
            device,
            max_preop_occlusions=args.max_preop_occlusions,
            max_postop_query_steps=args.max_postop_query_steps,
            max_events_per_query=args.max_events_per_query,
            lookback_hours=args.lookback_hours,
        )
        all_records.extend(recs)

    settings = {
        "max_episodes": args.max_episodes,
        "max_preop_occlusions": args.max_preop_occlusions,
        "max_postop_query_steps": args.max_postop_query_steps,
        "max_events_per_query": args.max_events_per_query,
        "lookback_hours": args.lookback_hours,
        "min_count": args.min_count,
        "top_k": args.top_k,
        "top_k_report": args.top_k_report,
    }
    report = build_report(all_records, args.checkpoint, args.episode_path, settings)
    save_json(report, out_dir / "event_probability_contribution_summary.json")

    cohort = report["cohort_all"]
    top_pos = report["top_positive_contributors"]
    top_neg = report["top_negative_contributors"]
    _write_ranking_csv(
        out_dir / "top_risk_increasing_events.csv",
        cohort["risk_increasing_events"],
        "risk_increasing",
    )
    _write_ranking_csv(
        out_dir / "top_risk_decreasing_events.csv",
        cohort["risk_decreasing_events"],
        "risk_decreasing",
    )
    _write_ranking_csv(
        out_dir / f"top{args.top_k_report}_risk_increasing_events.csv",
        top_pos,
        f"top{args.top_k_report}_risk_increasing",
    )
    _write_ranking_csv(
        out_dir / f"top{args.top_k_report}_risk_decreasing_events.csv",
        top_neg,
        f"top{args.top_k_report}_risk_decreasing",
    )

    if args.save_per_episode_records:
        save_json(all_records, out_dir / "event_probability_contribution_records.json")

    if args.plot_top:
        plot_top_events(
            report["cohort_all"],
            out_dir / "top_event_probability_contributions.png",
            title="Time-agnostic mean Δp by event (all episodes)",
            top_n=min(args.top_k_report, args.top_k),
            dpi=args.dpi,
        )

    print("=== Event → AKI risk contribution (time-agnostic, occlusion Δp) ===")
    print(f"  episodes: {report['n_episodes']} (AKI+: {report['n_aki_positive_episodes']})")
    print(f"  occlusion measurements: {report['n_occlusion_measurements']}")
    print(f"  event types ranked (min_count={args.min_count}): {cohort['n_event_types_ranked']}")
    print(f"  top {args.top_k_report} positive contributors (mean Δp):")
    for i, row in enumerate(top_pos, 1):
        print(
            f"    {i}. [{row['modality']}] {row['raw_code']}  "
            f"Δp={row['mean_delta_prob']:.6f}  n={row['n_measurements']}"
        )
    print(f"  top {args.top_k_report} negative contributors (mean Δp):")
    for i, row in enumerate(top_neg, 1):
        print(
            f"    {i}. [{row['modality']}] {row['raw_code']}  "
            f"Δp={row['mean_delta_prob']:.6f}  n={row['n_measurements']}"
        )
    print(f"Wrote {out_dir / 'event_probability_contribution_summary.json'}")


if __name__ == "__main__":
    main()
