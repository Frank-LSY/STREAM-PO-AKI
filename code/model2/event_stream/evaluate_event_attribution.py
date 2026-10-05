#!/usr/bin/env python3
"""Secondary analysis: model-attributed event contribution (occlusion on logits).

Not causal inference — measures how masking events changes model risk logits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_attribution import (
    aggregate_cohort_attributions,
    attributions_for_episode,
    episode_to_batch,
    forward_logits,
    select_case_study_episodes,
    top_contributors,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Model-attributed event contribution via occlusion (logit difference)."
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_episodes", type=int, default=50, help="Cap episodes for runtime.")
    p.add_argument("--max_preop_occlusions", type=int, default=80)
    p.add_argument("--max_postop_query_steps", type=int, default=40)
    p.add_argument("--lookback_hours", type=float, default=72.0)
    p.add_argument("--alert_threshold", type=float, default=0.5, help="For case-study alert marker.")
    p.add_argument("--plot_case_studies", action="store_true")
    p.add_argument("--n_case_aki", type=int, default=2)
    p.add_argument("--n_case_non", type=int, default=2)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def build_report(
    episodes_attr: List[Dict[str, Any]],
    checkpoint: str,
    episode_path: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    cohort = aggregate_cohort_attributions(episodes_attr)
    return {
        "analysis_type": "model_attributed_event_contribution",
        "method": "occlusion_masking_logit_difference",
        "disclaimer": (
            "Event contribution reflects model attribution under the fitted model and observed "
            "inputs, not causal effect. Positive delta means masking lowered the logit (event "
            "was associated with higher model risk); negative delta means the opposite."
        ),
        "definitions": {
            "baseline_contribution": "delta_logit = z_base - z_base_with_event_masked (preoperative events)",
            "online_contribution": "delta_logit = z_t - z_t_with_event_masked for postop query time t",
            "risk_update_contribution": "delta_logit_new_event = z_t - z_t_masked for events arriving at step t",
            "grouping": ["modality", "raw_code", "time_window (preop / 0-24h / 24-72h / 72h+)"],
            "medication": "Medication events mask code and the linked med_day_feats exposure column.",
            "laboratory": "Lab events mask code, value, and value_mask together.",
        },
        "checkpoint": str(Path(checkpoint).resolve()),
        "episode_path": str(Path(episode_path).resolve()),
        "settings": settings,
        "cohort_summary": cohort,
        "top_episode_level_online_aki_positive": top_contributors(
            episodes_attr, label_aki=1, attr_key="online_attribution", top_k=30
        ),
        "top_episode_level_online_aki_negative": top_contributors(
            episodes_attr, label_aki=0, attr_key="online_attribution", top_k=30
        ),
        "case_study_person_ids": [e.get("person_id") for e in select_case_study_episodes(episodes_attr)],
        "episodes": episodes_attr,
    }


def plot_case_study(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    attr_ep: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    out_path: Path,
    alert_tau: float,
    dpi: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    batch, seq, _ = episode_to_batch(episode, code_vocab, lab_stats, num_med_exposure, device)
    z_base, _, postop_idx, z_postop = forward_logits(model, batch)
    probs = [float(1.0 / (1.0 + np.exp(-z))) for z in z_postop]
    hours = [float(seq[i].get("time_hour", 0.0)) for i in postop_idx]

    query_t = None
    for h, p in zip(hours, probs):
        if p >= alert_tau:
            query_t = h
            break
    if query_t is None and hours:
        query_t = hours[-1]

    online = attr_ep.get("online_attribution", [])
    at_query = [r for r in online if query_t is not None and abs(r["query_time_hour"] - query_t) < 1e-3]
    if not at_query and online:
        last_q = max(r["query_time_hour"] for r in online)
        at_query = [r for r in online if abs(r["query_time_hour"] - last_q) < 1e-3]

    pos = sorted(at_query, key=lambda r: r["delta_logit"], reverse=True)[:8]
    neg = sorted(at_query, key=lambda r: r["delta_logit"])[:8]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2, 2]})

    ax0 = axes[0]
    if hours:
        ax0.plot(hours, probs, "C1-", linewidth=1.5, label="online AKI risk")
    ax0.axhline(alert_tau, color="gray", linestyle="--", linewidth=0.8, label=f"alert τ={alert_tau}")
    ax0.axhline(float(1.0 / (1.0 + np.exp(-z_base))), color="C0", linestyle=":", label="baseline risk")
    if query_t is not None and hours:
        qi = int(np.argmin([abs(h - query_t) for h in hours]))
        ax0.axvline(query_t, color="red", linewidth=1.2, label="explanation time")
        ax0.scatter([hours[qi]], [probs[qi]], c="red", s=40, zorder=5)
    ax0.set_ylabel("Predicted AKI probability")
    ax0.set_title(f"Case {episode.get('person_id')} (AKI label={attr_ep.get('label_aki')})")
    ax0.legend(fontsize=7, loc="upper left")
    ax0.grid(True, alpha=0.25)

    ax1 = axes[1]
    modality_colors = {
        "condition": "C2",
        "lab": "C3",
        "med": "C4",
        "procedure": "C5",
        "surgery": "C6",
        "static": "C7",
    }
    for tok in seq:
        if int(tok.get("is_postop", 0)) != 1:
            continue
        th = float(tok.get("time_hour", 0.0))
        mod = str(tok.get("modality", "other"))
        ax1.scatter([th], [0], c=modality_colors.get(mod, "gray"), s=12, alpha=0.7)
    ax1.set_yticks([])
    ax1.set_ylabel("Events")
    if query_t is not None:
        ax1.axvline(query_t, color="red", linewidth=1.0)

    ax2 = axes[2]
    ax2.axis("off")
    lines = ["Top model-attributed contributors at explanation time:", ""]
    lines.append("Risk-increasing (masking lowers logit):")
    for r in pos:
        lines.append(f"  + [{r['modality']}] {r['raw_code']}: Δlogit={r['delta_logit']:.3f}")
    lines.append("")
    lines.append("Risk-decreasing (masking raises logit):")
    for r in neg:
        lines.append(f"  - [{r['modality']}] {r['raw_code']}: Δlogit={r['delta_logit']:.3f}")
    lines.append("")
    lines.append("Note: attribution ≠ causation.")
    ax2.text(0.02, 0.98, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace")

    axes[-1].set_xlabel("Postoperative hours")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


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

    episodes_attr: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes):
        print(f"Attribution episode {i + 1}/{len(episodes)} person_id={ep.get('person_id')}")
        episodes_attr.append(
            attributions_for_episode(
                model,
                ep,
                code_vocab,
                lab_stats,
                num_med,
                device,
                max_preop_occlusions=args.max_preop_occlusions,
                max_postop_query_steps=args.max_postop_query_steps,
                lookback_hours=args.lookback_hours,
            )
        )

    settings = {
        "max_episodes": args.max_episodes,
        "max_preop_occlusions": args.max_preop_occlusions,
        "max_postop_query_steps": args.max_postop_query_steps,
        "lookback_hours": args.lookback_hours,
    }
    report = build_report(episodes_attr, args.checkpoint, args.episode_path, settings)

    save_json(report, out_dir / "event_attribution.json")
    slim = {k: v for k, v in report.items() if k != "episodes"}
    save_json(slim, out_dir / "event_attribution_summary.json")

    if args.plot_case_studies:
        cases = select_case_study_episodes(
            episodes_attr, n_aki=args.n_case_aki, n_non=args.n_case_non
        )
        ep_by_id = {str(e.get("person_id")): e for e in episodes}
        for attr_ep in cases:
            pid = str(attr_ep.get("person_id"))
            ep = ep_by_id.get(pid)
            if ep is None:
                continue
            plot_case_study(
                model,
                ep,
                attr_ep,
                code_vocab,
                lab_stats,
                num_med,
                device,
                out_dir / f"case_study_{pid}.png",
                alert_tau=args.alert_threshold,
                dpi=args.dpi,
            )

    print("=== Model-attributed event contribution (secondary / interpretability) ===")
    cohort = report["cohort_summary"]
    print(f"  episodes: {cohort.get('n_episodes')} (AKI+: {cohort.get('n_aki_positive')})")
    mod = cohort.get("online_by_modality", {})
    for k in sorted(mod.keys()):
        print(f"  online mean Δlogit by modality [{k}]: {mod[k].get('mean_delta_logit')} (n={mod[k].get('n')})")
    print(f"Wrote {out_dir / 'event_attribution.json'}")


if __name__ == "__main__":
    main()
