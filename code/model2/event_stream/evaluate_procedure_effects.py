#!/usr/bin/env python3
"""Case-mix-free risk ranking of index operations, by counterfactual substitution.

Every episode in this cohort has exactly one procedure token, pinned to the surgery index, drawn
from a small set of operations. Deleting it is therefore not a valid ablation: no training
episode lacks a procedure, so the model is asked an unseen question and the resulting delta
reflects distributional novelty. Substitution keeps the intervention inside the observed data.

Each patient is rescored under every candidate operation, so operations are compared on an
identical panel of patients and the ranking carries no case-mix confounding.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_contribution import procedure_substitution_for_episode, rank_procedures
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_episodes", type=int, default=0,
                   help="Cap episodes (0 = full file).")
    p.add_argument("--max_query_steps", type=int, default=25)
    p.add_argument("--min_procedure_count", type=int, default=5,
                   help="Only use operations seen at least this often as candidates.")
    p.add_argument("--reference_procedure", type=str, default="",
                   help="Reference operation; default is the median-risk one.")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def collect_candidates(episodes: Sequence[Dict[str, Any]], min_count: int) -> List[str]:
    counts: Counter = Counter()
    for ep in episodes:
        for e in ep.get("events", []):
            if str(e.get("modality", "")).lower() == "procedure":
                counts[str(e.get("code"))] += 1
    return [c for c, n in counts.most_common() if n >= min_count]


def plot_procedure_ranking(
    ranked: Sequence[Dict[str, Any]], reference: str, out_path: Path, dpi: int
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = list(ranked)[::-1]
    if not rows:
        return
    vals = [r["mean_delta_logit_vs_reference"] for r in rows]
    errs = [r["std_delta_logit_vs_reference"] for r in rows]
    labels = [f"{r['procedure_code']}  (n={r['n_episodes_actually_this_procedure']})" for r in rows]
    colors = ["#c0392b" if v > 0 else "#2471a3" for v in vals]

    fig, ax = plt.subplots(figsize=(9.5, 0.30 * len(rows) + 2.2))
    ax.barh(range(len(rows)), vals, xerr=errs, color=colors, alpha=0.85,
            error_kw={"elinewidth": 0.6, "ecolor": "0.5"})
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.axvline(0, color="0.3", linewidth=0.8)
    ax.set_xlabel(f"Mean Δlogit at the surgery index, relative to {reference}")
    ax.set_title("Counterfactual risk of each operation on an identical panel of patients\n"
                 "every patient rescored under every operation, so no case-mix adjustment is needed",
                 fontsize=9)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    candidates = collect_candidates(episodes, args.min_procedure_count)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if args.num_shards > 1:
        episodes = [ep for i, ep in enumerate(episodes) if i % args.num_shards == args.shard_id]
    print(f"  candidate operations: {len(candidates)} (min_count={args.min_procedure_count})")
    print(f"  episodes this worker: {len(episodes)}  shard={args.shard_id}/{args.num_shards}")

    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()

    rows: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes, 1):
        if i % 20 == 0 or i == 1:
            print(f"  substitution {i}/{len(episodes)}")
        rows.extend(
            procedure_substitution_for_episode(
                model, ep, code_vocab, lab_stats, num_med, device,
                candidate_codes=candidates, max_query_steps=args.max_query_steps,
                age_stats=age_stats,
            )
        )

    ranking = rank_procedures(rows, args.reference_procedure or None)
    ranked = ranking.get("ranked", [])

    # How much of the model's surgery-index risk is decided by the operation alone: for each
    # patient, the spread of logits across all candidate operations with everything else fixed.
    spreads: List[float] = []
    by_person: Dict[str, List[float]] = {}
    for r in rows:
        by_person.setdefault(r["person_id"], []).append(r["baseline_logit"])
    for vals in by_person.values():
        if len(vals) > 1:
            spreads.append(max(vals) - min(vals))

    actual = [r for r in rows if r["is_actual"]]
    report = {
        "analysis_type": "index_procedure_counterfactual_substitution",
        "method": "swap_procedure_code_hold_patient_fixed",
        "rationale": (
            "Every episode has exactly one procedure token at the surgery index, so deleting it "
            "creates a configuration absent from training and its delta is not interpretable as "
            "surgical risk. Substitution stays on-manifold, and rescoring the same patients under "
            "every candidate removes case-mix confounding from the between-operation comparison."
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path).resolve()),
        "settings": {
            "max_episodes": args.max_episodes,
            "max_query_steps": args.max_query_steps,
            "min_procedure_count": args.min_procedure_count,
            "reference_procedure": ranking.get("reference_procedure"),
        },
        "n_episodes": len(by_person),
        "n_candidate_procedures": len(candidates),
        "within_patient_logit_spread": {
            "mean": float(np.mean(spreads)) if spreads else None,
            "median": float(np.median(spreads)) if spreads else None,
            "p90": float(np.percentile(spreads, 90)) if spreads else None,
            "max": float(np.max(spreads)) if spreads else None,
            "note": "Range of surgery-index logit across operations for one fixed patient record",
        },
        "observed_procedure_mix": dict(Counter(r["actual_procedure"] for r in actual).most_common()),
        "ranking": ranking,
    }
    save_json(report, out_dir / "procedure_effects_summary.json")

    fields = ["procedure_code", "n_episodes_actually_this_procedure", "n_panel_patients",
              "mean_logit", "mean_delta_logit_vs_reference", "std_delta_logit_vs_reference"]
    csv_path = out_dir / "procedure_counterfactual_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in ranked:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  Wrote {csv_path} (n={len(ranked)})")

    if args.plot:
        plot_procedure_ranking(ranked, ranking.get("reference_procedure", ""),
                               out_dir / "procedure_counterfactual_ranking.png", args.dpi)

    sp = report["within_patient_logit_spread"]
    print("=== Index procedure, counterfactual substitution ===")
    print(f"  patients: {report['n_episodes']}   candidate operations: {len(candidates)}")
    print(f"  reference: {ranking.get('reference_procedure')}")
    print(f"  within-patient logit spread across operations: "
          f"median={sp['median']:.3f}  p90={sp['p90']:.3f}  max={sp['max']:.3f}")
    print("  highest-risk operations:")
    for i, r in enumerate(ranked[:8], 1):
        print(f"    {i}. {r['procedure_code']:24s} Δlogit={r['mean_delta_logit_vs_reference']:+.3f} "
              f"(observed n={r['n_episodes_actually_this_procedure']})")
    print("  lowest-risk operations:")
    for i, r in enumerate(ranked[-8:][::-1], 1):
        print(f"    {i}. {r['procedure_code']:24s} Δlogit={r['mean_delta_logit_vs_reference']:+.3f} "
              f"(observed n={r['n_episodes_actually_this_procedure']})")
    print(f"Wrote {out_dir / 'procedure_effects_summary.json'}")


if __name__ == "__main__":
    main()
