#!/usr/bin/env python3
"""Smoke / cohort evaluation of event-arrival Δlogit along the streaming head."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_arrival import event_arrivals_for_episode, pool_arrival_by_code
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--sample_n", type=int, default=0,
                   help="Random subsample of episodes (0 = all loaded).")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--min_patients", type=int, default=25)
    p.add_argument("--top_k", type=int, default=15)
    p.add_argument(
        "--n_tie_perm",
        type=int,
        default=20,
        help="Monte Carlo perms for same-timestamp blocks; if k!≤n_tie_perm, enumerate all.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--person_ids",
        type=str,
        default="",
        help="Comma-separated person_ids to evaluate (case studies). Empty = all.",
    )
    p.add_argument("--episode_start", type=int, default=0,
                   help="0-based inclusive start into the episode list (shards).")
    p.add_argument("--episode_end", type=int, default=0,
                   help="0-based exclusive end; 0 = len(episodes).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    if args.person_ids.strip():
        want = {s.strip() for s in args.person_ids.split(",") if s.strip()}
        episodes = [ep for ep in episodes if str(ep.get("person_id")) in want]
        print(f"  filtered to person_ids={sorted(want)} -> {len(episodes)} episodes", flush=True)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if args.episode_start > 0 or args.episode_end > 0:
        end = args.episode_end if args.episode_end > 0 else len(episodes)
        episodes = episodes[args.episode_start:end]
        print(f"  shard [{args.episode_start},{end}) -> {len(episodes)} episodes", flush=True)
    if 0 < args.sample_n < len(episodes):
        pick = np.random.default_rng(args.sample_seed).choice(
            len(episodes), size=args.sample_n, replace=False
        )
        episodes = [episodes[i] for i in sorted(pick)]

    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()
    print(
        f"  event-arrival (order-robust, n_tie_perm={args.n_tie_perm}) "
        f"on {len(episodes)} episodes @ {device}",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    n_skip = 0
    for i, ep in enumerate(episodes, 1):
        if i % 200 == 0 or i == 1:
            print(f"  {i}/{len(episodes)}  rows={len(rows)}", flush=True)
        part = event_arrivals_for_episode(
            model, ep, code_vocab, lab_stats, num_med, device,
            n_tie_perm=args.n_tie_perm, seed=args.seed, age_stats=age_stats,
        )
        if not part:
            n_skip += 1
            continue
        rows.extend(part)

    step_path = out / "event_arrival_steps.csv"
    fields = [
        "person_id", "label_aki", "step", "seq_index", "prev_seq_index",
        "modality", "raw_code", "time_hour", "time_bin",
        "logit", "prev_logit", "delta_logit",
        "prob", "prev_prob", "delta_prob", "baseline_logit",
        "attribution_mode", "tie_size", "block_delta_logit", "n_perm_used",
    ]
    with step_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    pooled = pool_arrival_by_code(rows, min_n=args.min_patients)
    pool_path = out / "event_arrival_by_code.csv"
    if pooled:
        with pool_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(pooled[0].keys()))
            w.writeheader()
            w.writerows(pooled)

    deltas = np.asarray([r["delta_logit"] for r in rows], dtype=float)
    n_shapley = sum(1 for r in rows if r.get("attribution_mode") == "shapley_mc")
    summary = {
        "n_episodes": len(episodes),
        "n_episodes_with_postop": len(episodes) - n_skip,
        "n_arrival_steps": len(rows),
        "n_shapley_rows": n_shapley,
        "n_tie_perm": args.n_tie_perm,
        "mean_abs_delta_logit": float(np.abs(deltas).mean()) if len(deltas) else 0.0,
        "median_abs_delta_logit": float(np.median(np.abs(deltas))) if len(deltas) else 0.0,
        "p90_abs_delta_logit": float(np.percentile(np.abs(deltas), 90)) if len(deltas) else 0.0,
        "frac_positive_delta": float((deltas > 0).mean()) if len(deltas) else 0.0,
        "n_codes_pooled": len(pooled),
        "attribution": "order-robust hybrid (sequential |G|=1; Shapley-MC |G|>1)",
    }
    (out / "event_arrival_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== event-arrival summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if pooled:
        print(f"\nTop {args.top_k} by |mean Δlogit| (patient-pooled, n≥{args.min_patients}):")
        for i, r in enumerate(pooled[: args.top_k], 1):
            print(
                f"  {i:2d}. [{r['modality']:9s}] {r['raw_code'][:32]:32s}  "
                f"Δz={r['mean_delta_logit']:+.4f}  |Δz|={r['mean_abs_delta_logit']:.4f}  "
                f"n={r['n_patients']}"
            )
    print(f"\nWrote {step_path}")
    if pooled:
        print(f"Wrote {pool_path}")
    print(f"Wrote {out / 'event_arrival_summary.json'}")


if __name__ == "__main__":
    main()
