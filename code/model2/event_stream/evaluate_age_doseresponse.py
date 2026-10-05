#!/usr/bin/env python3
"""Directional age effect, by counterfactual value sweep.

Deletion cannot answer "does being older raise predicted AKI risk?". Removing the AGE token
withdraws the whole age channel, so its delta is the average worth of knowing the age at all,
pooled over young and old patients whose effects point in opposite directions. Whatever sign
survives that average is an artefact of the cohort's age mix, not a dose-response direction.

Holding the token in place and overwriting its value asks the question directly. Every patient
is re-scored at every age on the grid, so the curve is within-patient and needs no case-mix
adjustment, and it is read on the [BASE] head, which only attends to preoperative tokens and is
therefore unaffected by the truncation leak.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_contribution import age_dose_response_for_episode
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_episodes", type=int, default=0)
    p.add_argument("--sample_n", type=int, default=0,
                   help="Random subsample of episodes; the curve is a within-patient contrast, "
                        "so a few thousand patients already pin it down.")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--age_min", type=float, default=20.0)
    p.add_argument("--age_max", type=float, default=90.0)
    p.add_argument("--age_step", type=float, default=5.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_csv).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    grid = list(np.arange(args.age_min, args.age_max + 1e-9, args.age_step))
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
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
    print(f"  {len(episodes)} episodes x {len(grid)} ages on {device}", flush=True)

    rows: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes, 1):
        if i % 100 == 0 or i == 1:
            print(f"  sweep {i}/{len(episodes)}", flush=True)
        rows.extend(
            age_dose_response_for_episode(
                model,
                ep,
                code_vocab,
                lab_stats,
                num_med,
                device,
                age_grid=grid,
                age_stats=age_stats,
            )
        )

    fields = ["person_id", "label_aki", "actual_age", "age_set", "baseline_logit"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out}  ({len(rows)} rows, {len({r['person_id'] for r in rows})} patients)")


if __name__ == "__main__":
    main()
