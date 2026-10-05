#!/usr/bin/env python3
"""Cheap case-study screen: one forward per episode, sequential online-head path.

Writes per-token rows (postop, time_hour <= --max_hour) and per-episode metadata so
candidate AKI+/AKI− case studies can be ranked before the order-robust re-run.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_attribution import episode_to_batch, forward_logits
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device


def _aki_hours(ep: dict) -> float | None:
    t0, aki = ep.get("t0"), ep.get("aki_time")
    if not (t0 and aki):
        return None
    try:
        a = datetime.fromisoformat(str(aki).replace(" ", "T"))
        b = datetime.fromisoformat(str(t0).replace(" ", "T"))
        return (a - b).total_seconds() / 3600.0
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--episode_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--max_hour", type=float, default=168.0)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device).eval()

    tok_rows, ep_rows = [], []
    for i, ep in enumerate(episodes, 1):
        if i % 2000 == 0 or i == 1:
            print(f"  {i}/{len(episodes)}", flush=True)
        batch, seq, label = episode_to_batch(
            ep, code_vocab, lab_stats, num_med, device, age_stats=age_stats
        )
        z_base, _bi, postop_idx, z_postop = forward_logits(model, batch)
        pid = int(ep.get("person_id"))
        procs = ep.get("procedure_tokens") or []
        ep_rows.append({
            "person_id": pid,
            "label_aki": int(label),
            "aki_hours": _aki_hours(ep),
            "procedure_code": str(procs[0].get("code")) if procs else None,
            "z_base": float(z_base),
            "n_postop": len(postop_idx),
        })
        z_prev = float(z_base)
        for step, (pos, z) in enumerate(zip(postop_idx, z_postop)):
            tok = seq[pos]
            th = float(tok.get("time_hour", 0.0))
            if th > args.max_hour:
                break
            tok_rows.append({
                "person_id": pid,
                "step": step,
                "time_hour": th,
                "modality": str(tok.get("modality", "")),
                "raw_code": str(tok.get("raw_code", "")),
                "logit": float(z),
                "delta_logit": float(z) - z_prev,
            })
            z_prev = float(z)

    pd.DataFrame(ep_rows).to_parquet(out / "episodes.parquet", index=False)
    pd.DataFrame(tok_rows).to_parquet(out / "tokens.parquet", index=False)
    (out / "meta.json").write_text(json.dumps({"checkpoint": args.checkpoint, "n": len(ep_rows)}))
    print("done", out, len(ep_rows), len(tok_rows))


if __name__ == "__main__":
    main()
