#!/usr/bin/env python3
"""Export per-episode baseline + fixed 6h postop AKI risk grid to JSONL.

For each episode in the input JSONL (e.g. 7d_window_clm.mixed.test.jsonl):
  - baseline: [BASE] token probability (full episode window)
  - buckets: every ``bucket_size`` hours from 0 to 30 days (720h), or until AKI+ last event bin
  - at each bucket: truncate to events with time_hour < bucket_end, forward, carry-forward risk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.fixed_time_grid import (
    _episode_label,
    _max_bucket_for_episode,
    episode_fixed_grid_rows,
    predict_probs_on_truncated_episode,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device


def _ensure_events(ep: Dict[str, Any]) -> Dict[str, Any]:
    """CLM JSONL may already include ``events``; otherwise merge modality lists."""
    if ep.get("events"):
        return ep
    parts = (
        ep.get("static_tokens", [])
        + ep.get("procedure_tokens", [])
        + ep.get("condition_events", [])
        + ep.get("lab_events", [])
        + ep.get("med_events", [])
    )
    if not parts:
        return ep
    out = dict(ep)
    out["events"] = sorted(parts, key=lambda x: x.get("time", ""))
    return out


def _hours_from_t0(t0: Any, t_event: Any) -> Optional[float]:
    from datetime import datetime

    def _parse(x: Any):
        if x is None:
            return None
        s = str(x).strip().replace("Z", "")
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(s)
        except ValueError:
            return None

    t0_dt, te_dt = _parse(t0), _parse(t_event)
    if t0_dt is None or te_dt is None:
        return None
    return (te_dt - t0_dt).total_seconds() / 3600.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export baseline + 6h bucket AKI risks per episode (JSONL).")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_jsonl", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--bucket_size", type=float, default=6.0)
    p.add_argument("--bucket_start", type=float, default=0.0)
    p.add_argument("--bucket_end", type=float, default=720.0, help="30 days in hours.")
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all episodes.")
    p.add_argument(
        "--no_carry_forward",
        action="store_true",
        help="Disable carry-forward between buckets (NaN if no postop token in bucket).",
    )
    p.add_argument("--no_progress", action="store_true")
    p.add_argument(
        "--episodes_only",
        action="store_true",
        help="Do not write a metadata line as the first JSONL row.",
    )
    return p.parse_args()


def _stopped_reason(
    label_aki: int,
    rows: List[Dict[str, Any]],
    bucket_end: float,
) -> str:
    if not rows:
        return "no_buckets"
    last_h = float(rows[-1]["bucket_hour_end"])
    if int(label_aki) == 1 and last_h < float(bucket_end) - 1e-6:
        return "aki_onset_last_event_bin"
    return "horizon_30d"


def build_episode_record(
    ep: Dict[str, Any],
    model: torch.nn.Module,
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    *,
    bucket_size: float,
    bucket_start: float,
    bucket_end: float,
    carry_forward: bool,
) -> Dict[str, Any]:
    events = ep.get("events", []) or []
    y_ep = int(_episode_label(ep) >= 0.5)

    baseline_prob = float("nan")
    try:
        _, base_v = predict_probs_on_truncated_episode(
            ep, model, code_vocab, lab_stats, num_med_exposure, device
        )
        if base_v is not None:
            baseline_prob = float(base_v)
    except Exception:
        pass

    grid_rows = episode_fixed_grid_rows(
        ep,
        model,
        code_vocab,
        lab_stats,
        num_med_exposure,
        device,
        bucket_size=bucket_size,
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        carry_forward=carry_forward,
    )

    buckets = np.arange(bucket_start, bucket_end + bucket_size, bucket_size, dtype=np.float64)
    max_b = _max_bucket_for_episode(events, y_ep, buckets, bucket_size) if len(buckets) >= 2 else 0

    bucket_preds: List[Dict[str, Any]] = []
    for r in grid_rows:
        bucket_preds.append(
            {
                "bucket_hour": int(r["bucket_hour"]),
                "bucket_hour_end": float(r["bucket_hour_end"]),
                "hour_range": r["hour_range"],
                "aki_risk_prob": float(r["y_prob"]) if np.isfinite(r["y_prob"]) else None,
                "aki_risk_prob_raw_postop": r["y_prob_raw_postop"],
                "baseline_prob_at_cutoff": float(r["baseline_prob"])
                if np.isfinite(r["baseline_prob"])
                else None,
                "carry_source": r["carry_source"],
                "n_events_truncated": int(r["n_events_truncated"]),
            }
        )

    t_aki_hours = _hours_from_t0(ep.get("t0"), ep.get("aki_time"))

    return {
        "person_id": ep.get("person_id"),
        "label_aki": int(y_ep),
        "aki_time": ep.get("aki_time"),
        "t0": ep.get("t0"),
        "t_aki_hours": float(t_aki_hours) if t_aki_hours is not None else None,
        "window_start": ep.get("window_start"),
        "window_end": ep.get("window_end"),
        "baseline_aki_risk_prob": baseline_prob if np.isfinite(baseline_prob) else None,
        "bucket_size_hours": float(bucket_size),
        "postop_horizon_hours": float(bucket_end),
        "max_bucket_hour_evaluated": int(max_b),
        "stopped_reason": _stopped_reason(y_ep, grid_rows, bucket_end),
        "n_buckets": len(bucket_preds),
        "carry_forward": bool(carry_forward),
        "bucket_predictions": bucket_preds,
    }


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

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

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    iterator: Any = episodes
    if not args.no_progress and tqdm is not None:
        iterator = tqdm(episodes, desc="export grid predictions")

    meta = {
        "type": "export_metadata",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path).resolve()),
        "bucket_size_hours": args.bucket_size,
        "bucket_start_hours": args.bucket_start,
        "bucket_end_hours": args.bucket_end,
        "carry_forward": not args.no_carry_forward,
        "protocol": (
            "baseline = [BASE] on full episode; each bucket [b,b+6) uses events with "
            "time_hour < b+6, last postop token prob with carry-forward."
        ),
        "n_episodes": len(episodes),
    }

    n_written = 0
    with out_path.open("w", encoding="utf-8") as f:
        if not args.episodes_only:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for ep in iterator:
            ep = _ensure_events(ep)
            rec = build_episode_record(
                ep,
                model,
                code_vocab,
                lab_stats,
                num_med,
                device,
                bucket_size=args.bucket_size,
                bucket_start=args.bucket_start,
                bucket_end=args.bucket_end,
                carry_forward=not args.no_carry_forward,
            )
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_written += 1

    meta_note = "" if args.episodes_only else " + 1 metadata line"
    print(f"Wrote {out_path} ({n_written} episodes{meta_note})")
    print(f"  device={device} bucket_size={args.bucket_size}h horizon={args.bucket_end}h")


if __name__ == "__main__":
    main()
