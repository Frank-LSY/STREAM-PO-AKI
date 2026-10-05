#!/usr/bin/env python3
"""Same-timestamp permutation invariance (supplementary robustness).

For events that share an identical recorded ``time_hour``, randomly shuffle
their *sequence* order (timestamps unchanged / already equal). This probes
sensitivity to database row-order among simultaneous events — not clinically
resolved temporal order.

Ideal: |Δz_tie| ≈ 0. Large values are a representation artifact, not evidence
of event-stream superiority.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from event_stream.evaluate_within_bin_order_sensitivity import (
    ES_CKPT,
    ROOT,
    TEST_JSONL,
    end_of_bin_logit,
    find_candidate_bins,
)

OUT_DIR = ROOT / "result/figures/order_sensitivity/supp_same_timestamp"


def _tie_groups_in_block(
    events: Sequence[Dict[str, Any]],
    indices: Sequence[int],
) -> List[List[int]]:
    """Local positions within ``indices`` that share exact time_hour."""
    by_t: Dict[float, List[int]] = defaultdict(list)
    for loc, gi in enumerate(indices):
        th = events[gi].get("time_hour")
        if th is None:
            continue
        by_t[float(th)].append(loc)
    return [locs for locs in by_t.values() if len(locs) >= 2]


def _shuffle_same_timestamp_only(
    events: List[Dict[str, Any]],
    indices: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Permute only within exact-timestamp tie groups; leave other slots fixed.

    Returns (new_events, n_tie_groups_shuffled, n_events_in_ties).
    """
    out = list(events)
    block = [copy.deepcopy(events[i]) for i in indices]
    groups = _tie_groups_in_block(events, indices)
    n_ev = 0
    for locs in groups:
        n_ev += len(locs)
        order = list(rng.permutation(len(locs)))
        src = [block[locs[j]] for j in range(len(locs))]
        for j, src_j in enumerate(order):
            # Identical timestamps: whole-event move ≡ identity-at-fixed-time.
            block[locs[j]] = copy.deepcopy(src[src_j])
    for loc, ev in enumerate(block):
        out[indices[loc]] = ev
    return out, len(groups), n_ev


def _eligible_tie_windows(
    episodes: Sequence[Dict[str, Any]],
    *,
    min_k: int,
    min_distinct: int,
    min_tie_events: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], List[List[int]]]]:
    out = []
    for ep in episodes:
        events = ep.get("events") or []
        for cand in find_candidate_bins(ep, min_k=min_k, min_distinct=min_distinct):
            groups = _tie_groups_in_block(events, cand["indices"])
            n_tie = sum(len(g) for g in groups)
            if n_tie < min_tie_events or not groups:
                continue
            out.append((ep, cand, groups))
    return out


def plot_hist(abs_dz: np.ndarray, out: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.hist(abs_dz, bins=40, color="#7f8c8d", edgecolor="white", alpha=0.9)
    ax.axvline(float(np.median(abs_dz)), color="#c0392b", lw=1.8, label=f"median={np.median(abs_dz):.4g}")
    ax.axvline(0.05, color="0.4", ls=(0, (3, 2)), lw=1.0, label="0.05 ref")
    ax.set_xlabel(r"$|\Delta z_{\mathrm{tie}}|$")
    ax.set_ylabel("Count (window × shuffle)")
    ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: Dict[str, Any], out: Path) -> None:
    es = summary["event_stream"]
    lines = [
        "# Same-timestamp permutation invariance (supplementary robustness)",
        "",
        "Shuffle **only** events that share an identical recorded `time_hour` "
        "inside an eligible 6 h bin. Timestamps stay equal; clinically unresolved "
        "row-order among simultaneous events is the sole perturbation.",
        "",
        f"**Eligible windows (with ties):** {summary['n_windows']}  ",
        f"**Shuffles per window:** {summary['n_shuffle']}  ",
        "",
        "## Results",
        "",
        "| Metric | Value |",
        "|--|--:|",
        f"| N (window × shuffle) | {es['n']} |",
        f"| Mean $|\\Delta z_{{\\mathrm{{tie}}}}|$ | {es['mean']:.6f} |",
        f"| Median $|\\Delta z_{{\\mathrm{{tie}}}}|$ | {es['median']:.6f} |",
        f"| P90 | {es['p90']:.6f} |",
        f"| P99 | {es['p99']:.6f} |",
        f"| Frac $>0.05$ | {es['frac_gt_0_05']:.4f} |",
        f"| Frac $>0.10$ | {es['frac_gt_0_10']:.4f} |",
        "",
        "![hist](same_timestamp_abs_dz_hist.png)",
        "",
        "## Interpretation",
        "",
    ]
    if es["median"] < 0.01 and es["frac_gt_0_05"] < 0.05:
        lines.append(
            "Predictions are **insensitive** to arbitrary ordering among events sharing "
            "an identical recorded timestamp ($|\\Delta z_{\\mathrm{tie}}|\\approx 0$). "
            "This supports that the primary within-bin order sensitivity is not driven by "
            "database row-order artifacts among simultaneous events."
        )
    elif es["median"] < 0.05:
        lines.append(
            "Same-timestamp sensitivity is **small** relative to chronology-preserving "
            "distinct-time order perturbations, but not exactly zero — report as a "
            "robustness caveat and consider same-timestamp shuffle augmentation if needed."
        )
    else:
        lines.append(
            "**Warning:** non-trivial $|\\Delta z_{\\mathrm{tie}}|$ suggests the model uses "
            "unresolved row-order among simultaneous events. Treat as a representation "
            "artifact to fix (training shuffle / simultaneous blocks), not as evidence of "
            "clinically meaningful temporal ordering."
        )
    lines.append("")
    out.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all")
    p.add_argument("--max_windows", type=int, default=0, help="0 = all eligible tie windows")
    p.add_argument("--max_windows_per_episode", type=int, default=2)
    p.add_argument("--n_shuffle", type=int, default=5)
    p.add_argument("--min_k", type=int, default=3)
    p.add_argument("--min_distinct", type=int, default=2)
    p.add_argument("--min_tie_events", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    import sys

    sys.path.insert(0, str(ROOT / "code/model2"))
    from common.med_exposure import infer_num_med_exposure
    from common.utils import resolve_torch_device
    from event_stream.dataset import load_episode_jsonl
    from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
    from event_stream.event_attribution import episode_to_batch

    device = resolve_torch_device(args.device)
    print(f"device={device}", flush=True)
    episodes = load_episode_jsonl(TEST_JSONL)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    print(f"episodes={len(episodes)}", flush=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(str(ES_CKPT))
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    pool = _eligible_tie_windows(
        episodes,
        min_k=args.min_k,
        min_distinct=args.min_distinct,
        min_tie_events=args.min_tie_events,
    )
    # Cap per episode then global
    by_pid: Dict[str, List] = defaultdict(list)
    for item in pool:
        by_pid[str(item[0].get("person_id"))].append(item)
    capped: List = []
    for items in by_pid.values():
        if len(items) > args.max_windows_per_episode:
            pick = rng.choice(len(items), size=args.max_windows_per_episode, replace=False)
            items = [items[i] for i in pick]
        capped.extend(items)
    if args.max_windows > 0 and len(capped) > args.max_windows:
        pick = rng.choice(len(capped), size=args.max_windows, replace=False)
        capped = [capped[i] for i in pick]
    print(f"tie_windows={len(capped)}", flush=True)

    rows: List[Dict[str, Any]] = []
    for wi, (ep, cand, groups) in enumerate(capped):
        h0, h1 = float(cand["h0"]), float(cand["h1"])
        idxs = list(cand["indices"])
        batch0, seq0, _ = episode_to_batch(ep, code_vocab, lab_stats, num_med, device)
        z0 = end_of_bin_logit(model, batch0, seq0, h0, h1)
        if z0 is None:
            continue
        n_tie = sum(len(g) for g in groups)
        for s in range(args.n_shuffle):
            ep_p = copy.deepcopy(ep)
            new_events, n_grp, n_ev = _shuffle_same_timestamp_only(ep["events"], idxs, rng)
            ep_p["events"] = new_events
            batch_p, seq_p, _ = episode_to_batch(ep_p, code_vocab, lab_stats, num_med, device)
            z_p = end_of_bin_logit(model, batch_p, seq_p, h0, h1)
            if z_p is None:
                continue
            dz = float(z0 - z_p)
            rows.append(
                {
                    "person_id": str(ep.get("person_id")),
                    "bin_j": int(cand["bin_j"]),
                    "h0": h0,
                    "h1": h1,
                    "k": int(cand["k"]),
                    "n_distinct": int(cand["n_distinct"]),
                    "n_tie_groups": n_grp,
                    "n_tie_events": n_ev,
                    "shuffle": s,
                    "z_orig": z0,
                    "z_perm": float(z_p),
                    "delta_logit": dz,
                    "abs_delta_logit": abs(dz),
                }
            )
        if (wi + 1) % 100 == 0:
            print(f"  {wi+1}/{len(capped)}  rows={len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "same_timestamp_rows.csv"
    df.to_csv(csv_path, index=False)
    abs_dz = df["abs_delta_logit"].to_numpy(dtype=float) if len(df) else np.array([])
    es_sum = {
        "n": int(len(abs_dz)),
        "mean": float(np.mean(abs_dz)) if len(abs_dz) else 0.0,
        "median": float(np.median(abs_dz)) if len(abs_dz) else 0.0,
        "p90": float(np.percentile(abs_dz, 90)) if len(abs_dz) else 0.0,
        "p99": float(np.percentile(abs_dz, 99)) if len(abs_dz) else 0.0,
        "frac_gt_0_05": float(np.mean(abs_dz > 0.05)) if len(abs_dz) else 0.0,
        "frac_gt_0_10": float(np.mean(abs_dz > 0.10)) if len(abs_dz) else 0.0,
    }
    summary = {
        "n_windows": int(df[["person_id", "bin_j"]].drop_duplicates().shape[0]) if len(df) else 0,
        "n_shuffle": args.n_shuffle,
        "event_stream": es_sum,
        "paths": {"rows": str(csv_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if len(abs_dz):
        plot_hist(abs_dz, out_dir / "same_timestamp_abs_dz_hist.png", "Same-timestamp shuffle")
    write_report(summary, out_dir / "REPORT.md")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
