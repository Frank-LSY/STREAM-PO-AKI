#!/usr/bin/env python3
"""Within-bin order sensitivity: event-stream vs fixed 6h time-series.

Keeps the event *set*, values, and 6h bucket fixed; permutes which event
occupies which chronological timestamp slot inside the bin:

    x = {(e_i, t_i)}_{i=1..k}  with  t_1 < … < t_k
    x^{(π)} = {(e_{π(i)}, t_i)}_{i=1..k}

Timestamps / chronology stay legal; only within-bin event ordering changes.
Event-stream predictions can change; bag-of-features time-series
representations are order-invariant by construction.

Smoke / full report under ``result/figures/order_sensitivity/``.
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
import torch
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
ES_CKPT = ROOT / "result/event_stream/transformer/fold1/best_model.pt"
TS_CKPT = ROOT / "result/timeseries/transformer/fold1/best_model.pt"
TEST_JSONL = ROOT / "data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl"
TS_NPZ = ROOT / "data/five_folds_ts6h4/test.npz"
TS_LAYOUT = ROOT / "data/five_folds_ts6h4/vocab/feature_layout.json"
OUT_DIR = ROOT / "result/figures/order_sensitivity"

BUCKET_H = 6.0
HORIZON_H = 168.0  # first 7 postoperative days
MIN_K = 3
MIN_DISTINCT = 2

# Fields that define the chronological *slot* (kept at destination position).
_TIME_SLOT_KEYS = ("time_hour", "time_day")


def _bin_index(time_hour: float) -> int:
    return int(np.floor(float(time_hour) / BUCKET_H))


def _bin_bounds(j: int) -> Tuple[float, float]:
    return float(j * BUCKET_H), float((j + 1) * BUCKET_H)


def _is_postop_stream_event(ev: Dict[str, Any]) -> bool:
    th = ev.get("time_hour")
    if th is None:
        return False
    th = float(th)
    if th < 0.0 or th >= HORIZON_H:
        return False
    if str(ev.get("modality", "")).lower() == "procedure":
        return False
    return True


def find_candidate_bins(
    episode: Dict[str, Any],
    *,
    min_k: int = MIN_K,
    min_distinct: int = MIN_DISTINCT,
) -> List[Dict[str, Any]]:
    """Contiguous blocks of postop events inside the same 6h bin."""
    events = episode.get("events") or []
    by_bin: Dict[int, List[int]] = defaultdict(list)
    for i, ev in enumerate(events):
        if not _is_postop_stream_event(ev):
            continue
        j = _bin_index(float(ev["time_hour"]))
        by_bin[j].append(i)

    cands: List[Dict[str, Any]] = []
    for j, idxs in sorted(by_bin.items()):
        if len(idxs) < min_k:
            continue
        # Require contiguity in the stored event order (true if sorted by time).
        if idxs != list(range(min(idxs), max(idxs) + 1)):
            continue
        codes = [str(events[i].get("code", "")) for i in idxs]
        n_distinct = len(set(codes))
        if n_distinct < min_distinct:
            continue
        h0, h1 = _bin_bounds(j)
        cands.append(
            {
                "bin_j": int(j),
                "h0": h0,
                "h1": h1,
                "indices": list(idxs),
                "k": len(idxs),
                "n_distinct": n_distinct,
                "codes": codes,
            }
        )
    return cands


def _permute_event_list(
    events: List[Dict[str, Any]],
    indices: Sequence[int],
    order: Sequence[int],
) -> List[Dict[str, Any]]:
    """Chronology-preserving within-bin permutation.

    Destination slot ``indices[loc]`` keeps its original timestamps ``t_loc``;
    event identity + value come from ``indices[order[loc]]``:

        x^{(π)} = {(e_{π(i)}, t_i)}

    Sequence positions remain time-sorted; only which event occupies each
    timestamp changes. This avoids nonchronological (token, time) pairs.
    """
    out = list(events)
    block = [copy.deepcopy(events[i]) for i in indices]
    slot_times = [{k: events[i].get(k) for k in _TIME_SLOT_KEYS} for i in indices]
    for loc, src_loc in enumerate(order):
        ev = copy.deepcopy(block[src_loc])
        for k, v in slot_times[loc].items():
            if v is not None:
                ev[k] = v
        out[indices[loc]] = ev
    return out


def _permute_modality_lists(
    episode: Dict[str, Any],
    h0: float,
    h1: float,
    rng: np.random.Generator,
    *,
    reverse: bool = False,
) -> Dict[str, Any]:
    """Same chronology-preserving permute within each modality list (TS sanity)."""
    ep = copy.deepcopy(episode)

    def _perm_list(key: str) -> None:
        arr = list(ep.get(key) or [])
        in_bin = []
        for i, ev in enumerate(arr):
            th = ev.get("time_hour")
            if th is None:
                continue
            thf = float(th)
            if h0 <= thf < h1:
                in_bin.append(i)
        if len(in_bin) < 2:
            return
        block = [copy.deepcopy(arr[i]) for i in in_bin]
        slot_times = [{k: arr[i].get(k) for k in _TIME_SLOT_KEYS} for i in in_bin]
        if reverse:
            order = list(range(len(block)))[::-1]
        else:
            order = list(rng.permutation(len(block)))
        for loc, src in enumerate(order):
            ev = copy.deepcopy(block[src])
            for k, v in slot_times[loc].items():
                if v is not None:
                    ev[k] = v
            arr[in_bin[loc]] = ev
        ep[key] = arr

    for key in ("condition_events", "lab_events", "med_events", "events"):
        _perm_list(key)
    return ep


def make_permutations(k: int, rng: np.random.Generator, n_random: int = 3) -> List[Tuple[str, List[int]]]:
    """Identity excluded; reverse + random shuffles (deduplicated)."""
    seen = {tuple(range(k))}
    out: List[Tuple[str, List[int]]] = []
    rev = list(range(k))[::-1]
    if tuple(rev) not in seen:
        seen.add(tuple(rev))
        out.append(("reverse", rev))
    tries = 0
    while len([x for x in out if x[0].startswith("random")]) < n_random and tries < n_random * 20:
        tries += 1
        perm = list(rng.permutation(k))
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        out.append((f"random{len([x for x in out if x[0].startswith('random')])}", perm))
    return out


def seq_from_feat(feat: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(feat["sequence"])


def end_of_bin_logit(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    seq: List[Dict[str, Any]],
    h0: float,
    h1: float,
) -> Optional[float]:
    """Online logit at the last postoperative token whose time_hour ∈ [h0, h1)."""
    from event_stream.event_attribution import forward_logits

    z_base, base_idx, postop_idx, z_postop = forward_logits(model, batch)
    if not postop_idx:
        return None
    last_i = None
    last_z = None
    for pos, z in zip(postop_idx, z_postop):
        th = float(seq[pos].get("time_hour", -1.0))
        if h0 <= th < h1:
            last_i = pos
            last_z = float(z)
    return last_z


def run_es_window(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    cand: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Any,
    num_med: int,
    device: torch.device,
    rng: np.random.Generator,
    n_random: int = 3,
    age_stats: Any = None,
) -> List[Dict[str, Any]]:
    from event_stream.event_attribution import episode_to_batch

    h0, h1 = float(cand["h0"]), float(cand["h1"])
    idxs = list(cand["indices"])
    k = len(idxs)

    batch0, seq0, _ = episode_to_batch(
        episode, code_vocab, lab_stats, num_med, device, age_stats=age_stats
    )
    z0 = end_of_bin_logit(model, batch0, seq0, h0, h1)
    if z0 is None:
        return []

    rows: List[Dict[str, Any]] = []
    for name, order in make_permutations(k, rng, n_random=n_random):
        ep_p = copy.deepcopy(episode)
        ep_p["events"] = _permute_event_list(episode["events"], idxs, order)
        batch_p, seq_p, _ = episode_to_batch(
            ep_p, code_vocab, lab_stats, num_med, device, age_stats=age_stats
        )
        z_p = end_of_bin_logit(model, batch_p, seq_p, h0, h1)
        if z_p is None:
            continue
        dz = float(z0 - z_p)
        p0 = 1.0 / (1.0 + np.exp(-z0))
        pp = 1.0 / (1.0 + np.exp(-z_p))
        rows.append(
            {
                "model": "event_stream",
                "person_id": str(episode.get("person_id")),
                "bin_j": int(cand["bin_j"]),
                "h0": h0,
                "h1": h1,
                "k": k,
                "n_distinct": int(cand["n_distinct"]),
                "perm": name,
                "z_orig": z0,
                "z_perm": z_p,
                "delta_logit": dz,
                "abs_delta_logit": abs(dz),
                "delta_prob": float(p0 - pp),
                "abs_delta_prob": abs(float(p0 - pp)),
            }
        )
    return rows


def run_ts_window(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    layout: Dict[str, Any],
    lab_stats: Dict[str, Any],
    cand: Dict[str, Any],
    device: torch.device,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """Sanity: bag features + TS logits unchanged under within-bin permute."""
    from timeseries.timeseries_6h_common import (
        BASE_BUCKET_IDX,
        episode_to_bucket_features,
    )

    h0, h1 = float(cand["h0"]), float(cand["h1"])
    j = int(cand["bin_j"])
    # Feature row for this postop bucket
    x0 = episode_to_bucket_features(episode, layout, lab_stats)
    token_idx = BASE_BUCKET_IDX + 1 + j
    if token_idx >= x0.shape[0]:
        return []

    ep_rev = _permute_modality_lists(episode, h0, h1, rng, reverse=True)
    x_rev = episode_to_bucket_features(ep_rev, layout, lab_stats)
    feat_l2 = float(np.linalg.norm(x0[token_idx] - x_rev[token_idx]))

    def _forward_z(feats: np.ndarray) -> float:
        from timeseries.timeseries_6h_common import SEQ_LEN

        B = 1
        bf = torch.from_numpy(feats.astype(np.float32)).unsqueeze(0).to(device)
        attn = torch.ones(B, SEQ_LEN, device=device)
        base = torch.zeros(B, SEQ_LEN, device=device)
        base[:, BASE_BUCKET_IDX] = 1.0
        out = model(bucket_features=bf, attention_mask=attn, base_mask=base)
        z = out["token_logits"][0, token_idx]
        return float(z.item())

    z0 = _forward_z(x0)
    z_rev = _forward_z(x_rev)
    dz = float(z0 - z_rev)
    return [
        {
            "model": "timeseries",
            "person_id": str(episode.get("person_id")),
            "bin_j": int(cand["bin_j"]),
            "h0": h0,
            "h1": h1,
            "k": int(cand["k"]),
            "n_distinct": int(cand["n_distinct"]),
            "perm": "reverse",
            "z_orig": z0,
            "z_perm": z_rev,
            "delta_logit": dz,
            "abs_delta_logit": abs(dz),
            "feat_l2": feat_l2,
        }
    ]


def _summarize(abs_dz: np.ndarray) -> Dict[str, float]:
    if abs_dz.size == 0:
        return {}
    return {
        "n": int(abs_dz.size),
        "mean": float(abs_dz.mean()),
        "median": float(np.median(abs_dz)),
        "p90": float(np.percentile(abs_dz, 90)),
        "p99": float(np.percentile(abs_dz, 99)),
        "frac_gt_0_05": float((abs_dz > 0.05).mean()),
        "frac_gt_0_10": float((abs_dz > 0.10).mean()),
        "frac_gt_0_20": float((abs_dz > 0.20).mean()),
    }


def plot_es_hist(abs_dz: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.hist(abs_dz, bins=40, color="#1f6cb0", alpha=0.85, edgecolor="white", lw=0.4)
    ax.axvline(np.median(abs_dz), color="#c0392b", ls="--", lw=1.2, label=f"median={np.median(abs_dz):.3f}")
    ax.set_xlabel(r"$|\Delta z_{\mathrm{ES}}|$ (end-of-bin online logit)")
    ax.set_ylabel("Count (window × permutation)")
    ax.set_title("Within-bin order sensitivity — event-stream")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(
    path: Path,
    *,
    smoke: bool,
    n_episodes: int,
    n_windows: int,
    es_sum: Dict[str, float],
    ts_sum: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    lines = []
    lines.append("# Within-bin order sensitivity report")
    lines.append("")
    lines.append(f"**Mode:** `{'smoke' if smoke else 'full'}`  ")
    lines.append(f"**Episodes scanned:** {n_episodes}  ")
    lines.append(
        f"**Eligible windows (k≥{args.min_k}, distinct≥{args.min_distinct}, "
        f"0–{HORIZON_H:.0f}h):** {n_windows}  "
    )
    lines.append(f"**ES checkpoint:** `{ES_CKPT}`  ")
    lines.append(f"**TS checkpoint:** `{TS_CKPT}`  ")
    lines.append("")
    lines.append("## Method (brief)")
    lines.append("")
    lines.append(
        "For each postoperative 6h bin with multiple events, keep the event set / values / "
        "bin fixed and **reassign event identities onto the original chronological timestamp "
        "slots** (timestamps stay sorted; only which event occurs earlier/later changes):"
    )
    lines.append("")
    lines.append(r"$$\mathbf{x}=\{(e_i,t_i)\}_{i=1}^{k},\qquad \mathbf{x}^{(\pi)}=\{(e_{\pi(i)},t_i)\}_{i=1}^{k}$$")
    lines.append("")
    lines.append(r"$$\Delta z_{\mathrm{order}} = z(\mathbf{x}) - z(\mathbf{x}^{(\pi)})$$")
    lines.append("")
    lines.append(
        "This keeps the stream chronologically legal under the causal mask. "
        "Fixed-window time-series uses commutative bag features inside the bin, so "
        r"$X(x)=X(x^{(\pi)})$ and $\Delta z_{\mathrm{TS}}\equiv 0$ by construction."
    )
    lines.append("")
    lines.append("## Event-stream results")
    lines.append("")
    if es_sum:
        lines.append("| Metric | Value |")
        lines.append("|--|--:|")
        lines.append(f"| N (window × perm) | {es_sum['n']} |")
        lines.append(f"| Mean $|\\Delta z|$ | {es_sum['mean']:.4f} |")
        lines.append(f"| Median $|\\Delta z|$ | {es_sum['median']:.4f} |")
        lines.append(f"| P90 $|\\Delta z|$ | {es_sum['p90']:.4f} |")
        lines.append(f"| P99 $|\\Delta z|$ | {es_sum['p99']:.4f} |")
        lines.append(f"| Frac $|\\Delta z| > 0.05$ | {es_sum['frac_gt_0_05']:.3f} |")
        lines.append(f"| Frac $|\\Delta z| > 0.10$ | {es_sum['frac_gt_0_10']:.3f} |")
        lines.append(f"| Frac $|\\Delta z| > 0.20$ | {es_sum['frac_gt_0_20']:.3f} |")
    else:
        lines.append("_No ES rows._")
    lines.append("")
    lines.append("![ES histogram](order_sensitivity_es_hist.png)")
    lines.append("")
    lines.append("## Time-series null check")
    lines.append("")
    if ts_sum:
        lines.append("| Metric | Value |")
        lines.append("|--|--:|")
        lines.append(f"| N windows | {ts_sum.get('n', 0)} |")
        lines.append(f"| Max feature L2 $\\|X-X^{{(\\pi)}}\\|$ | {ts_sum.get('max_feat_l2', 0):.3e} |")
        lines.append(f"| Mean $|\\Delta z_{{\\mathrm{{TS}}}}|$ | {ts_sum.get('mean_abs_dz', 0):.3e} |")
        lines.append(f"| Max $|\\Delta z_{{\\mathrm{{TS}}}}|$ | {ts_sum.get('max_abs_dz', 0):.3e} |")
        ok = ts_sum.get("max_feat_l2", 1) < 1e-6 and ts_sum.get("max_abs_dz", 1) < 1e-5
        lines.append("")
        lines.append(
            "**Null confirmed.** "
            if ok
            else "**Warning:** TS null not numerically exact — inspect floating-point / bucket indexing. "
        )
        lines.append(
            "Within-bin permutation leaves the 6h bag representation unchanged, so TS predictions do not move."
        )
    else:
        lines.append("_TS check not run / empty._")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if es_sum and es_sum.get("median", 0) > 0.01:
        lines.append(
            "Event-stream end-of-bin logits **change under within-bin order permutation**, "
            "while the matched fixed-window representation is **order-invariant**. "
            "This is direct evidence that streaming modeling captures within-window temporal "
            "ordering that bag-of-features 6h time-series cannot represent."
        )
    else:
        lines.append(
            "ES effect is small in this smoke sample; re-run with more episodes / windows before a strong claim."
        )
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "PYTHONPATH=code/model2:$PYTHONPATH python "
        "code/model2/event_stream/evaluate_within_bin_order_sensitivity.py "
        f"--max_episodes {args.max_episodes} --max_windows {args.max_windows} "
        f"--seed {args.seed}"
    )
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max_episodes", type=int, default=200, help="Smoke default 200; 0 = all.")
    p.add_argument(
        "--max_windows",
        type=int,
        default=400,
        help="Cap eligible windows evaluated; 0 = no cap.",
    )
    p.add_argument("--max_windows_per_episode", type=int, default=2)
    p.add_argument("--n_random", type=int, default=3)
    p.add_argument("--min_k", type=int, default=MIN_K)
    p.add_argument("--min_distinct", type=int, default=MIN_DISTINCT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--skip_ts", action="store_true")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument(
        "--episode_start",
        type=int,
        default=0,
        help="0-based inclusive start into the episode list (multi-GPU shards).",
    )
    p.add_argument(
        "--episode_end",
        type=int,
        default=0,
        help="0-based exclusive end; 0 = len(episodes).",
    )
    p.add_argument(
        "--shard_tag",
        type=str,
        default="",
        help="Optional output suffix for shard CSVs (e.g. shard03).",
    )
    return p.parse_args()


def _window_rng(seed: int, person_id: str, bin_j: int) -> np.random.Generator:
    """Deterministic per-window RNG so multi-GPU shards match a single-process run."""
    import hashlib

    digest = hashlib.md5(f"{seed}:{person_id}:{bin_j}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**63 - 1))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import sys

    sys.path.insert(0, str(ROOT / "code/model2"))
    from common.utils import resolve_torch_device
    from event_stream.dataset import load_episode_jsonl
    from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
    from common.med_exposure import infer_num_med_exposure

    device = resolve_torch_device(args.device)
    print(f"device={device}", flush=True)

    episodes = load_episode_jsonl(TEST_JSONL)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    ep_end = args.episode_end if args.episode_end > 0 else len(episodes)
    ep_start = max(0, int(args.episode_start))
    ep_end = min(len(episodes), int(ep_end))
    episodes_shard = episodes[ep_start:ep_end]
    print(
        f"episodes_total={len(episodes)} shard=[{ep_start},{ep_end}) n={len(episodes_shard)}",
        flush=True,
    )

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(str(ES_CKPT))
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes_shard or episodes[:32])
    es_model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    es_model.load_state_dict(ckpt["model_state_dict"], strict=False)
    es_model.to(device)
    es_model.eval()

    ts_model = None
    ts_layout = None
    ts_lab = None
    if not args.skip_ts:
        from timeseries.checkpoint import load_bucket_checkpoint

        ts_model, _, _, ts_layout = load_bucket_checkpoint(TS_CKPT, device)
        if TS_LAYOUT.exists():
            ts_layout = json.loads(TS_LAYOUT.read_text())
        ts_lab = lab_stats or {"per_code": {}, "global_mean": 0.0, "global_std": 1.0, "version": 1}
        lab_p = ROOT / "data/five_folds_ts6h4/vocab/lab_zscore_stats.json"
        if lab_p.exists():
            ts_lab = json.loads(lab_p.read_text())

    es_rows: List[Dict[str, Any]] = []
    ts_rows: List[Dict[str, Any]] = []
    n_windows = 0
    tag = f"_{args.shard_tag}" if args.shard_tag else ""

    for ei, ep in enumerate(episodes_shard, 1):
        if ei % 200 == 0 or ei == 1:
            print(
                f"  {ei}/{len(episodes_shard)} (global {ep_start + ei - 1})  "
                f"windows={n_windows}  es_rows={len(es_rows)}",
                flush=True,
            )
        cands = find_candidate_bins(ep, min_k=args.min_k, min_distinct=args.min_distinct)
        if not cands:
            continue
        cands = sorted(cands, key=lambda c: (c["n_distinct"], c["k"]), reverse=True)
        cands = cands[: args.max_windows_per_episode]
        for cand in cands:
            if args.max_windows > 0 and n_windows >= args.max_windows:
                break
            pid = str(ep.get("person_id"))
            rng_w = _window_rng(args.seed, pid, int(cand["bin_j"]))
            es_rows.extend(
                run_es_window(
                    es_model, ep, cand, code_vocab, lab_stats, num_med, device, rng_w,
                    n_random=args.n_random, age_stats=age_stats,
                )
            )
            if ts_model is not None:
                rng_ts = _window_rng(args.seed + 17, pid, int(cand["bin_j"]))
                ts_rows.extend(
                    run_ts_window(ts_model, ep, ts_layout, ts_lab, cand, device, rng_ts)
                )
            n_windows += 1
        if args.max_windows > 0 and n_windows >= args.max_windows:
            break

    import pandas as pd

    es_df = pd.DataFrame(es_rows)
    ts_df = pd.DataFrame(ts_rows)
    es_path = out_dir / f"order_sensitivity_es_rows{tag}.csv"
    ts_path = out_dir / f"order_sensitivity_ts_rows{tag}.csv"
    es_df.to_csv(es_path, index=False)
    ts_df.to_csv(ts_path, index=False)

    abs_dz = es_df["abs_delta_logit"].to_numpy(dtype=float) if len(es_df) else np.array([])
    es_sum = _summarize(abs_dz)
    if len(abs_dz) and not args.shard_tag:
        plot_es_hist(abs_dz, out_dir / "order_sensitivity_es_hist.png")

    ts_sum: Dict[str, Any] = {}
    if len(ts_df):
        ts_sum = {
            "n": int(len(ts_df)),
            "max_feat_l2": float(ts_df["feat_l2"].max()),
            "mean_abs_dz": float(ts_df["abs_delta_logit"].mean()),
            "max_abs_dz": float(ts_df["abs_delta_logit"].max()),
        }

    summary = {
        "n_episodes": len(episodes_shard),
        "episode_start": ep_start,
        "episode_end": ep_end,
        "n_windows": n_windows,
        "n_random": args.n_random,
        "event_stream": es_sum,
        "timeseries": ts_sum,
        "paths": {"es_rows": str(es_path), "ts_rows": str(ts_path)},
    }
    (out_dir / f"summary{tag}.json").write_text(json.dumps(summary, indent=2))

    if not args.shard_tag:
        smoke = args.max_episodes > 0 and args.max_episodes <= 500
        write_report(
            out_dir / "REPORT.md",
            smoke=smoke,
            n_episodes=len(episodes_shard),
            n_windows=n_windows,
            es_sum=es_sum,
            ts_sum=ts_sum,
            args=args,
        )
        print(f"wrote {out_dir / 'REPORT.md'}", flush=True)
    print(json.dumps(summary, indent=2))
    print(f"wrote {es_path}", flush=True)


if __name__ == "__main__":
    main()
