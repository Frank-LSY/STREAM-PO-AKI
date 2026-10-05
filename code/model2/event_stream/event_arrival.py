#!/usr/bin/env python3
"""Order-robust event-arrival attribution along the online (streaming) head.

For each postoperative token i with a unique timestamp:

    a_i = Δz_i = z_i - z_{i-1}

For a simultaneous block G_t = {e_1,…,e_k}, k>1 (exact equal ``time_hour``):

    φ̂_j = (1/R) Σ_r [ z(P_j^{π_r} ∪ {e_j}) - z(P_j^{π_r}) ]

Monte Carlo (or full enumeration when k! is tiny) averages marginal logit updates
over within-block permutations, so attributions do not depend on arbitrary EHR
row order. Efficiency:

    Σ_j φ̂_j = E_π[z_after^{(π)} - z_before]

Code-level pooling aggregates a_i (Δz or φ̂) by code as before.
"""
from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from common.utils import _forward_model
from event_stream.dataset import collate_fn
from event_stream.event_attribution import (
    episode_to_batch,
    forward_logits,
    logit_to_prob,
)
from event_stream.input_builder import episode_to_sequence_features


def _time_bin(time_hour: float) -> str:
    if time_hour < 0:
        return "preop"
    if time_hour < 6:
        return "postop_0_6h"
    if time_hour < 24:
        return "postop_6_24h"
    if time_hour < 72:
        return "postop_24_72h"
    if time_hour < 168:
        return "postop_72h_7d"
    return "postop_7d_plus"


def _sigmoid_deriv(z: float) -> float:
    p = float(logit_to_prob(z))
    return p * (1.0 - p)


def _event_index_for_seq_pos(seq_pos: int, base_idx: int) -> int:
    """Map sequence index → index into ``episode['events']`` (BASE inserted at ``base_idx``)."""
    if seq_pos == base_idx:
        raise ValueError("BASE has no event index")
    return seq_pos if seq_pos < base_idx else seq_pos - 1


def _permute_event_indices(
    events: List[Dict[str, Any]],
    indices: Sequence[int],
    order: Sequence[int],
) -> List[Dict[str, Any]]:
    """Reorder ``events[indices]`` by local permutation ``order`` (whole event dicts).

    Same-timestamp ties: times are already equal, so this only changes row/causal order.
    """
    out = list(events)
    block = [copy.deepcopy(events[i]) for i in indices]
    for loc, src in enumerate(order):
        out[indices[loc]] = block[src]
    return out


def _tie_groups_from_postop(
    seq: List[Dict[str, Any]],
    postop_idx: Sequence[int],
) -> List[List[int]]:
    """Group consecutive postop seq positions sharing exact ``time_hour``."""
    groups: List[List[int]] = []
    cur: List[int] = []
    cur_t: Optional[float] = None
    for pos in postop_idx:
        th = float(seq[pos].get("time_hour", 0.0))
        if cur and th == cur_t:
            cur.append(int(pos))
        else:
            if cur:
                groups.append(cur)
            cur = [int(pos)]
            cur_t = th
    if cur:
        groups.append(cur)
    return groups


def _all_or_sample_perms(k: int, n_perm: int, rng: np.random.Generator) -> List[List[int]]:
    """Enumerate all k! orders when k! ≤ n_perm; otherwise sample ``n_perm`` perms.

    So for n_perm=20: k=2→2, k=3→6, k≥4→20 (not 4!=24).
    """
    import itertools
    import math

    if k <= 1:
        return [list(range(k))]
    n_all = int(math.factorial(k))
    if n_all <= max(int(n_perm), 1):
        return [list(p) for p in itertools.permutations(range(k))]
    seen = set()
    out: List[List[int]] = []
    # Always include identity
    ident = list(range(k))
    seen.add(tuple(ident))
    out.append(ident)
    tries = 0
    while len(out) < n_perm and tries < n_perm * 40:
        tries += 1
        perm = list(rng.permutation(k))
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        out.append(perm)
    return out


def _window_rng(seed: int, person_id: str, time_hour: float) -> np.random.Generator:
    digest = hashlib.md5(f"{seed}:{person_id}:{time_hour:.10g}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:16], 16) % (2**63 - 1))


@torch.no_grad()
def _forward_postop_logits(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    age_stats: Optional[Any] = None,
) -> Tuple[float, int, List[int], List[float], List[Dict[str, Any]], int]:
    batch, seq, label = episode_to_batch(
        episode, code_vocab, lab_stats, num_med_exposure, device, age_stats=age_stats
    )
    outputs = _forward_model(model, batch)
    att = batch["attention_mask"][0].cpu().numpy() > 0
    valid = np.nonzero(att)[0]
    if valid.size == 0:
        return 0.0, 0, [], [], seq, int(label)
    bm = batch["base_mask"][0].cpu().numpy() > 0
    base_hits = valid[bm[valid]]
    base_idx = int(base_hits[0]) if base_hits.size else int(valid[0])
    z_base = float(outputs["baseline_logits_all"][0, base_idx].item())
    pm = (batch["postop_mask"][0].cpu().numpy() > 0) & att
    postop_idx = [int(i) for i in np.nonzero(pm)[0]]
    tl = outputs["token_logits"][0].float().cpu().numpy()
    z_postop = [float(tl[i]) for i in postop_idx]
    return float(z_base), int(base_idx), list(postop_idx), [float(z) for z in z_postop], seq, int(label)


@torch.no_grad()
def _batched_postop_logits(
    model: torch.nn.Module,
    episodes: Sequence[Dict[str, Any]],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    age_stats: Optional[Any] = None,
    max_tokens: int = 32768,
) -> List[Tuple[List[int], List[float], List[Dict[str, Any]]]]:
    """(postop_idx, z_postop, seq) per episode, with padded batched forwards."""
    feats = [
        episode_to_sequence_features(
            ep, code_vocab, add_base=True, lab_stats=lab_stats,
            age_stats=age_stats, num_med_exposure=num_med_exposure,
        )
        for ep in episodes
    ]
    out: List[Any] = [None] * len(feats)
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["sequence"]))
    chunks: List[List[int]] = []
    cur: List[int] = []
    for i in order:
        L = len(feats[i]["sequence"])
        if cur and (len(cur) + 1) * L > max_tokens:
            chunks.append(cur)
            cur = []
        cur.append(i)
    if cur:
        chunks.append(cur)
    for idx in chunks:
        batch = collate_fn([feats[i] for i in idx])
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = _forward_model(model, batch)
        tl = outputs["token_logits"].float().cpu().numpy()
        pm = ((batch["postop_mask"] > 0) & (batch["attention_mask"] > 0)).cpu().numpy()
        for b, i in enumerate(idx):
            pidx = np.nonzero(pm[b])[0]
            out[i] = ([int(p) for p in pidx], [float(z) for z in tl[b, pidx]], feats[i]["sequence"])
    return out


@torch.no_grad()
def event_arrivals_for_episode(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    *,
    n_tie_perm: int = 20,
    seed: int = 0,
    age_stats: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """One row per postoperative token with order-robust arrival attribution ``a_i``.

    ``delta_logit`` stores a_i (sequential Δz for unique timestamps; Monte Carlo φ̂ for ties).
    """
    z_base, base_idx, postop_idx, z_postop, seq, label = _forward_postop_logits(
        model, episode, code_vocab, lab_stats, num_med_exposure, device, age_stats
    )
    if not postop_idx:
        return []

    person_id = str(episode.get("person_id", ""))
    groups = _tie_groups_from_postop(seq, postop_idx)
    # Map seq pos → index into z_postop / step
    pos_to_k = {int(p): i for i, p in enumerate(postop_idx)}

    # Attribution a_i aligned with postop step index
    a = np.zeros(len(postop_idx), dtype=float)
    mode = ["sequential"] * len(postop_idx)
    tie_size = [1] * len(postop_idx)
    block_dz = [0.0] * len(postop_idx)
    n_perm_used = [0] * len(postop_idx)

    # Sequential path logits for unique stamps / display
    z_path = list(z_postop)
    prev_online = float(z_base)

    jobs: List[Tuple[int, List[int]]] = []
    job_eps: List[Dict[str, Any]] = []
    for gi, group in enumerate(groups):
        if len(group) == 1:
            continue
        event_idxs = [_event_index_for_seq_pos(p, base_idx) for p in group]
        if event_idxs != list(range(min(event_idxs), max(event_idxs) + 1)):
            continue
        th = float(seq[group[0]].get("time_hour", 0.0))
        rng = _window_rng(seed, person_id, th)
        for order in _all_or_sample_perms(len(group), n_tie_perm, rng):
            ep_p = dict(episode)
            ep_p["events"] = _permute_event_indices(episode["events"], event_idxs, order)
            jobs.append((gi, order))
            job_eps.append(ep_p)
    job_out = _batched_postop_logits(
        model, job_eps, code_vocab, lab_stats, num_med_exposure, device, age_stats
    ) if job_eps else []
    perm_results: Dict[int, List[Tuple[List[int], Tuple[List[int], List[float], List[Dict[str, Any]]]]]] = defaultdict(list)
    for (gi, order), res in zip(jobs, job_out):
        perm_results[gi].append((order, res))

    for gi, group in enumerate(groups):
        ks = [pos_to_k[p] for p in group]
        z_before = prev_online
        if len(group) == 1:
            k0 = ks[0]
            a[k0] = float(z_path[k0]) - z_before
            mode[k0] = "sequential"
            tie_size[k0] = 1
            block_dz[k0] = float(a[k0])
            n_perm_used[k0] = 0
            prev_online = float(z_path[k0])
            continue

        # Tied block: Monte Carlo / enumerated Shapley-style marginals
        event_idxs = [_event_index_for_seq_pos(p, base_idx) for p in group]
        # Contiguity check — required for simple block permute
        if event_idxs != list(range(min(event_idxs), max(event_idxs) + 1)):
            # Fallback: keep sequential (rare); still mark
            for k0 in ks:
                a[k0] = float(z_path[k0]) - (float(z_path[k0 - 1]) if k0 > 0 else z_before)
                # fix chain for first in group
            a[ks[0]] = float(z_path[ks[0]]) - z_before
            for j in range(1, len(ks)):
                a[ks[j]] = float(z_path[ks[j]]) - float(z_path[ks[j - 1]])
            for k0 in ks:
                mode[k0] = "sequential_noncontiguous"
                tie_size[k0] = len(group)
                block_dz[k0] = float(sum(a[k] for k in ks))
            prev_online = float(z_path[ks[-1]])
            continue

        th = float(seq[group[0]].get("time_hour", 0.0))
        perms = perm_results[gi]
        phi_acc = np.zeros(len(group), dtype=float)
        z_after_acc = 0.0

        for order, (pidx_p, z_p, seq_p) in perms:
            # Block occupies the same seq index range as ``group`` when only that
            # contiguous event block is reordered (BASE insert position unchanged).
            # Match by time_hour + codes in order along postop stream.
            # Safer: take logits at the original seq positions of the block — after
            # permute, those positions hold the permuted events in arrival order.
            pos_set = set(group)
            block_positions = [p for p in pidx_p if p in pos_set]
            if len(block_positions) != len(group):
                # Fallback: consecutive postop tokens with this timestamp
                block_positions = []
                for p in pidx_p:
                    if abs(float(seq_p[p].get("time_hour", 0.0)) - th) < 1e-12:
                        block_positions.append(p)
                # keep a contiguous run of correct length near original
                if len(block_positions) >= len(group):
                    # pick run overlapping original span
                    lo, hi = min(group), max(group)
                    block_positions = [p for p in block_positions if lo <= p <= hi][: len(group)]
            if len(block_positions) != len(group):
                continue

            z_block = []
            for p in block_positions:
                # z_p aligned with pidx_p
                zi = pidx_p.index(p)
                z_block.append(float(z_p[zi]))

            # Map block position → which original local event (by matching event content)
            # After permute with ``order``, local slot loc holds original local index order[loc].
            # Marginal for original event j: when it appears at some slot in this perm.
            prev_z = z_before
            # order[loc] = source local index placed at loc
            for loc, src_local in enumerate(order):
                z_here = z_block[loc]
                phi_acc[src_local] += z_here - prev_z
                prev_z = z_here
            z_after_acc += prev_z

        n_ok = len(perms)
        if n_ok <= 0:
            # degenerate fallback
            for j, k0 in enumerate(ks):
                a[k0] = float(z_path[k0]) - (z_before if j == 0 else float(z_path[ks[j - 1]]))
                mode[k0] = "sequential_fallback"
                tie_size[k0] = len(group)
            prev_online = float(z_path[ks[-1]])
            continue

        phi = phi_acc / float(n_ok)
        z_after_mean = z_after_acc / float(n_ok)
        for j, k0 in enumerate(ks):
            a[k0] = float(phi[j])
            mode[k0] = "shapley_mc"
            tie_size[k0] = len(group)
            block_dz[k0] = float(z_after_mean - z_before)
            n_perm_used[k0] = int(n_ok)
        prev_online = float(z_after_mean)

    # Per-group bounds for display (efficiency: sum φ = E[z_after]-z_before).
    group_bounds: Dict[int, Tuple[float, float]] = {}
    z_cursor = float(z_base)
    for gi, group in enumerate(groups):
        ks = [pos_to_k[p] for p in group]
        z_before = z_cursor
        if mode[ks[0]] == "shapley_mc":
            z_after = z_before + float(sum(a[k] for k in ks))
        else:
            z_after = float(z_path[ks[-1]])
        group_bounds[gi] = (z_before, z_after)
        z_cursor = z_after

    # Build rows with clean display fields.
    rows: List[Dict[str, Any]] = []
    for gi, group in enumerate(groups):
        ks = [pos_to_k[p] for p in group]
        z_before_g, z_after_g = group_bounds[gi]
        p_before = float(logit_to_prob(z_before_g))
        p_after = float(logit_to_prob(z_after_g))
        is_shapley = mode[ks[0]] == "shapley_mc"
        if is_shapley:
            scale = 0.5 * (_sigmoid_deriv(z_before_g) + _sigmoid_deriv(z_after_g))
            for k0 in ks:
                pos = postop_idx[k0]
                tok = seq[pos]
                dz = float(a[k0])
                dp = float(dz * scale)
                rows.append(
                    {
                        "person_id": person_id,
                        "label_aki": int(label),
                        "step": int(k0),
                        "seq_index": int(pos),
                        "prev_seq_index": int(postop_idx[k0 - 1]) if k0 > 0 else int(base_idx),
                        "modality": str(tok.get("modality", "")),
                        "raw_code": str(tok.get("raw_code", "")),
                        "time_hour": float(tok.get("time_hour", 0.0)),
                        "time_bin": _time_bin(float(tok.get("time_hour", 0.0))),
                        "logit": float(z_after_g),
                        "prev_logit": float(z_before_g),
                        "delta_logit": dz,
                        "prob": float(p_before + dp),
                        "prev_prob": p_before,
                        "delta_prob": dp,
                        "baseline_logit": float(z_base),
                        "attribution_mode": mode[k0],
                        "tie_size": int(tie_size[k0]),
                        "block_delta_logit": float(block_dz[k0]),
                        "n_perm_used": int(n_perm_used[k0]),
                    }
                )
        else:
            prev_z = z_before_g
            for k0 in ks:
                pos = postop_idx[k0]
                tok = seq[pos]
                z_here = float(z_path[k0])
                dz = float(a[k0])
                p_prev = float(logit_to_prob(prev_z))
                p_new = float(logit_to_prob(z_here))
                rows.append(
                    {
                        "person_id": person_id,
                        "label_aki": int(label),
                        "step": int(k0),
                        "seq_index": int(pos),
                        "prev_seq_index": int(postop_idx[k0 - 1]) if k0 > 0 else int(base_idx),
                        "modality": str(tok.get("modality", "")),
                        "raw_code": str(tok.get("raw_code", "")),
                        "time_hour": float(tok.get("time_hour", 0.0)),
                        "time_bin": _time_bin(float(tok.get("time_hour", 0.0))),
                        "logit": z_here,
                        "prev_logit": float(prev_z),
                        "delta_logit": dz,
                        "prob": p_new,
                        "prev_prob": p_prev,
                        "delta_prob": float(p_new - p_prev),
                        "baseline_logit": float(z_base),
                        "attribution_mode": mode[k0],
                        "tie_size": int(tie_size[k0]),
                        "block_delta_logit": float(block_dz[k0]),
                        "n_perm_used": int(n_perm_used[k0]),
                    }
                )
                prev_z = z_here

    rows.sort(key=lambda r: int(r["step"]))
    return rows


def pool_arrival_by_code(
    rows: List[Dict[str, Any]],
    min_n: int = 25,
) -> List[Dict[str, Any]]:
    """Pool arrival a_i (Δz or φ̂) by (modality, raw_code).

    One value per (person, code) = mean over that patient's arrivals of the code.
    """
    by_pc: Dict[tuple, List[float]] = {}
    for r in rows:
        key = (str(r["person_id"]), str(r["modality"]), str(r["raw_code"]))
        by_pc.setdefault(key, []).append(float(r["delta_logit"]))

    by_code: Dict[tuple, List[float]] = {}
    for (pid, mod, code), deltas in by_pc.items():
        ck = (mod, code)
        by_code.setdefault(ck, []).append(float(np.mean(deltas)))

    out: List[Dict[str, Any]] = []
    for (mod, code), vals in by_code.items():
        if len(vals) < min_n:
            continue
        arr = np.asarray(vals, dtype=float)
        out.append(
            {
                "modality": mod,
                "raw_code": code,
                "n_patients": int(len(arr)),
                "mean_delta_logit": float(arr.mean()),
                "median_delta_logit": float(np.median(arr)),
                "std_delta_logit": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "mean_abs_delta_logit": float(np.abs(arr).mean()),
            }
        )
    out.sort(key=lambda r: -abs(r["mean_delta_logit"]))
    return out
