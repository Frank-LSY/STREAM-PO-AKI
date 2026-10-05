"""Model-attributed event contribution via occlusion (logit difference).

This measures how masking an event changes model logits. It is **not** a causal effect.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from event_stream.dataset import collate_fn
from event_stream.input_builder import PAD_TOKEN, episode_to_sequence_features
from common.utils import _forward_model


def time_window_label(time_day: float, time_hour: float) -> str:
    if time_day <= 0.0:
        return "preoperative"
    if time_hour <= 24.0:
        return "postop_0_24h"
    if time_hour <= 72.0:
        return "postop_24_72h"
    return "postop_72h_plus"


@dataclass
class EventRef:
    seq_index: int
    raw_code: str
    modality: str
    time_day: float
    time_hour: float
    time_window: str
    med_exposure_idx: int = -1


def build_event_refs(seq: List[Dict[str, Any]]) -> List[EventRef]:
    refs: List[EventRef] = []
    for i, tok in enumerate(seq):
        if tok.get("is_base", 0) == 1:
            continue
        refs.append(
            EventRef(
                seq_index=i,
                raw_code=str(tok.get("raw_code", "")),
                modality=str(tok.get("modality", "")),
                time_day=float(tok.get("time_day", 0.0)),
                time_hour=float(tok.get("time_hour", 0.0)),
                time_window=time_window_label(
                    float(tok.get("time_day", 0.0)), float(tok.get("time_hour", 0.0))
                ),
                med_exposure_idx=int(tok.get("med_exposure_idx", -1)),
            )
        )
    return refs


def episode_to_batch(
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    age_stats: Optional[Any] = None,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]], int]:
    feat = episode_to_sequence_features(
        episode,
        code_vocab,
        add_base=True,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med_exposure,
    )
    seq = feat["sequence"]
    batch = collate_fn([feat])
    batch = {k: v.to(device) for k, v in batch.items()}
    return batch, seq, int(feat["label"])


def _pad_id(code_vocab: Dict[str, int]) -> int:
    return int(code_vocab.get(PAD_TOKEN, 0))


def occlude_sequence_position(
    batch: Dict[str, torch.Tensor],
    seq: List[Dict[str, Any]],
    pos: int,
    code_vocab: Dict[str, int],
) -> Dict[str, torch.Tensor]:
    """Remove one event's information (code, lab value, med exposure column)."""
    out = {k: v.clone() for k, v in batch.items()}
    pad = _pad_id(code_vocab)
    meta = seq[pos]
    mod = str(meta.get("modality", ""))

    out["code_ids"][0, pos] = pad
    out["values"][0, pos] = 0.0
    out["value_masks"][0, pos] = 0.0
    if "age_values" in out:
        out["age_values"][0, pos] = 0.0
    if "age_masks" in out:
        out["age_masks"][0, pos] = 0.0
    out["time_days"][0, pos] = 0.0
    out["time_hours"][0, pos] = 0.0

    if mod == "lab":
        pass  # code + value already cleared

    if mod == "med" and "med_day_feats" in out:
        idx = int(meta.get("med_exposure_idx", -1))
        if idx >= 0:
            out["med_day_feats"][0, :, idx] = 0.0
            for j, tok in enumerate(seq):
                if int(tok.get("med_exposure_idx", -1)) == idx:
                    out["code_ids"][0, j] = pad
                    out["values"][0, j] = 0.0
                    out["value_masks"][0, j] = 0.0
    return out


@torch.no_grad()
def forward_logits(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
) -> Tuple[float, int, List[int], List[float]]:
    outputs = _forward_model(model, batch)
    valid = (batch["attention_mask"][0] > 0).nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return 0.0, 0, [], []
    base_pos = torch.nonzero(batch["base_mask"][0, valid] > 0, as_tuple=False).flatten()
    if base_pos.numel() == 0:
        base_idx = int(valid[0].item())
    else:
        base_idx = int(valid[base_pos[0]].item())
    z_base = float(outputs["baseline_logits_all"][0, base_idx].item())

    postop_idx: List[int] = []
    z_postop: List[float] = []
    for i in valid.tolist():
        if batch["postop_mask"][0, i].item() > 0:
            postop_idx.append(int(i))
            z_postop.append(float(outputs["token_logits"][0, i].item()))
    return z_base, base_idx, postop_idx, z_postop


def _subsample_indices(indices: List[int], max_n: int) -> List[int]:
    if max_n <= 0 or len(indices) <= max_n:
        return indices
    pick = np.linspace(0, len(indices) - 1, max_n, dtype=int)
    return [indices[i] for i in pick]


def logit_to_prob(z: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(z))))


def delta_prob_from_logits(z: float, z_masked: float) -> float:
    """Change in predicted probability when an event is occluded (present → masked)."""
    return logit_to_prob(z) - logit_to_prob(z_masked)


def probability_contributions_for_episode(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    max_preop_occlusions: int = 80,
    max_postop_query_steps: int = 40,
    max_events_per_query: int = 80,
    lookback_hours: float = 72.0,
) -> List[Dict[str, Any]]:
    """Flat per-occlusion records with delta_prob (and delta_logit) for cohort pooling.

    Positive delta_prob: masking lowered risk → the event is associated with higher risk
    when present (model attribution, not causal).
    """
    batch, seq, label = episode_to_batch(episode, code_vocab, lab_stats, num_med_exposure, device)
    z_base, _, postop_idx, z_postop = forward_logits(model, batch)
    refs = build_event_refs(seq)
    ref_by_pos = {r.seq_index: r for r in refs}
    person_id = str(episode.get("person_id", ""))
    records: List[Dict[str, Any]] = []

    def _row(
        scope: str,
        pos: int,
        z: float,
        z_m: float,
        query_seq_index: Optional[int] = None,
        query_time_hour: Optional[float] = None,
    ) -> Dict[str, Any]:
        r = ref_by_pos[pos]
        dl = float(z - z_m)
        dp = delta_prob_from_logits(z, z_m)
        return {
            "person_id": person_id,
            "label_aki": int(label),
            "scope": scope,
            "raw_code": r.raw_code,
            "modality": r.modality,
            "event_seq_index": pos,
            "query_seq_index": query_seq_index,
            "query_time_hour": query_time_hour,
            "delta_logit": dl,
            "delta_prob": dp,
            "prob_with_event": logit_to_prob(z_m),
            "prob_without_event": logit_to_prob(z),
        }

    preop_positions = [r.seq_index for r in refs if r.time_window == "preoperative"]
    preop_positions = _subsample_indices(preop_positions, max_preop_occlusions)
    for pos in preop_positions:
        batch_m = occlude_sequence_position(batch, seq, pos, code_vocab)
        z_m, _, _, _ = forward_logits(model, batch_m)
        records.append(_row("baseline", pos, z_base, z_m))

    query_positions = _subsample_indices(postop_idx, max_postop_query_steps)
    for t_pos in query_positions:
        z_t = z_postop[postop_idx.index(t_pos)]
        t_hour = float(seq[t_pos]["time_hour"])
        influencers = [
            r.seq_index
            for r in refs
            if r.time_day <= 0.0 or (r.seq_index <= t_pos and r.time_hour >= t_hour - lookback_hours)
        ]
        influencers = _subsample_indices([p for p in influencers if p != t_pos], max_events_per_query)
        for pos in influencers:
            batch_m = occlude_sequence_position(batch, seq, pos, code_vocab)
            _, _, postop_m, z_m_list = forward_logits(model, batch_m)
            if t_pos not in postop_m:
                continue
            z_tm = z_m_list[postop_m.index(t_pos)]
            records.append(_row("online", pos, z_t, z_tm, query_seq_index=t_pos, query_time_hour=t_hour))

    return records


def _event_group_key(modality: str, raw_code: str) -> str:
    return f"{modality}|{raw_code}"


def aggregate_time_agnostic_probability_contributions(
    records: Sequence[Dict[str, Any]],
    *,
    min_count: int = 10,
    top_k: int = 50,
    label_aki: Optional[int] = None,
) -> Dict[str, Any]:
    """Pool occlusion Δprob by (modality, raw_code), ignoring time windows."""
    pools: Dict[str, List[float]] = defaultdict(list)
    pool_logit: Dict[str, List[float]] = defaultdict(list)
    meta: Dict[str, Dict[str, str]] = {}

    for row in records:
        if label_aki is not None and int(row.get("label_aki", 0)) != label_aki:
            continue
        key = _event_group_key(str(row.get("modality", "")), str(row.get("raw_code", "")))
        if not row.get("raw_code"):
            continue
        pools[key].append(float(row["delta_prob"]))
        pool_logit[key].append(float(row["delta_logit"]))
        meta[key] = {"modality": str(row["modality"]), "raw_code": str(row["raw_code"])}

    ranked: List[Dict[str, Any]] = []
    for key, vals in pools.items():
        if len(vals) < min_count:
            continue
        a = np.asarray(vals, dtype=np.float64)
        al = np.asarray(pool_logit[key], dtype=np.float64)
        ranked.append(
            {
                **meta[key],
                "event_key": key,
                "n_measurements": int(len(a)),
                "mean_delta_prob": float(a.mean()),
                "median_delta_prob": float(np.median(a)),
                "std_delta_prob": float(a.std()) if len(a) > 1 else 0.0,
                "sum_delta_prob": float(a.sum()),
                "mean_delta_logit": float(al.mean()),
                "fraction_positive": float(np.mean(a > 0)),
            }
        )

    pos = sorted(
        [r for r in ranked if r["mean_delta_prob"] > 0],
        key=lambda x: x["mean_delta_prob"],
        reverse=True,
    )[:top_k]
    neg = sorted(
        [r for r in ranked if r["mean_delta_prob"] < 0],
        key=lambda x: x["mean_delta_prob"],
    )[:top_k]
    return {
        "label_filter": label_aki,
        "min_count": min_count,
        "n_event_types_ranked": len(ranked),
        "n_measurements_total": int(
            len(records)
            if label_aki is None
            else sum(1 for r in records if int(r.get("label_aki", 0)) == label_aki)
        ),
        "risk_increasing_events": pos,
        "risk_decreasing_events": neg,
        "all_ranked_events": sorted(ranked, key=lambda x: abs(x["mean_delta_prob"]), reverse=True),
    }


def attributions_for_episode(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    max_preop_occlusions: int = 80,
    max_postop_query_steps: int = 40,
    lookback_hours: float = 72.0,
) -> Dict[str, Any]:
    batch, seq, label = episode_to_batch(episode, code_vocab, lab_stats, num_med_exposure, device)
    z_base, base_idx, postop_idx, z_postop = forward_logits(model, batch)
    refs = build_event_refs(seq)
    ref_by_pos = {r.seq_index: r for r in refs}

    preop_positions = [
        r.seq_index
        for r in refs
        if r.time_window == "preoperative"
    ]
    preop_positions = _subsample_indices(preop_positions, max_preop_occlusions)

    baseline_attr: List[Dict[str, Any]] = []
    for pos in preop_positions:
        batch_m = occlude_sequence_position(batch, seq, pos, code_vocab)
        z_m, _, _, _ = forward_logits(model, batch_m)
        r = ref_by_pos[pos]
        baseline_attr.append(
            {
                "seq_index": pos,
                "raw_code": r.raw_code,
                "modality": r.modality,
                "time_window": r.time_window,
                "delta_logit": z_base - z_m,
            }
        )

    query_positions = _subsample_indices(postop_idx, max_postop_query_steps)
    online_attr: List[Dict[str, Any]] = []
    risk_updates: List[Dict[str, Any]] = []

    for qi, t_pos in enumerate(query_positions):
        z_t = z_postop[postop_idx.index(t_pos)]
        t_hour = float(seq[t_pos]["time_hour"])
        influencers = [
            r.seq_index
            for r in refs
            if r.time_day <= 0.0 or (r.seq_index <= t_pos and r.time_hour >= t_hour - lookback_hours)
        ]
        influencers = [p for p in influencers if p != t_pos][:80]

        for pos in influencers:
            batch_m = occlude_sequence_position(batch, seq, pos, code_vocab)
            _, _, postop_m, z_m_list = forward_logits(model, batch_m)
            if t_pos not in postop_m:
                continue
            z_tm = z_m_list[postop_m.index(t_pos)]
            r = ref_by_pos[pos]
            online_attr.append(
                {
                    "query_seq_index": t_pos,
                    "query_time_hour": t_hour,
                    "event_seq_index": pos,
                    "raw_code": r.raw_code,
                    "modality": r.modality,
                    "time_window": r.time_window,
                    "delta_logit": z_t - z_tm,
                }
            )

        if qi > 0:
            t_prev = query_positions[qi - 1]
            z_prev = z_postop[postop_idx.index(t_prev)]
            delta_update = z_t - z_prev
            new_event_positions = [t_pos]
            for r in refs:
                if r.seq_index == t_pos:
                    continue
                if abs(r.time_hour - t_hour) < 1e-6 and r.seq_index > t_prev and r.seq_index <= t_pos:
                    new_event_positions.append(r.seq_index)
            for pos in sorted(set(new_event_positions)):
                batch_m = occlude_sequence_position(batch, seq, pos, code_vocab)
                _, _, postop_m, z_m_list = forward_logits(model, batch_m)
                z_tm = z_m_list[postop_m.index(t_pos)] if t_pos in postop_m else z_t
                r = ref_by_pos.get(pos)
                risk_updates.append(
                    {
                        "query_seq_index": t_pos,
                        "event_seq_index": pos,
                        "time_hour": t_hour,
                        "delta_logit_step": delta_update,
                        "delta_logit_new_event": z_t - z_tm,
                        "raw_code": r.raw_code if r else "",
                        "modality": r.modality if r else "",
                        "time_window": r.time_window if r else "",
                    }
                )

    return {
        "person_id": episode.get("person_id"),
        "label_aki": label,
        "z_baseline": z_base,
        "baseline_attribution": baseline_attr,
        "online_attribution": online_attr,
        "risk_update_attribution": risk_updates,
        "n_postop_steps": len(postop_idx),
        "disclaimer": "Model-attributed contribution (occlusion logit difference); not causal.",
    }


def cohort_top_codes(
    episodes_attr: Sequence[Dict[str, Any]],
    attr_key: str,
    label_aki: Optional[int],
    top_k: int = 30,
) -> Dict[str, List[Dict[str, Any]]]:
    by_code: Dict[str, List[float]] = defaultdict(list)
    for ep in episodes_attr:
        if label_aki is not None and int(ep.get("label_aki", 0)) != label_aki:
            continue
        for row in ep.get(attr_key, []):
            code = str(row.get("raw_code", ""))
            if code:
                by_code[code].append(float(row["delta_logit"]))
    ranked = []
    for code, vals in by_code.items():
        a = np.asarray(vals, dtype=np.float64)
        ranked.append(
            {
                "raw_code": code,
                "mean_delta_logit": float(a.mean()),
                "sum_delta_logit": float(a.sum()),
                "n": int(len(a)),
            }
        )
    pos = sorted([r for r in ranked if r["mean_delta_logit"] > 0], key=lambda x: x["mean_delta_logit"], reverse=True)[
        :top_k
    ]
    neg = sorted([r for r in ranked if r["mean_delta_logit"] < 0], key=lambda x: x["mean_delta_logit"])[:top_k]
    return {"risk_increasing": pos, "risk_decreasing": neg}


def aggregate_cohort_attributions(episodes_attr: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool attribution deltas by modality, code, and time window."""
    pools: Dict[str, Dict[str, List[float]]] = {
        "baseline_by_modality": defaultdict(list),
        "baseline_by_code": defaultdict(list),
        "online_by_modality": defaultdict(list),
        "online_by_code": defaultdict(list),
        "online_by_time_window": defaultdict(list),
        "risk_update_by_modality": defaultdict(list),
    }

    for ep in episodes_attr:
        label = int(ep.get("label_aki", 0))
        for row in ep.get("baseline_attribution", []):
            d = float(row["delta_logit"])
            pools["baseline_by_modality"][row["modality"]].append(d)
            pools["baseline_by_code"][row["raw_code"]].append(d)
        for row in ep.get("online_attribution", []):
            d = float(row["delta_logit"])
            pools["online_by_modality"][row["modality"]].append(d)
            pools["online_by_code"][row["raw_code"]].append(d)
            pools["online_by_time_window"][row["time_window"]].append(d)
        for row in ep.get("risk_update_attribution", []):
            pools["risk_update_by_modality"][row.get("modality", "")].append(
                float(row["delta_logit_step"])
            )

    def _summarize(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {"mean_delta_logit": None, "sum_delta_logit": None, "n": 0}
        a = np.asarray(vals, dtype=np.float64)
        return {
            "mean_delta_logit": float(a.mean()),
            "sum_delta_logit": float(a.sum()),
            "n": int(len(a)),
        }

    out: Dict[str, Any] = {}
    for pool_name, groups in pools.items():
        out[pool_name] = {k: _summarize(v) for k, v in sorted(groups.items(), key=lambda x: -abs(sum(x[1])))}
    out["n_episodes"] = len(episodes_attr)
    out["n_aki_positive"] = int(sum(int(e.get("label_aki", 0)) for e in episodes_attr))
    out["top_codes_aki_positive_online"] = cohort_top_codes(
        episodes_attr, "online_attribution", label_aki=1
    )
    out["top_codes_aki_negative_online"] = cohort_top_codes(
        episodes_attr, "online_attribution", label_aki=0
    )
    out["top_codes_baseline_all"] = cohort_top_codes(episodes_attr, "baseline_attribution", label_aki=None)
    return out


def top_contributors(
    episodes_attr: Sequence[Dict[str, Any]],
    label_aki: Optional[int],
    attr_key: str = "online_attribution",
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for ep in episodes_attr:
        if label_aki is not None and int(ep.get("label_aki", 0)) != label_aki:
            continue
        rows.extend(ep.get(attr_key, []))
    rows.sort(key=lambda r: abs(float(r["delta_logit"])), reverse=True)
    return rows[:top_k]


def select_case_study_episodes(
    episodes_attr: Sequence[Dict[str, Any]],
    n_aki: int = 2,
    n_non: int = 2,
) -> List[Dict[str, Any]]:
    pos = [e for e in episodes_attr if int(e.get("label_aki", 0)) == 1]
    neg = [e for e in episodes_attr if int(e.get("label_aki", 0)) == 0]
    pos.sort(key=lambda e: len(e.get("online_attribution", [])), reverse=True)
    neg.sort(key=lambda e: len(e.get("online_attribution", [])), reverse=True)
    return pos[:n_aki] + neg[:n_non]
