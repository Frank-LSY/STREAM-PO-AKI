"""Group-level event contribution by true deletion, stratified by time.

This replaces the single-token occlusion in ``event_attribution.py`` on three points.

**True deletion.** The occlusion path there pads a token's code and zeroes its time while
leaving ``attention_mask`` untouched, so the position survives as an attendable key sitting at
hour 0 — a relocated ghost rather than a removed event. Here the token is dropped from the
sequence and the batch is rebuilt, so ``collate_fn`` shortens ``attention_mask`` and the event is
genuinely gone. Medications are removed from both places that encode them: the event tokens and
the drug's column in the day-level ``med_day_feats`` of the remaining tokens, restricted to the
days the group covers.

**Groups, not single events.** Repeated measurements make single-event deletion saturate: drop
one of six creatinines and the model barely moves, so the concept looks unimportant. The unit
here is ``(modality, raw_code, time_bin)`` — every occurrence of a code inside one time window
leaves together. This also makes each forward answer the question directly: what does this kind
of event, happening in this period, contribute?

**Delta logit, not delta probability.** Baseline risk and cohort prevalence both fall steeply
across the postoperative horizon, so a fixed logit shift maps to a much smaller probability
shift late than early. Ranking time-stratified contributions by Δprob would manufacture a decay
that is only sigmoid compression. Δlogit is reported first; Δprob is kept for reference within a
single time bin.

Sign convention follows ``event_attribution``: delta = (logit with the group) - (logit without
it), so a positive delta means the events raise predicted risk. These are model attributions
under an off-manifold intervention, not causal effects.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch

from event_stream.dataset import collate_fn
from event_stream.event_attribution import (
    _subsample_indices,
    forward_logits,
    logit_to_prob,
)
from event_stream.input_builder import episode_to_sequence_features
from common.age_zscore import apply_age_zscore
from common.utils import _forward_model

# Postoperative bins align with the 6h evaluation grid so contribution and discrimination can be
# read on the same time axis. Preoperative events split at 48h, which is where event density
# starts climbing toward surgery.
PREOP_FAR_H = -48.0
POSTOP_BIN_EDGES_H: Tuple[float, ...] = (6.0, 24.0, 72.0, 168.0)
POSTOP_BIN_NAMES: Tuple[str, ...] = (
    "postop_0_6h",
    "postop_6_24h",
    "postop_24_72h",
    "postop_72h_7d",
    "postop_7d_plus",
)
LAG_BIN_EDGES_H: Tuple[float, ...] = (6.0, 24.0, 72.0)
LAG_BIN_NAMES: Tuple[str, ...] = ("lag_0_6h", "lag_6_24h", "lag_24_72h", "lag_72h_plus")
LAG_FROM_PREOP = "from_preop"


def event_time_bin(tok: Dict[str, Any]) -> str:
    """Time category of one token, using the pre/post flags fixed relative to [BASE]."""
    if int(tok.get("is_base", 0)) == 1:
        return "base"
    hour = float(tok.get("time_hour", 0.0))
    if int(tok.get("is_preop", 0)) == 1:
        return "preop_7d_48h" if hour <= PREOP_FAR_H else "preop_48h_0h"
    for edge, name in zip(POSTOP_BIN_EDGES_H, POSTOP_BIN_NAMES):
        if hour < edge:
            return name
    return POSTOP_BIN_NAMES[-1]


def lag_bin(lag_hours: float) -> str:
    for edge, name in zip(LAG_BIN_EDGES_H, LAG_BIN_NAMES):
        if lag_hours < edge:
            return name
    return LAG_BIN_NAMES[-1]


def sequence_without(
    feat: Dict[str, Any],
    drop: Set[int],
    med_col: Optional[int] = None,
    med_day_range: Optional[Tuple[float, float]] = None,
) -> Tuple[Dict[str, Any], List[int]]:
    """Drop tokens and, for medications, clear the drug's exposure column over its days.

    Returns the reduced feature dict and the kept original indices, so callers can map a query
    token to its new position.
    """
    seq = feat["sequence"]
    keep = [i for i in range(len(seq)) if i not in drop]
    new_seq: List[Dict[str, Any]] = []
    for i in keep:
        tok = dict(seq[i])
        row = tok.get("med_day_feats")
        if med_col is not None and row:
            day = float(tok.get("time_day", 0.0))
            inside = med_day_range is None or (med_day_range[0] <= day <= med_day_range[1])
            if inside and 0 <= med_col < len(row):
                row = list(row)
                row[med_col] = 0.0
                tok["med_day_feats"] = row
        new_seq.append(tok)
    reduced = {
        "sequence": new_seq,
        "label": feat.get("label"),
        "person_id": feat.get("person_id"),
    }
    return reduced, keep


def enumerate_groups(seq: Sequence[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[int]]:
    """Map (modality, raw_code, time_bin) to the token positions it covers."""
    groups: Dict[Tuple[str, str, str], List[int]] = defaultdict(list)
    for i, tok in enumerate(seq):
        if int(tok.get("is_base", 0)) == 1:
            continue
        code = str(tok.get("raw_code", ""))
        if not code:
            continue
        groups[(str(tok.get("modality", "")), code, event_time_bin(tok))].append(i)
    return dict(groups)


def _med_removal_spec(
    seq: Sequence[Dict[str, Any]], positions: Sequence[int]
) -> Tuple[Optional[int], Optional[Tuple[float, float]]]:
    cols = {int(seq[i].get("med_exposure_idx", -1)) for i in positions}
    cols.discard(-1)
    if len(cols) != 1:
        return None, None
    days = [float(seq[i].get("time_day", 0.0)) for i in positions]
    return cols.pop(), (min(days), max(days))


@torch.no_grad()
def group_contributions_for_episode(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    max_query_steps: int = 40,
    lookback_hours: float = 72.0,
    max_groups: int = 0,
    max_query_hour: float = 0.0,
    age_stats: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """One deletion per (code, time bin) group; score [BASE] and each postoperative query.

    A group is only scored against queries at or after its last event, so nothing is credited
    with influencing a prediction made before it happened. Postoperative groups are additionally
    limited to queries within ``lookback_hours``, matching the window the online head can
    plausibly still be reacting to.

    ``max_query_hour`` caps how late a prediction may be scored. This matters because episodes end
    at the AKI event for positive cases but run the full horizon for negatives, so the risk set
    loses positives steadily while negatives remain. Attributions computed deep into the horizon
    describe a population that is almost entirely negative by construction, and their near-zero
    magnitude reflects that thinning rather than any property of the events.
    """
    feat = episode_to_sequence_features(
        episode,
        code_vocab,
        add_base=True,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med_exposure,
    )
    seq = feat["sequence"]
    if not seq:
        return []

    batch = {k: v.to(device) for k, v in collate_fn([feat]).items()}
    z_base_ref, _, postop_idx, z_postop_ref = forward_logits(model, batch)
    ref_postop = {pos: z for pos, z in zip(postop_idx, z_postop_ref)}
    eligible = postop_idx
    if max_query_hour > 0:
        eligible = [p for p in postop_idx if float(seq[p].get("time_hour", 0.0)) <= max_query_hour]
    query_positions = _subsample_indices(eligible, max_query_steps)

    person_id = str(episode.get("person_id", ""))
    label = int(feat.get("label") or 0)
    proc_pos = find_procedure_position(seq)
    index_procedure = str(seq[proc_pos].get("raw_code", "")) if proc_pos is not None else ""

    groups = enumerate_groups(seq)
    keys = list(groups)
    if max_groups > 0 and len(keys) > max_groups:
        keys = [keys[i] for i in np.linspace(0, len(keys) - 1, max_groups, dtype=int)]

    records: List[Dict[str, Any]] = []
    for key in keys:
        modality, raw_code, tbin = key
        positions = groups[key]
        med_col, day_range = (
            _med_removal_spec(seq, positions) if modality == "med" else (None, None)
        )
        reduced, keep = sequence_without(feat, set(positions), med_col, day_range)
        if not reduced["sequence"]:
            continue
        sub = {k: v.to(device) for k, v in collate_fn([reduced]).items()}
        z_base_m, _, postop_idx_m, z_postop_m = forward_logits(model, sub)
        new_index = {old: new for new, old in enumerate(keep)}
        masked_postop = {pos: z for pos, z in zip(postop_idx_m, z_postop_m)}

        common = {
            "person_id": person_id,
            "label_aki": label,
            "index_procedure": index_procedure,
            "modality": modality,
            "raw_code": raw_code,
            "time_bin": tbin,
            "n_events_in_group": len(positions),
        }

        if tbin.startswith("preop"):
            records.append(
                {
                    **common,
                    "scope": "baseline",
                    "lag_bin": "",
                    "query_time_hour": None,
                    "delta_logit": float(z_base_ref - z_base_m),
                    "delta_prob": float(logit_to_prob(z_base_ref) - logit_to_prob(z_base_m)),
                }
            )

        group_last_hour = max(float(seq[i].get("time_hour", 0.0)) for i in positions)
        is_preop_group = tbin.startswith("preop")
        for t_old in query_positions:
            if t_old in positions:
                continue
            t_hour = float(seq[t_old].get("time_hour", 0.0))
            lag = t_hour - group_last_hour
            if lag < 0:
                continue
            if not is_preop_group and lag > lookback_hours:
                continue
            t_new = new_index.get(t_old)
            if t_new is None or t_new not in masked_postop:
                continue
            z_ref = ref_postop[t_old]
            z_m = masked_postop[t_new]
            records.append(
                {
                    **common,
                    "scope": "online",
                    "lag_bin": LAG_FROM_PREOP if is_preop_group else lag_bin(lag),
                    "query_time_hour": t_hour,
                    "delta_logit": float(z_ref - z_m),
                    "delta_prob": float(logit_to_prob(z_ref) - logit_to_prob(z_m)),
                }
            )
    return records


def aggregate_contributions(
    records: Sequence[Dict[str, Any]],
    by: Sequence[str] = ("modality", "raw_code", "time_bin"),
    *,
    scope: Optional[str] = None,
    label_aki: Optional[int] = None,
    min_count: int = 10,
    min_patients: int = 5,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Pool Δlogit over the requested key, ranked by mean Δlogit.

    ``min_patients`` matters more than ``min_count``: one patient with many query steps can fill a
    cell on its own, so a measurement threshold alone will happily rank a code seen in a handful
    of episodes alongside one seen in hundreds.
    """
    pools: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    probs: Dict[Tuple[str, ...], List[float]] = defaultdict(list)
    people: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    n_used = 0
    for row in records:
        if scope is not None and row.get("scope") != scope:
            continue
        if label_aki is not None and int(row.get("label_aki", 0)) != label_aki:
            continue
        key = tuple(str(row.get(f, "")) for f in by)
        pools[key].append(float(row["delta_logit"]))
        probs[key].append(float(row["delta_prob"]))
        people[key].add(str(row.get("person_id", "")))
        n_used += 1

    ranked: List[Dict[str, Any]] = []
    for key, vals in pools.items():
        if len(vals) < min_count or len(people[key]) < min_patients:
            continue
        a = np.asarray(vals, dtype=np.float64)
        p = np.asarray(probs[key], dtype=np.float64)
        ranked.append(
            {
                **dict(zip(by, key)),
                "n_measurements": int(a.size),
                "n_patients": len(people[key]),
                "mean_delta_logit": float(a.mean()),
                "median_delta_logit": float(np.median(a)),
                "std_delta_logit": float(a.std()) if a.size > 1 else 0.0,
                "mean_delta_prob": float(p.mean()),
                "fraction_positive": float(np.mean(a > 0)),
            }
        )

    ranked.sort(key=lambda r: r["mean_delta_logit"], reverse=True)
    return {
        "group_by": list(by),
        "scope": scope,
        "label_filter": label_aki,
        "min_count": min_count,
        "min_patients": min_patients,
        "n_measurements_used": n_used,
        "n_groups_ranked": len(ranked),
        "risk_increasing": ranked[:top_k],
        "risk_decreasing": list(reversed(ranked[-top_k:])) if ranked else [],
        "all_ranked": ranked,
    }


def compare_across_strata(
    records: Sequence[Dict[str, Any]],
    stratum_field: str = "procedure_region",
    by: Sequence[str] = ("modality", "raw_code", "time_bin"),
    *,
    scope: str = "online",
    min_count: int = 10,
    min_patients: int = 5,
    min_strata: int = 2,
) -> List[Dict[str, Any]]:
    """Split each contribution by surgical stratum to separate real signal from case mix.

    Index procedure is by far the strongest determinant of baseline risk here, and it is present
    in every episode, so a contribution pooled over the whole cohort can be driven entirely by
    which operations happen to carry that code. A contribution that keeps its sign and rough size
    in every stratum is a property of the event; one that appears in a single stratum is case mix.
    """
    pools: Dict[Tuple[str, ...], Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    seen: Dict[Tuple[str, ...], Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in records:
        if row.get("scope") != scope:
            continue
        key = tuple(str(row.get(f, "")) for f in by)
        stratum = str(row.get(stratum_field, ""))
        pools[key][stratum].append(float(row["delta_logit"]))
        seen[key][stratum].add(str(row.get("person_id", "")))

    out: List[Dict[str, Any]] = []
    for key, per_stratum in pools.items():
        kept = {
            s: v for s, v in per_stratum.items()
            if len(v) >= min_count and len(seen[key][s]) >= min_patients
        }
        if len(kept) < min_strata:
            continue
        means = {s: float(np.mean(v)) for s, v in kept.items()}
        vals = list(means.values())
        pooled_n = sum(len(v) for v in kept.values())
        pooled = float(sum(float(np.sum(v)) for v in kept.values()) / pooled_n)
        signs = {np.sign(v) for v in vals if abs(v) > 0.05}
        row: Dict[str, Any] = {
            **dict(zip(by, key)),
            "n_strata": len(kept),
            "pooled_mean_delta_logit": pooled,
            "min_stratum_mean": min(vals),
            "max_stratum_mean": max(vals),
            "across_stratum_range": max(vals) - min(vals),
            "consistent_sign": len(signs) <= 1,
            "n_measurements": pooled_n,
            "n_patients_total": len(set().union(*(seen[key][s] for s in kept))),
        }
        for s, m in means.items():
            row[f"mean__{s}"] = m
            row[f"n__{s}"] = len(kept[s])
            row[f"pat__{s}"] = len(seen[key][s])
        out.append(row)
    out.sort(key=lambda r: abs(r["pooled_mean_delta_logit"]), reverse=True)
    return out


def find_procedure_position(seq: Sequence[Dict[str, Any]]) -> Optional[int]:
    for i, tok in enumerate(seq):
        if int(tok.get("is_base", 0)) == 1:
            continue
        if str(tok.get("modality", "")).lower() == "procedure":
            return i
    return None


@torch.no_grad()
def procedure_substitution_for_episode(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    candidate_codes: Sequence[str],
    max_query_steps: int = 25,
    age_stats: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Score every candidate operation on one patient, holding the rest of the record fixed.

    Deletion is the wrong intervention for the index procedure: every episode in the cohort has
    exactly one, so removing it produces a patient configuration the model never saw in training,
    and the resulting delta measures distributional novelty rather than surgical risk. Swapping
    one operation for another stays inside the observed data, and because the same patient is
    rescored under each candidate, the comparison carries no case-mix confounding.

    Returns one row per candidate with the absolute logits under that operation; differences are
    taken later, against whichever reference the caller picks.
    """
    feat = episode_to_sequence_features(
        episode,
        code_vocab,
        add_base=True,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med_exposure,
    )
    seq = feat["sequence"]
    pos = find_procedure_position(seq)
    if pos is None:
        return []

    actual_code = str(seq[pos].get("raw_code", ""))
    person_id = str(episode.get("person_id", ""))
    label = int(feat.get("label") or 0)

    # Batched forwards over candidates (chunked to bound peak memory).
    chunk = 24
    rows: List[Dict[str, Any]] = []
    for start in range(0, len(candidate_codes), chunk):
        codes_chunk = list(candidate_codes[start : start + chunk])
        variants: List[Dict[str, Any]] = []
        codes_used: List[str] = []
        for code in codes_chunk:
            code_id = code_vocab.get(code)
            if code_id is None:
                continue
            swapped = {
                "sequence": [dict(t) for t in seq],
                "label": feat.get("label"),
                "person_id": feat.get("person_id"),
            }
            swapped["sequence"][pos]["code_id"] = int(code_id)
            swapped["sequence"][pos]["raw_code"] = code
            variants.append(swapped)
            codes_used.append(code)
        if not variants:
            continue

        batch = {k: v.to(device) for k, v in collate_fn(variants).items()}
        outputs = _forward_model(model, batch)
        valid = (batch["attention_mask"][0] > 0).nonzero(as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        base_pos = torch.nonzero(batch["base_mask"][0, valid] > 0, as_tuple=False).flatten()
        base_idx = int(valid[base_pos[0]].item()) if base_pos.numel() else int(valid[0].item())
        z_base = outputs["baseline_logits_all"][:, base_idx].detach().float().cpu().numpy()

        for code, zi in zip(codes_used, z_base):
            rows.append(
                {
                    "person_id": person_id,
                    "label_aki": label,
                    "actual_procedure": actual_code,
                    "counterfactual_procedure": code,
                    "is_actual": int(code == actual_code),
                    "baseline_logit": float(zi),
                    "mean_online_logit": None,
                    "n_online_queries": 0,
                }
            )
    return rows


def age_dose_response_for_episode(
    model: torch.nn.Module,
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    age_grid: Sequence[float],
    age_stats: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Re-score one patient at each age on the grid, holding the rest of the record fixed.

    Age is the one static token that carries a number: ``SEX|*`` and ``RACE|*`` are bare category
    tokens, so deleting them asks an answerable question ("what if the model did not know this
    patient's sex?"), but ``AGE`` stores the age in years in its ``value`` field. Deleting it only
    measures what the age *channel* is worth, averaged over the cohort, and says nothing about
    whether being older raises or lowers risk — the two directions are averaged together and the
    sign that survives is uninterpretable.

    Sweeping the value instead keeps the token in place and asks the directional question. Because
    the same patient is re-scored at every age, the curve is within-patient and carries no case-mix
    confounding. Scored on the [BASE] head, which only attends to preoperative tokens.
    """
    feat = episode_to_sequence_features(
        episode,
        code_vocab,
        add_base=True,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med_exposure,
    )
    seq = feat["sequence"]
    pos = next((i for i, t in enumerate(seq) if str(t.get("raw_code", "")) == "AGE"), None)
    if pos is None:
        return []

    person_id = str(episode.get("person_id", ""))
    label = int(feat.get("label") or 0)
    # Raw years (event_to_feature zeroes the shared value channel for AGE).
    actual_age = None
    for ev in episode.get("events", []) or []:
        if str(ev.get("code", "")) == "AGE" and ev.get("value") is not None:
            actual_age = float(ev["value"])
            break
    if actual_age is None:
        for tok in episode.get("static_tokens", []) or []:
            if str(tok.get("code", "")) == "AGE" and tok.get("value") is not None:
                actual_age = float(tok["value"])
                break
    if actual_age is None:
        actual_age = 0.0

    # The variants differ only in one token's age channel, so they share a length and a [BASE]
    # position and can go through as a single batch: one forward per patient instead of one per age.
    variants = []
    for age in age_grid:
        sw = {
            "sequence": [dict(t) for t in seq],
            "label": feat.get("label"),
            "person_id": feat.get("person_id"),
        }
        # AGE uses age_mlp (z-scored age_value), not value_mlp.
        sw["sequence"][pos]["value"] = 0.0
        sw["sequence"][pos]["value_mask"] = 0.0
        sw["sequence"][pos]["age_value"] = apply_age_zscore(float(age), age_stats)
        sw["sequence"][pos]["age_mask"] = 1.0
        variants.append(sw)

    batch = {k: v.to(device) for k, v in collate_fn(variants).items()}
    outputs = _forward_model(model, batch)
    valid = (batch["attention_mask"][0] > 0).nonzero(as_tuple=False).flatten()
    if valid.numel() == 0:
        return []
    base_pos = torch.nonzero(batch["base_mask"][0, valid] > 0, as_tuple=False).flatten()
    base_idx = int(valid[base_pos[0]].item()) if base_pos.numel() else int(valid[0].item())
    z = outputs["baseline_logits_all"][:, base_idx].detach().float().cpu().numpy()

    return [
        {
            "person_id": person_id,
            "label_aki": label,
            "actual_age": actual_age,
            "age_set": float(age),
            "baseline_logit": float(zi),
        }
        for age, zi in zip(age_grid, z)
    ]


def rank_procedures(
    rows: Sequence[Dict[str, Any]],
    reference_code: Optional[str] = None,
    *,
    min_episodes: int = 10,
) -> Dict[str, Any]:
    """Case-mix-free ranking: every operation scored on the same panel of patients.

    Each patient contributes one logit per candidate operation, so the between-operation
    comparison is within-patient and no adjustment for patient severity is needed. Values are
    reported relative to ``reference_code`` (default: the operation with the median mean logit).
    """
    by_code: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_code[str(r["counterfactual_procedure"])][str(r["person_id"])] = float(r["baseline_logit"])

    people_all: Optional[Set[str]] = None
    for code, per_person in by_code.items():
        people_all = set(per_person) if people_all is None else people_all & set(per_person)
    panel = sorted(people_all or set())
    if len(panel) < min_episodes:
        return {"error": "insufficient shared patient panel", "n_panel": len(panel)}

    means = {code: float(np.mean([by_code[code][p] for p in panel])) for code in by_code}
    if reference_code is None or reference_code not in means:
        reference_code = sorted(means, key=means.get)[len(means) // 2]
    ref_per_person = by_code[reference_code]

    ranked: List[Dict[str, Any]] = []
    for code, per_person in by_code.items():
        deltas = np.asarray([per_person[p] - ref_per_person[p] for p in panel], dtype=np.float64)
        ranked.append(
            {
                "procedure_code": code,
                "n_panel_patients": len(panel),
                "mean_logit": means[code],
                "mean_delta_logit_vs_reference": float(deltas.mean()),
                "std_delta_logit_vs_reference": float(deltas.std()) if deltas.size > 1 else 0.0,
                "n_episodes_actually_this_procedure": sum(
                    1 for r in rows if r["is_actual"] and r["counterfactual_procedure"] == code
                ),
            }
        )
    ranked.sort(key=lambda r: r["mean_delta_logit_vs_reference"], reverse=True)
    return {
        "reference_procedure": reference_code,
        "n_panel_patients": len(panel),
        "n_procedures": len(ranked),
        "ranked": ranked,
    }


def time_profile_for_code(
    records: Sequence[Dict[str, Any]],
    modality: str,
    raw_code: str,
    *,
    scope: str = "online",
    min_count: int = 5,
) -> List[Dict[str, Any]]:
    """Mean Δlogit of one code across time bins — the per-code shape of the time effect."""
    pools: Dict[str, List[float]] = defaultdict(list)
    for row in records:
        if row.get("scope") != scope:
            continue
        if row.get("modality") != modality or row.get("raw_code") != raw_code:
            continue
        pools[str(row.get("time_bin", ""))].append(float(row["delta_logit"]))
    order = ["preop_7d_48h", "preop_48h_0h", *POSTOP_BIN_NAMES]
    out: List[Dict[str, Any]] = []
    for tbin in order:
        vals = pools.get(tbin, [])
        if len(vals) < min_count:
            continue
        a = np.asarray(vals, dtype=np.float64)
        out.append(
            {
                "time_bin": tbin,
                "n": int(a.size),
                "mean_delta_logit": float(a.mean()),
                "std_delta_logit": float(a.std()) if a.size > 1 else 0.0,
            }
        )
    return out
