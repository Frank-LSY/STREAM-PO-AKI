"""Shared 6h time-grid featurization (-7d preop .. +30d postop, BASE at t0)."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

EPS = 1e-6
BUCKET_SIZE_H = 6.0
PREOP_DAYS = 7
POSTOP_DAYS = 30
PREOP_HOURS = PREOP_DAYS * 24.0
POSTOP_HOURS = POSTOP_DAYS * 24.0
NUM_PREOP_BUCKETS = int(PREOP_HOURS / BUCKET_SIZE_H)  # 28
NUM_POSTOP_BUCKETS = int(POSTOP_HOURS / BUCKET_SIZE_H)  # 120
BASE_BUCKET_IDX = NUM_PREOP_BUCKETS
SEQ_LEN = NUM_PREOP_BUCKETS + 1 + NUM_POSTOP_BUCKETS  # 149


def preop_bucket_hour_left(i: int) -> float:
    return -PREOP_HOURS + i * BUCKET_SIZE_H


def postop_bucket_hour_left(j: int) -> float:
    return j * BUCKET_SIZE_H


def bucket_hour_left_for_index(idx: int) -> float:
    if idx < NUM_PREOP_BUCKETS:
        return preop_bucket_hour_left(idx)
    if idx == BASE_BUCKET_IDX:
        return 0.0
    return postop_bucket_hour_left(idx - BASE_BUCKET_IDX - 1)


def _sanitize_token(token: str) -> str:
    return str(token).replace("|", "_").replace("/", "_").replace(" ", "_").replace(".", "_")


def _lab_event_code(lab_code: int) -> str:
    return f"LAB|{int(lab_code)}"


def collect_vocab(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ccs_codes: Set[str] = set()
    procedure_codes: Set[str] = set()
    sex_codes: Set[str] = set()
    race_codes: Set[str] = set()

    for ep in episodes:
        for tok in ep.get("static_tokens", []):
            code = str(tok.get("code", ""))
            if code.startswith("SEX|"):
                sex_codes.add(code)
            elif code.startswith("RACE|"):
                race_codes.add(code)
        for tok in ep.get("procedure_tokens", []):
            procedure_codes.add(str(tok.get("code", "PROC_OTHERS")))
        for ev in ep.get("condition_events", []) + ep.get("lab_events", []) + ep.get("med_events", []):
            if ev.get("modality") == "condition" or str(ev.get("code", "")).startswith("CCS|"):
                ccs_codes.add(str(ev.get("code", "COND_OTHERS")))

    lab_codes = list(episodes[0].get("selected_lab_names", [])) if episodes else []
    rxcui_codes = list(episodes[0].get("selected_rxcui_names", [])) if episodes else []
    return {
        "ccs_codes": sorted(ccs_codes),
        "procedure_codes": sorted(procedure_codes),
        "sex_codes": sorted(sex_codes),
        "race_codes": sorted(race_codes),
        "lab_codes": [int(x) for x in lab_codes],
        "rxcui_codes": [str(x) for x in rxcui_codes],
    }


def fit_lab_zscore_all_labs(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_code: Dict[str, List[float]] = defaultdict(list)
    all_vals: List[float] = []
    for ep in episodes:
        for ev in ep.get("lab_events", []):
            v = ev.get("value")
            if v is None:
                continue
            code = str(ev.get("code", ""))
            if not code.startswith("LAB|"):
                continue
            fv = float(v)
            by_code[code].append(fv)
            all_vals.append(fv)
    per_code: Dict[str, Dict[str, float]] = {}
    for code, vals in by_code.items():
        arr = np.asarray(vals, dtype=np.float64)
        std = float(arr.std())
        if std < EPS or not np.isfinite(std):
            std = 1.0
        per_code[code] = {"mean": float(arr.mean()), "std": std}
    if all_vals:
        arr = np.asarray(all_vals, dtype=np.float64)
        g_std = float(arr.std())
        if g_std < EPS:
            g_std = 1.0
        g_mean = float(arr.mean())
    else:
        g_mean, g_std = 0.0, 1.0
    return {"per_code": per_code, "global_mean": g_mean, "global_std": g_std, "version": 1}


def apply_lab_zscore(raw: float, lab_event_code: str, lab_stats: Dict[str, Any]) -> float:
    pc = lab_stats.get("per_code") or {}
    if lab_event_code in pc:
        m = float(pc[lab_event_code]["mean"])
        s = float(pc[lab_event_code]["std"])
    else:
        m = float(lab_stats.get("global_mean", 0.0))
        s = float(lab_stats.get("global_std", 1.0))
    if s < EPS:
        s = 1.0
    z = (float(raw) - m) / s
    return float(z) if np.isfinite(z) else 0.0


def build_feature_layout(vocab: Dict[str, Any]) -> Dict[str, Any]:
    n_ccs = len(vocab["ccs_codes"])
    n_lab = len(vocab["lab_codes"])
    n_med = len(vocab["rxcui_codes"])
    n_sex = len(vocab["sex_codes"])
    n_race = len(vocab["race_codes"])
    n_proc = len(vocab["procedure_codes"])
    def sl(a: int, b: int) -> List[int]:
        return [a, b]

    slices = {
        "ccs": sl(0, n_ccs),
        "lab": sl(n_ccs, n_ccs + 2 * n_lab),
        "med": sl(n_ccs + 2 * n_lab, n_ccs + 2 * n_lab + n_med),
        "has_event": sl(n_ccs + 2 * n_lab + n_med, n_ccs + 2 * n_lab + n_med + 1),
        "age": sl(n_ccs + 2 * n_lab + n_med + 1, n_ccs + 2 * n_lab + n_med + 2),
        "sex": sl(n_ccs + 2 * n_lab + n_med + 2, n_ccs + 2 * n_lab + n_med + 2 + n_sex),
        "race": sl(
            n_ccs + 2 * n_lab + n_med + 2 + n_sex,
            n_ccs + 2 * n_lab + n_med + 2 + n_sex + n_race,
        ),
        "procedure": sl(
            n_ccs + 2 * n_lab + n_med + 2 + n_sex + n_race,
            n_ccs + 2 * n_lab + n_med + 2 + n_sex + n_race + n_proc,
        ),
    }
    return {
        "n_features": slices["procedure"][1],
        "slices": slices,
        "ccs_to_i": {c: i for i, c in enumerate(vocab["ccs_codes"])},
        "lab_to_i": {int(c): i for i, c in enumerate(vocab["lab_codes"])},
        "rxcui_to_i": {str(c): i for i, c in enumerate(vocab["rxcui_codes"])},
        "sex_to_i": {c: i for i, c in enumerate(vocab["sex_codes"])},
        "race_to_i": {c: i for i, c in enumerate(vocab["race_codes"])},
        "proc_to_i": {c: i for i, c in enumerate(vocab["procedure_codes"])},
    }


def _days_overlapping_bin(h0: float, h1: float) -> List[int]:
    d0 = int(math.floor(h0 / 24.0))
    d1 = int(math.floor((h1 - 1e-9) / 24.0))
    if d1 < d0:
        d1 = d0
    return list(range(d0, d1 + 1))


def _med_max_for_bin(
    ep: Dict[str, Any],
    h0: float,
    h1: float,
    rxcui_to_i: Dict[str, int],
    n_med: int,
) -> np.ndarray:
    out = np.zeros(n_med, dtype=np.float32)
    preop_days = int(ep.get("preop_days", PREOP_DAYS))
    mat = ep.get("med_day_vector") or []
    for d in _days_overlapping_bin(h0, h1):
        bi = d + preop_days
        if 0 <= bi < len(mat):
            row = mat[bi]
            for i in range(min(n_med, len(row))):
                out[i] = max(out[i], float(row[i]))
    names = ep.get("selected_rxcui_names") or []
    for ev in ep.get("med_events", []):
        th = ev.get("time_hour")
        if th is None:
            continue
        thf = float(th)
        if thf < h0 or thf >= h1:
            continue
        rx = str(ev.get("rxcui") or ev.get("code", ""))
        if rx in rxcui_to_i:
            out[rxcui_to_i[rx]] = 1.0
    return out


def _assign_events_to_bin(
    ep: Dict[str, Any],
    h0: float,
    h1: float,
    row: np.ndarray,
    layout: Dict[str, Any],
    lab_stats: Dict[str, Any],
) -> bool:
    has = False
    sl = layout["slices"]
    ccs_map = layout["ccs_to_i"]
    lab_map = layout["lab_to_i"]
    lab_off = int(sl["lab"][0])

    for ev in ep.get("condition_events", []):
        th = ev.get("time_hour")
        if th is None:
            continue
        thf = float(th)
        if thf < h0 or thf >= h1:
            continue
        code = str(ev.get("code", "COND_OTHERS"))
        if code in ccs_map:
            row[ccs_map[code]] = 1.0
            has = True

    last_lab: Dict[int, float] = {}
    for ev in ep.get("lab_events", []):
        th = ev.get("time_hour")
        if th is None:
            continue
        thf = float(th)
        if thf < h0 or thf >= h1:
            continue
        code = str(ev.get("code", ""))
        if not code.startswith("LAB|"):
            continue
        try:
            lc = int(code.split("|", 1)[1])
        except (IndexError, ValueError):
            continue
        v = ev.get("value")
        if v is not None and lc in lab_map:
            last_lab[lc] = float(v)
            has = True

    for lc, val in last_lab.items():
        li = lab_map[lc]
        row[lab_off + 2 * li] = apply_lab_zscore(val, _lab_event_code(lc), lab_stats)
        row[lab_off + 2 * li + 1] = 1.0

    med = _med_max_for_bin(ep, h0, h1, layout["rxcui_to_i"], int(sl["med"][1]) - int(sl["med"][0]))
    if med.any():
        m0, m1 = int(sl["med"][0]), int(sl["med"][1])
        row[m0:m1] = np.maximum(row[m0:m1], med)
        has = True
    if has:
        row[int(sl["has_event"][0])] = 1.0
    return has


def _apply_locf_across_buckets(out: np.ndarray, layout: Dict[str, Any]) -> None:
    """Carry last observed values forward across 6h buckets (preop → BASE → postop).

    Applies to:
    - CCS (binary): once a code is 1, it stays 1 in later buckets
    - Labs (value + missing flag): last z-score carried; missing=1 when carried
    - Medications (continuous exposure): last positive exposure carried
    """
    sl = layout["slices"]
    has_event_idx = int(sl["has_event"][0])

    ccs0, ccs1 = int(sl["ccs"][0]), int(sl["ccs"][1])
    n_ccs = ccs1 - ccs0
    carried_ccs = np.zeros(n_ccs, dtype=np.float32)
    ccs_active = np.zeros(n_ccs, dtype=bool)

    lab_off = int(sl["lab"][0])
    n_lab = (int(sl["lab"][1]) - lab_off) // 2
    carried_lab = np.zeros(n_lab, dtype=np.float32)
    lab_active = np.zeros(n_lab, dtype=bool)

    med0, med1 = int(sl["med"][0]), int(sl["med"][1])
    n_med = med1 - med0
    carried_med = np.zeros(n_med, dtype=np.float32)
    med_active = np.zeros(n_med, dtype=bool)

    for t in range(out.shape[0]):
        row = out[t]
        carried_any = False

        if n_ccs > 0:
            ccs = row[ccs0:ccs1]
            for i in range(n_ccs):
                if ccs[i] > 0.5:
                    carried_ccs[i] = 1.0
                    ccs_active[i] = True
                elif ccs_active[i]:
                    ccs[i] = carried_ccs[i]
                    carried_any = True

        if n_lab > 0:
            for li in range(n_lab):
                val_idx = lab_off + 2 * li
                miss_idx = val_idx + 1
                if row[miss_idx] > 0.5:
                    carried_lab[li] = float(row[val_idx])
                    lab_active[li] = True
                elif lab_active[li]:
                    row[val_idx] = carried_lab[li]
                    row[miss_idx] = 1.0
                    carried_any = True

        if n_med > 0:
            med = row[med0:med1]
            for i in range(n_med):
                if med[i] > 0.0:
                    carried_med[i] = float(med[i])
                    med_active[i] = True
                elif med_active[i]:
                    med[i] = carried_med[i]
                    carried_any = True

        if carried_any:
            row[has_event_idx] = 1.0


# Backward-compatible alias
_apply_lab_locf_across_buckets = _apply_locf_across_buckets


def _fill_static_base(row: np.ndarray, ep: Dict[str, Any], layout: Dict[str, Any]) -> None:
    sl = layout["slices"]
    for tok in ep.get("static_tokens", []):
        code = str(tok.get("code", ""))
        if code == "AGE":
            v = tok.get("value")
            if v is not None:
                row[int(sl["age"][0])] = float(v) / 100.0
        elif code.startswith("SEX|") and code in layout["sex_to_i"]:
            row[int(sl["sex"][0]) + layout["sex_to_i"][code]] = 1.0
        elif code.startswith("RACE|") and code in layout["race_to_i"]:
            row[int(sl["race"][0]) + layout["race_to_i"][code]] = 1.0
    for tok in ep.get("procedure_tokens", []):
        code = str(tok.get("code", "PROC_OTHERS"))
        if code in layout["proc_to_i"]:
            row[int(sl["procedure"][0]) + layout["proc_to_i"][code]] = 1.0
    row[int(sl["has_event"][0])] = 1.0


def obs_feature_indices(layout: Dict[str, Any]) -> np.ndarray:
    """Feature columns covered by the observation mask: CCS, lab values, medications (in that order)."""
    sl = layout["slices"]
    ccs = np.arange(int(sl["ccs"][0]), int(sl["ccs"][1]))
    lab = np.arange(int(sl["lab"][0]), int(sl["lab"][1]), 2)
    med = np.arange(int(sl["med"][0]), int(sl["med"][1]))
    return np.concatenate([ccs, lab, med])


def episode_to_bucket_obs(
    ep: Dict[str, Any],
    layout: Dict[str, Any],
    lab_stats: Dict[str, Any],
) -> np.ndarray:
    """Bool [SEQ_LEN, n_obs]: True where a CCS / lab / medication is actually observed in the bucket
    (before LOCF). Column order follows ``obs_feature_indices``."""
    raw = _episode_to_bucket_raw(ep, layout, lab_stats)
    sl = layout["slices"]
    ccs = raw[:, int(sl["ccs"][0]):int(sl["ccs"][1])] > 0.5
    lab = raw[:, int(sl["lab"][0]) + 1:int(sl["lab"][1]):2] > 0.5
    med = raw[:, int(sl["med"][0]):int(sl["med"][1])] > 0.0
    return np.concatenate([ccs, lab, med], axis=1)


def episode_to_bucket_features(
    ep: Dict[str, Any],
    layout: Dict[str, Any],
    lab_stats: Dict[str, Any],
) -> np.ndarray:
    """Return float32 array [SEQ_LEN, n_features]."""
    out = _episode_to_bucket_raw(ep, layout, lab_stats)
    _apply_locf_across_buckets(out, layout)
    return out


def _episode_to_bucket_raw(
    ep: Dict[str, Any],
    layout: Dict[str, Any],
    lab_stats: Dict[str, Any],
) -> np.ndarray:
    n_f = layout["n_features"]
    out = np.zeros((SEQ_LEN, n_f), dtype=np.float32)

    for i in range(NUM_PREOP_BUCKETS):
        h0 = preop_bucket_hour_left(i)
        h1 = h0 + BUCKET_SIZE_H
        _assign_events_to_bin(ep, h0, h1, out[i], layout, lab_stats)

    _fill_static_base(out[BASE_BUCKET_IDX], ep, layout)

    for j in range(NUM_POSTOP_BUCKETS):
        h0 = postop_bucket_hour_left(j)
        h1 = h0 + BUCKET_SIZE_H
        idx = BASE_BUCKET_IDX + 1 + j
        _assign_events_to_bin(ep, h0, h1, out[idx], layout, lab_stats)
    return out


def max_postop_bucket_index(ep: Dict[str, Any]) -> int:
    """Last postop bucket to score (AKI+ truncates at last event bin)."""
    label = int(ep.get("label_aki", 0))
    if label != 1:
        return NUM_POSTOP_BUCKETS - 1
    last_h = None
    for ev in ep.get("events", []) or []:
        th = ev.get("time_hour")
        if th is not None and float(th) >= 0:
            last_h = float(th) if last_h is None else max(last_h, float(th))
    if last_h is None or not np.isfinite(last_h):
        return NUM_POSTOP_BUCKETS - 1
    for j in range(NUM_POSTOP_BUCKETS):
        h0 = postop_bucket_hour_left(j)
        h1 = h0 + BUCKET_SIZE_H
        if last_h >= h0 and last_h < h1:
            return j
    return NUM_POSTOP_BUCKETS - 1


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
