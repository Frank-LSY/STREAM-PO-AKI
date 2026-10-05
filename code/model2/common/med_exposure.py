"""Medication daily exposure utilities.

The model can attach a day-level active-medication vector to every event token.
When medication event codes are masked for MEM, the corresponding exposure entry
should also be masked to avoid trivial reconstruction.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional


def episode_med_day_matrix(episode: Dict[str, Any]) -> Optional[List[List[int]]]:
    """Prefer med_day_vector; accept active_med_day as alias."""
    m = episode.get("med_day_vector")
    if m is None:
        m = episode.get("active_med_day")
    if not m:
        return None
    return m


def infer_num_med_exposure(episodes: List[Dict[str, Any]]) -> int:
    for ep in episodes:
        mat = episode_med_day_matrix(ep)
        if mat and len(mat) > 0 and len(mat[0]) > 0:
            return len(mat[0])
    return 0


def time_day_to_bin_index(time_day: float, preop_days: int, num_bins: int) -> int:
    d = int(math.floor(float(time_day)))
    idx = d + int(preop_days)
    return min(max(idx, 0), num_bins - 1)


def _align_row(row: List[float], num_med_exposure: int) -> List[float]:
    if len(row) < num_med_exposure:
        return row + [0.0] * (num_med_exposure - len(row))
    if len(row) > num_med_exposure:
        return row[:num_med_exposure]
    return row


def med_vector_for_time_day(
    episode: Dict[str, Any],
    time_day: float,
    num_med_exposure: int,
) -> List[float]:
    if num_med_exposure <= 0:
        return []
    mat = episode_med_day_matrix(episode)
    if not mat:
        return [0.0] * num_med_exposure
    preop_days = int(episode.get("preop_days", 7))
    bi = time_day_to_bin_index(float(time_day), preop_days, len(mat))
    return _align_row([float(x) for x in mat[bi]], num_med_exposure)


def base_med_vector(episode: Dict[str, Any], num_med_exposure: int) -> List[float]:
    """Medication exposure at surgery index, used for [BASE]."""
    return med_vector_for_time_day(episode, 0.0, num_med_exposure)


def _candidate_med_code_lists(episode: Dict[str, Any]) -> List[List[str]]:
    """Return candidate episode-level med exposure vocab lists.

    The preprocessing code may store medication exposure dimensions under
    different names. This function tries common aliases. Each list position is
    interpreted as the dimension index of med_day_vector.
    """
    keys = [
        "med_exposure_codes",
        "selected_med_codes",
        "med_codes",
        "rxcui_list",
        "rxnorm_list",
        "med_list",
        "med_vocab",
    ]
    out: List[List[str]] = []
    for k in keys:
        v = episode.get(k)
        if isinstance(v, list) and v:
            out.append([str(x) for x in v])
    return out


_MED_EVENT_TYPE_SUFFIXES = ("|start", "|end")

# Separate from code vocab: shared drug embedding + additive boundary type.
EVENT_TYPE_NONE = 0
EVENT_TYPE_START = 1
EVENT_TYPE_END = 2
NUM_EVENT_TYPES = 3


def event_type_id_for_event(event: Dict[str, Any]) -> int:
    """Map medication ``event_type`` to embedding id; non-med / unknown → NONE (0)."""
    modality = str(event.get("modality", "")).lower()
    if modality != "med":
        return EVENT_TYPE_NONE
    et = str(event.get("event_type", "")).lower().strip()
    if et == "start":
        return EVENT_TYPE_START
    if et == "end":
        return EVENT_TYPE_END
    return EVENT_TYPE_NONE


def _strip_med_event_type_suffix(s: str) -> str:
    """Strip legacy ``|start``/``|end`` code suffixes if present."""
    for suf in _MED_EVENT_TYPE_SUFFIXES:
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _normalize_med_code(x: Any) -> List[str]:
    s = _strip_med_event_type_suffix(str(x))
    vals = {s}
    if "|" in s:
        vals.add(s.split("|", 1)[1])
    if s.startswith("RXNORM|"):
        vals.add(s.replace("RXNORM|", ""))
    if s.startswith("RXCUI|"):
        vals.add(s.replace("RXCUI|", ""))
    return list(vals)


def med_exposure_index_for_event(
    episode: Dict[str, Any],
    event: Dict[str, Any],
    num_med_exposure: int,
) -> int:
    """Infer the exposure-vector column for a medication event.

    Returns -1 when the mapping is unavailable. Supported event fields include
    `med_exposure_idx`, `med_idx`, `exposure_idx`, `drug_index`, `rxcui`,
    `drug_code`, and `code`.
    """
    if num_med_exposure <= 0:
        return -1
    for k in ("med_exposure_idx", "med_idx", "exposure_idx", "drug_index"):
        if k in event:
            try:
                idx = int(event[k])
                if 0 <= idx < num_med_exposure:
                    return idx
            except Exception:
                pass

    cand_vals = []
    for k in ("rxcui", "drug_code", "code"):
        if k in event and event[k] is not None:
            cand_vals.extend(_normalize_med_code(event[k]))
    cand_set = set(cand_vals)
    if not cand_set:
        return -1

    for code_list in _candidate_med_code_lists(episode):
        for idx, code in enumerate(code_list[:num_med_exposure]):
            if cand_set.intersection(_normalize_med_code(code)):
                return idx
    return -1
