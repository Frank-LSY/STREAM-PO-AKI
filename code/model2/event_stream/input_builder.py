from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List, Iterable, Union, Optional
from collections import Counter

from common.lab_zscore import apply_lab_zscore
from common.age_zscore import apply_age_zscore
from common.med_exposure import (
    base_med_vector,
    event_type_id_for_event,
    med_vector_for_time_day,
    med_exposure_index_for_event,
)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[MASK]", "[BASE]", "[SURGERY]"]
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"
BASE_TOKEN = "[BASE]"
SURGERY_TOKEN = "[SURGERY]"

MODALITY_VOCAB = {
    "static": 0,
    "procedure": 1,
    "condition": 2,
    "lab": 3,
    "med": 4,
    "base": 5,
}

def load_code_vocab_json(path: Union[str, Path]) -> Dict[str, int]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    vocab = {str(k): int(v) for k, v in raw.items()}
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            # Allow loading older vocab by appending missing special tokens.
            vocab[tok] = max(vocab.values(), default=-1) + 1
    return vocab


def build_code_vocab(episodes: Iterable[Dict[str, Any]], min_freq: int = 1) -> Dict[str, int]:
    counter = Counter()
    for ep in episodes:
        for ev in ep.get("events", []):
            # Drug identity only; start/end is a separate event_type embedding.
            counter[str(ev.get("code", UNK_TOKEN))] += 1
    vocab: Dict[str, int] = {}
    for idx, tok in enumerate(SPECIAL_TOKENS):
        vocab[tok] = idx
    next_idx = len(vocab)
    for code, freq in counter.items():
        if freq >= min_freq and code not in vocab:
            vocab[code] = next_idx
            next_idx += 1
    return vocab


def _is_procedure_event(event: Dict[str, Any]) -> bool:
    modality = str(event.get("modality", "")).lower()
    if modality == "procedure":
        return True
    et = str(event.get("event_type", "")).lower()
    return et in {"procedure", "surgery", "index_surgery"}


def event_to_feature(
    event: Dict[str, Any],
    code_vocab: Dict[str, int],
    lab_stats: Optional[Dict[str, Any]] = None,
    age_stats: Optional[Dict[str, Any]] = None,
    *,
    drop_lab_values: bool = False,
    generic_surgery: bool = False,
) -> Dict[str, Any]:
    code = str(event.get("code", UNK_TOKEN))
    modality = str(event.get("modality", "condition")).lower()
    is_procedure = _is_procedure_event(event)
    if generic_surgery and is_procedure:
        code = SURGERY_TOKEN
    code_id = code_vocab.get(code, code_vocab[UNK_TOKEN])
    modality_id = MODALITY_VOCAB.get(modality, MODALITY_VOCAB["condition"])
    event_type_id = event_type_id_for_event(event)

    # AGE uses a dedicated z-scored channel + age_mlp (not shared value_mlp).
    is_age = code == "AGE"
    age_value = 0.0
    age_mask = 0.0
    value = event.get("value", None)
    if is_age and value is not None:
        age_value = apply_age_zscore(float(value), age_stats)
        age_mask = 1.0
        value_float = 0.0
        value_mask = 0.0
    elif value is None or (drop_lab_values and modality == "lab"):
        value_float = 0.0
        value_mask = 0.0
    else:
        value_float = float(value)
        if modality == "lab" and lab_stats is not None:
            value_float = apply_lab_zscore(value_float, str(event.get("code", UNK_TOKEN)), lab_stats)
        value_mask = 1.0

    time_day = float(event.get("time_day", 0.0))
    time_hour = float(event.get("time_hour", time_day * 24.0))

    is_preop = bool(time_day < 0.0 or is_procedure)
    is_postop = bool(time_day > 0.0 and not is_procedure)
    mem_candidate = bool(is_preop and modality in {"condition", "lab", "med"})

    return {
        "code_id": code_id,
        "raw_code": code,
        "modality_id": modality_id,
        "modality": modality,
        "event_type_id": event_type_id,
        "time_day": time_day,
        "time_hour": time_hour,
        "value": value_float,
        "value_mask": value_mask,
        "age_value": age_value,
        "age_mask": age_mask,
        "is_base": 0,
        "is_preop": 1 if is_preop else 0,
        "is_postop": 1 if is_postop else 0,
        "is_procedure": 1 if is_procedure else 0,
        "mem_candidate": 1 if mem_candidate else 0,
        "med_exposure_idx": -1,
    }


def _make_base_token(
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    num_med_exposure: int = 0,
) -> Dict[str, Any]:
    tok: Dict[str, Any] = {
        "code_id": code_vocab[BASE_TOKEN],
        "raw_code": BASE_TOKEN,
        "modality_id": MODALITY_VOCAB["base"],
        "modality": "base",
        "event_type_id": 0,
        "time_day": 0.0,
        "time_hour": 0.0,
        "value": 0.0,
        "value_mask": 0.0,
        "age_value": 0.0,
        "age_mask": 0.0,
        "is_base": 1,
        "is_preop": 0,
        "is_postop": 0,
        "is_procedure": 0,
        "mem_candidate": 0,
        "med_exposure_idx": -1,
    }
    if num_med_exposure > 0:
        tok["med_day_feats"] = base_med_vector(episode, num_med_exposure)
    return tok


def _find_base_insert_index(seq: List[Dict[str, Any]]) -> int:
    """Insert [BASE] after the surgery/procedure token when available.

    Fallback: after the last preoperative token and before the first postoperative token.
    """
    proc_indices = [i for i, x in enumerate(seq) if x.get("is_procedure", 0) == 1]
    if proc_indices:
        return proc_indices[-1] + 1
    pre_indices = [i for i, x in enumerate(seq) if float(x.get("time_day", 0.0)) <= 0.0]
    if pre_indices:
        return pre_indices[-1] + 1
    return 0


def _recompute_pre_post_around_base(seq: List[Dict[str, Any]]) -> None:
    """Tokens before BASE are preoperative; after BASE are postoperative."""
    base_idx = next((i for i, x in enumerate(seq) if x.get("is_base", 0) == 1), None)
    if base_idx is None:
        return
    for i, x in enumerate(seq):
        if i < base_idx:
            x["is_preop"] = 1
            x["is_postop"] = 0
            if x["modality"] in {"condition", "lab", "med"}:
                x["mem_candidate"] = 1
            else:
                x["mem_candidate"] = 0
        elif i == base_idx:
            x["is_preop"] = 0
            x["is_postop"] = 0
            x["mem_candidate"] = 0
        else:
            x["is_preop"] = 0
            x["is_postop"] = 1
            x["mem_candidate"] = 0


def episode_to_sequence_features(
    episode: Dict[str, Any],
    code_vocab: Dict[str, int],
    add_base: bool = True,
    lab_stats: Optional[Dict[str, Any]] = None,
    age_stats: Optional[Dict[str, Any]] = None,
    num_med_exposure: int = 0,
    *,
    baseline_token: str = "dedicated",
    drop_conditions: bool = False,
    drop_lab_values: bool = False,
    drop_labs: bool = False,
    drop_meds: bool = False,
    generic_surgery: bool = False,
) -> Dict[str, Any]:
    """Build token sequence for one episode.

    ``baseline_token``:
      - ``dedicated``: insert ``[BASE]`` after the procedure token (default).
      - ``procedure``: do not insert ``[BASE]``; mark the index-surgery token
        as the baseline summary (``is_base=1``) so postop attention structure
        is preserved without a dedicated BASE embedding.
    """
    baseline_token = str(baseline_token).lower().strip()
    if baseline_token not in {"dedicated", "procedure"}:
        raise ValueError(f"baseline_token must be 'dedicated' or 'procedure', got {baseline_token!r}")
    use_dedicated_base = bool(add_base) and baseline_token == "dedicated"

    seq: List[Dict[str, Any]] = []
    for ev in episode.get("events", []):
        modality = str(ev.get("modality", "")).lower()
        if drop_conditions and modality == "condition":
            continue
        if drop_labs and modality == "lab":
            continue
        if drop_meds and modality == "med":
            continue
        tok = event_to_feature(
            ev,
            code_vocab,
            lab_stats=lab_stats,
            age_stats=age_stats,
            drop_lab_values=drop_lab_values,
            generic_surgery=generic_surgery,
        )
        if num_med_exposure > 0:
            tok["med_day_feats"] = med_vector_for_time_day(
                episode, float(tok["time_day"]), num_med_exposure
            )
            if tok["modality"] == "med":
                tok["med_exposure_idx"] = med_exposure_index_for_event(
                    episode, ev, num_med_exposure
                )
        seq.append(tok)

    if use_dedicated_base:
        insert_at = _find_base_insert_index(seq)
        seq.insert(insert_at, _make_base_token(episode, code_vocab, num_med_exposure))
    elif baseline_token == "procedure":
        # Use the (last) procedure token as the single baseline summary state.
        proc_indices = [i for i, x in enumerate(seq) if x.get("is_procedure", 0) == 1]
        if proc_indices:
            bi = proc_indices[-1]
            seq[bi]["is_base"] = 1
            # Keep procedure modality/code; only the base role changes.
        else:
            # Fallback: insert dedicated BASE if no procedure token exists.
            insert_at = _find_base_insert_index(seq)
            seq.insert(insert_at, _make_base_token(episode, code_vocab, num_med_exposure))

    _recompute_pre_post_around_base(seq)

    lbl = episode.get("label_aki")
    return {
        "sequence": seq,
        "label": float(lbl) if lbl is not None else None,
        "person_id": episode.get("person_id"),
    }
