# preprocess_real_ehr.py

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from tqdm import tqdm

@dataclass
class PreprocessConfig:
    preop_days: int = 7
    postop_days: int = 30
    day_bin_size: int = 1                # lab grid按天分桶
    use_ccs_for_condition: bool = True   # condition优先使用ccs_category
    include_med_end_event: bool = False  # v1只保留med start event
    deduplicate_same_event: bool = True
    lowercase_lab_name: bool = False
    keep_only_window_overlap_meds: bool = True


def parse_time(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    s = s.replace("Z", "")

    fmts = [
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def hours_from_anchor(t: datetime, t0: datetime) -> float:
    return (t - t0).total_seconds() / 3600.0


def days_from_anchor(t: datetime, t0: datetime) -> float:
    return (t - t0).total_seconds() / 86400.0


def in_window(t: datetime, start: datetime, end: datetime) -> bool:
    return start <= t <= end


def interval_overlaps(
    start_a: datetime,
    end_a: datetime,
    start_b: datetime,
    end_b: datetime,
) -> bool:
    return start_a <= end_b and start_b <= end_a


def normalize_ccs(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.endswith(".0"):
        s = s[:-2]
    return s


# CCS category containing the PO-AKI outcome ICD family (ICD-9 584.x / ICD-10 N17.x,
# and N19). Kept out of predictors to avoid outcome leakage into condition features.
EXCLUDED_CONDITION_CCS: frozenset = frozenset({"157"})


def is_outcome_equivalent_aki_condition(condition_item: Dict[str, Any]) -> bool:
    """True if this condition is in the PO-AKI outcome code family.

    CCS|157 is the bucket that ``icd_to_ccs`` assigns to ICD-9 584.x and ICD-10 N17.x
    (and N19). Those codes define the label, so they must not appear as input events.
    Also matches raw ICD source values when CCS mapping is unavailable.
    """
    ccs = normalize_ccs(condition_item.get("ccs_category"))
    if ccs in EXCLUDED_CONDITION_CCS:
        return True
    raw = str(condition_item.get("condition_source_value", "") or "").strip().upper()
    # forms like ICD10CM|N17.9, ICD9CM|584.5, or bare N17.9 / 584.9
    code = raw.split("|", 1)[-1].strip()
    if code.startswith("N17") or code.startswith("584") or code == "N19" or code.startswith("N19."):
        return True
    return False


def normalize_rxcui(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s else None


def normalize_lab_name(x: Any, lowercase: bool = False) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    return s.lower() if lowercase else s


def infer_aki_label(record: Dict[str, Any]) -> int:
    outcomes = record.get("outcome", []) or []
    for x in outcomes:
        s = str(x).replace("'", "").strip().upper()
        if s == "AKI":
            return 1
    return 0


def infer_aki_time_from_last_condition(record: Dict[str, Any]) -> Optional[datetime]:
    """
    User-defined rule:
    - no AKI -> None
    - AKI -> use the last condition timestamp
    """
    if infer_aki_label(record) == 0:
        return None

    cond_times = []
    for c in record.get("condition", []) or []:
        t = parse_time(c.get("time"))
        if t is not None:
            cond_times.append(t)

    return max(cond_times) if cond_times else None


def map_condition_code(
    condition_item: Dict[str, Any],
    ccs_set: set,
    use_ccs_for_condition: bool = True,
) -> str:
    if use_ccs_for_condition:
        ccs = normalize_ccs(condition_item.get("ccs_category"))
        if ccs is not None and ccs in ccs_set:
            return f"CCS|{ccs}"
        return "COND_OTHERS"

    raw_code = str(condition_item.get("condition_source_value", "")).strip()
    return raw_code if raw_code else "COND_OTHERS"


def map_lab_code(
    lab_item: Dict[str, Any],
    lab_list: Sequence[int],
) -> str:
    lab_code = lab_item.get("lab_code")
    if lab_code is not None and lab_code in lab_list:
        return f"LAB|{lab_code}"
    return "LAB_OTHERS"


def map_med_code(
    med_item: Dict[str, Any],
    rxcui_set: set,
) -> str:
    rxcui = normalize_rxcui(med_item.get("rxcui"))
    if rxcui is not None and rxcui in rxcui_set:
        return f"RXCUI|{rxcui}"
    return "MED_OTHERS"


def build_episode_from_record(
    record: Dict[str, Any],
    ccs_list: Sequence[Any],
    rxcui_list: Sequence[Any],
    lab_list: Sequence[str],
    config: PreprocessConfig = PreprocessConfig(),
) -> Dict[str, Any]:
    """
    输入一条record，输出单个perioperative episode。
    """

    ccs_set = {normalize_ccs(x) for x in ccs_list if normalize_ccs(x) is not None}
    rxcui_set = {normalize_rxcui(x) for x in rxcui_list if normalize_rxcui(x) is not None}

    lab_set = {int(x) for x in lab_list if x is not None}
    selected_lab_names = [int(x) for x in lab_list if x is not None]
    lab_name_to_idx = {name: i for i, name in enumerate(selected_lab_names)}

    selected_rxcui_names = [normalize_rxcui(x) for x in rxcui_list if normalize_rxcui(x) is not None]
    rxcui_name_to_idx = {name: i for i, name in enumerate(selected_rxcui_names)}

    person_id = str(record.get("person_id"))
    t0 = parse_time(record.get("procedure_datetime"))
    if t0 is None:
        raise ValueError(f"Invalid procedure_datetime for person_id={person_id}")

    label_aki = infer_aki_label(record)
    aki_time = infer_aki_time_from_last_condition(record)

    window_start = t0 - timedelta(days=config.preop_days)
    default_window_end = t0 + timedelta(days=config.postop_days)
    window_end = min(default_window_end, aki_time) if aki_time is not None else default_window_end

    static_tokens: List[Dict[str, Any]] = []
    procedure_tokens: List[Dict[str, Any]] = []
    condition_events: List[Dict[str, Any]] = []
    lab_events: List[Dict[str, Any]] = []
    med_events: List[Dict[str, Any]] = []

    seen_event_keys = set()

    # static tokens at day -7
    age = record.get("age")
    if age is not None:
        static_tokens.append({
            "time": window_start.isoformat(sep=" "),
            "time_day": -float(config.preop_days),
            "time_hour": -24.0 * config.preop_days,
            "phase": 0,
            "modality": "static",
            "code": "AGE",
            "value": float(age),
        })

    gender = record.get("gender_concept_id")
    static_tokens.append({
        "time": window_start.isoformat(sep=" "),
        "time_day": -float(config.preop_days),
        "time_hour": -24.0 * config.preop_days,
        "phase": 0,
        "modality": "static",
        "code": f"SEX|{gender if gender is not None else 'UNKNOWN'}",
        "value": None,
    })

    race = record.get("race_concept_id")
    static_tokens.append({
        "time": window_start.isoformat(sep=" "),
        "time_day": -float(config.preop_days),
        "time_hour": -24.0 * config.preop_days,
        "phase": 0,
        "modality": "static",
        "code": f"RACE|{race if race is not None else 'UNKNOWN'}",
        "value": None,
    })

    # procedure token at day 0
    procedure_tokens.append({
        "time": t0.isoformat(sep=" "),
        "time_day": 0.0,
        "time_hour": 0.0,
        "phase": 1,
        "modality": "procedure",
        "code": str(record.get("procedure_source_value", "PROC_OTHERS")),
        "value": None,
    })

    # condition events
    for c in record.get("condition", []) or []:
        t = parse_time(c.get("time"))
        if t is None or not in_window(t, window_start, window_end):
            continue
        # Drop PO-AKI outcome-equivalent diagnoses from predictors (CCS|157 / 584 / N17 / N19).
        if is_outcome_equivalent_aki_condition(c):
            continue

        code = map_condition_code(
            c,
            ccs_set=ccs_set,
            use_ccs_for_condition=config.use_ccs_for_condition,
        )
        # Belt-and-suspenders: skip even if mapping produced CCS|157 via another path.
        if code == "CCS|157":
            continue

        event = {
            "time": t.isoformat(sep=" "),
            "time_day": days_from_anchor(t, t0),
            "time_hour": hours_from_anchor(t, t0),
            "phase": 0 if t < t0 else 1,
            "modality": "condition",
            "code": code,
            "raw_code": c.get("condition_source_value"),
            "value": None,
        }

        if config.deduplicate_same_event:
            key = (event["time"], event["modality"], event["code"])
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)

        condition_events.append(event)

    # lab events
    for l in record.get("lab", []) or []:
        t = parse_time(l.get("time"))
        if t is None or not in_window(t, window_start, window_end):
            continue

        code = map_lab_code(
            l,
            lab_list=lab_list,
        )

        value = l.get("value", None)
        try:
            value = float(value) if value is not None else None
        except (TypeError, ValueError):
            value = None

        event = {
            "time": t.isoformat(sep=" "),
            "time_day": days_from_anchor(t, t0),
            "time_hour": hours_from_anchor(t, t0),
            "phase": 0 if t < t0 else 1,
            "modality": "lab",
            "code": code,
            "raw_code": l.get("lab_name"),
            "lab_code": l.get("lab_code"),
            "unit": l.get("unit"),
            "value": value,
        }

        if config.deduplicate_same_event:
            key = (event["time"], event["modality"], event["code"], event["value"])
            if key in seen_event_keys:
                continue
            seen_event_keys.add(key)

        lab_events.append(event)

    # -- medication feature: daily exposure matrix --
    num_bins = int((config.preop_days + config.postop_days) / config.day_bin_size) + 1
    num_meds = len(selected_rxcui_names)
    med_day_vector = [[0 for _ in range(num_meds)] for _ in range(num_bins)]

    # Build medication exposure matrix: med_day_vector[day_idx][med_idx] = 1 if active
    for m in record.get("med", []) or []:
        st = parse_time(m.get("start_time"))
        et = parse_time(m.get("end_time"))
        if st is None:
            continue
        if et is None:
            et = st

        # Determine overlap window
        if config.keep_only_window_overlap_meds:
            if not interval_overlaps(st, et, window_start, window_end):
                continue
        else:
            if not in_window(st, window_start, window_end):
                continue

        rxcui = normalize_rxcui(m.get("rxcui"))
        if rxcui not in rxcui_name_to_idx:
            continue
        med_idx = rxcui_name_to_idx[rxcui]

        # effective medication period within window
        effective_start = max(st, window_start)
        effective_end = min(et, window_end)

        # Determine which days to mark as active
        first_day = math.ceil((effective_start - t0).total_seconds() / 86400.0)
        last_day = math.floor((effective_end - t0).total_seconds() / 86400.0)
        # Bin index: 0 ~ num_bins-1 for day -preop_days ~ postop_days
        for day in range(first_day, last_day + 1):
            bin_idx = day + config.preop_days
            if 0 <= bin_idx < num_bins:
                med_day_vector[bin_idx][med_idx] = 1

    # medication events (if you still want, but NOT repeated for every day)
    for m in record.get("med", []) or []:
        st = parse_time(m.get("start_time"))
        et = parse_time(m.get("end_time"))
        if st is None:
            continue
        if et is None:
            et = st

        if config.keep_only_window_overlap_meds:
            if not interval_overlaps(st, et, window_start, window_end):
                continue
        else:
            if not in_window(st, window_start, window_end):
                continue

        code = map_med_code(m, rxcui_set=rxcui_set)

        effective_start = max(st, window_start)
        if in_window(effective_start, window_start, window_end):
            event = {
                "time": effective_start.isoformat(sep=" "),
                "time_day": days_from_anchor(effective_start, t0),
                "time_hour": hours_from_anchor(effective_start, t0),
                "phase": 0 if effective_start < t0 else 1,
                "modality": "med",
                "event_type": "start",
                "code": code,
                "raw_code": m.get("drug_code"),
                "rxcui": m.get("rxcui"),
                "value": None,
            }
            if config.deduplicate_same_event:
                key = (event["time"], event["modality"], event["event_type"], event["code"])
                if key in seen_event_keys:
                    continue
                seen_event_keys.add(key)
            med_events.append(event)

        if config.include_med_end_event:
            effective_end = min(et, window_end)
            if in_window(effective_end, window_start, window_end):
                event = {
                    "time": effective_end.isoformat(sep=" "),
                    "time_day": days_from_anchor(effective_end, t0),
                    "time_hour": hours_from_anchor(effective_end, t0),
                    "phase": 0 if effective_end < t0 else 1,
                    "modality": "med",
                    "event_type": "end",
                    "code": code,
                    "raw_code": m.get("drug_code"),
                    "rxcui": m.get("rxcui"),
                    "value": None,
                }
                if config.deduplicate_same_event:
                    key = (event["time"], event["modality"], event["event_type"], event["code"])
                    if key in seen_event_keys:
                        continue
                    seen_event_keys.add(key)
                med_events.append(event)

    condition_events.sort(key=lambda x: x["time"])
    lab_events.sort(key=lambda x: x["time"])
    med_events.sort(key=lambda x: x["time"])

    # 移除lab_grid和lab_obs_mask，不再包含lab grid和lab mask
    merged_events = static_tokens + procedure_tokens + condition_events + lab_events + med_events
    merged_events.sort(key=lambda x: x["time"])

    return {
        "person_id": person_id,
        "label_aki": label_aki,
        "aki_time": aki_time.isoformat(sep=" ") if aki_time is not None else None,
        "t0": t0.isoformat(sep=" "),
        "window_start": window_start.isoformat(sep=" "),
        "window_end": window_end.isoformat(sep=" "),
        "preop_days": config.preop_days,
        "postop_days": config.postop_days,
        "static_tokens": static_tokens,
        "procedure_tokens": procedure_tokens,
        "condition_events": condition_events,
        "lab_events": lab_events,
        "med_events": med_events,
        "events": merged_events,
        "selected_lab_names": selected_lab_names,
        "selected_rxcui_names": selected_rxcui_names,
        "med_day_vector": med_day_vector,
    }


def load_json_lines(path: str) -> List[Dict[str, Any]]:
    """
    Read a JSONL file, return list of dicts (one per line).
    """
    path_obj = Path(path)
    records = []
    with path_obj.open("r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
    return records


def build_episodes_from_file(
    input_path: str,
    ccs_list: Sequence[Any],
    rxcui_list: Sequence[Any],
    lab_list: Sequence[str],
    config: PreprocessConfig = PreprocessConfig(),
) -> List[Dict[str, Any]]:
    # The input is a JSONL file with multiple lines of data.
    data = load_json_lines(input_path)

    episodes = []
    for rec in tqdm(data):
        try:
            ep = build_episode_from_record(
                record=rec,
                ccs_list=ccs_list,
                rxcui_list=rxcui_list,
                lab_list=lab_list,
                config=config,
            )
            episodes.append(ep)
        except Exception as e:
            print(f"[WARN] skip person_id={rec.get('person_id')} because: {e}")

    return episodes


def save_jsonl(records: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_pickle(obj: Any, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build perioperative episodes JSONL from raw EHR records.")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input JSONL path (one raw record per line).",
    )
    parser.add_argument(
        "--output_jsonl",
        "-o",
        type=str,
        required=True,
        help="Output path for episodes JSONL.",
    )
    parser.add_argument(
        "--output_pickle",
        type=str,
        default=None,
        help="Optional path to also save episodes as pickle.",
    )
    parser.add_argument(
        "--rxcui_json",
        type=str,
        default="../data/top90_rxcuis.json",
        help="JSON list of RxCUIs for medication coding.",
    )
    parser.add_argument(
        "--lab_json",
        type=str,
        default="../data/lab_codes.json",
        help="JSON list of lab codes for lab events / lab_grid columns.",
    )
    parser.add_argument(
        "--ccs_json",
        type=str,
        default="../data/ccs_codes.json",
        help="JSON list of CCS categories for condition coding.",
    )
    args = parser.parse_args()

    with open(args.rxcui_json, "r", encoding="utf-8") as f:
        rxcui_list = json.load(f)
    with open(args.lab_json, "r", encoding="utf-8") as f:
        lab_list = json.load(f)
    with open(args.ccs_json, "r", encoding="utf-8") as f:
        ccs_list = json.load(f)

    config = PreprocessConfig(
        preop_days=7,
        postop_days=30,
        day_bin_size=1,
        use_ccs_for_condition=True,
        include_med_end_event=True,
        deduplicate_same_event=True,
        lowercase_lab_name=False,
        keep_only_window_overlap_meds=True,
    )

    episodes = build_episodes_from_file(
        input_path=args.input,
        ccs_list=ccs_list,
        rxcui_list=rxcui_list,
        lab_list=lab_list,
        config=config,
    )

    save_jsonl(episodes, args.output_jsonl)
    if args.output_pickle:
        save_pickle(episodes, args.output_pickle)

    print(f"Built {len(episodes)} episodes.")
    print(f"Wrote JSONL: {args.output_jsonl}")
    if args.output_pickle:
        print(f"Wrote pickle: {args.output_pickle}")
