#!/usr/bin/env python3
"""Build five_folds_deduplicate4: same split protocol as deduplicate2/3, excluding SEX|0.

Pipeline (matches process_data.ipynb):
  1. Load pos/neg JSONL
  2. Filter: age>=18; drop aki_time==t0 and aki_time<t0; drop SEX|0
  3. Time-split by episode t0 (80/20) with no person_id leakage
  4. 5-fold by person_id on train pool (seed=42); fold shuffle seed=10086
  5. Test: keep earliest t0 per person_id
"""
from __future__ import annotations

import os

import argparse
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = Path(os.environ.get("PERIOP_ROOT", Path(__file__).resolve().parents[1]))


def parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    if isinstance(x, datetime):
        return x
    s = str(x).strip()
    if not s:
        return None
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def age_of(ep: Dict[str, Any]) -> Optional[float]:
    for token in ep.get("static_tokens") or []:
        if token.get("code") == "AGE" and token.get("value") is not None:
            try:
                return float(token["value"])
            except (TypeError, ValueError):
                return None
    age = ep.get("age")
    if age is None:
        return None
    try:
        return float(age)
    except (TypeError, ValueError):
        return None


def sex_code_of(ep: Dict[str, Any]) -> Optional[str]:
    for token in ep.get("static_tokens") or []:
        code = str(token.get("code", ""))
        if code.startswith("SEX|"):
            return code
    for event in ep.get("events") or []:
        code = str(event.get("code", ""))
        if code.startswith("SEX|"):
            return code
    return None


def keep_episode(ep: Dict[str, Any], drop_sex_codes: Tuple[str, ...]) -> Tuple[bool, str]:
    age = age_of(ep)
    if age is None or age < 18:
        return False, "age_lt_18_or_missing"

    sex = sex_code_of(ep)
    if sex in drop_sex_codes:
        return False, f"drop_{sex}"

    aki_time = parse_dt(ep.get("aki_time"))
    t0 = parse_dt(ep.get("t0"))
    if t0 is None:
        return False, "missing_t0"
    if aki_time is not None:
        if aki_time == t0:
            return False, "aki_time_eq_t0"
        if aki_time < t0:
            return False, "aki_time_lt_t0"
    return True, "keep"


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_filtered(
    path: Path,
    drop_sex_codes: Tuple[str, ...],
    desc: str,
) -> Tuple[List[Dict[str, Any]], Counter]:
    kept: List[Dict[str, Any]] = []
    reason = Counter()
    for ep in tqdm(iter_jsonl(path), desc=desc):
        ok, why = keep_episode(ep, drop_sex_codes)
        reason[why] += 1
        if ok:
            kept.append(ep)
    return kept, reason


def time_split_by_episode(
    all_episodes: List[Dict[str, Any]],
    val_ratio: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    episodes_with_time = [(ep, parse_dt(ep.get("t0"))) for ep in all_episodes]
    episodes_with_time = [(ep, t) for ep, t in episodes_with_time if t is not None]
    episodes_with_time.sort(key=lambda x: x[1])

    total = len(episodes_with_time)
    split_idx = int(total * (1 - val_ratio))
    train_candidates = episodes_with_time[:split_idx]
    val_candidates = episodes_with_time[split_idx:]

    train_person_ids = {ep["person_id"] for ep, _ in train_candidates}
    val_episodes = [ep for ep, _ in val_candidates if ep["person_id"] not in train_person_ids]
    val_person_ids = {ep["person_id"] for ep in val_episodes}
    train_episodes = [ep for ep, _ in train_candidates if ep["person_id"] not in val_person_ids]
    return train_episodes, val_episodes


def split_folds_by_pid_all_labels(
    data: List[Dict[str, Any]],
    n_folds: int = 5,
    seed: int = 42,
    pid_key: str = "person_id",
) -> List[List[Dict[str, Any]]]:
    pid_to_episodes: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for ep in data:
        pid_to_episodes[ep[pid_key]].append(ep)
    all_pids = list(pid_to_episodes.keys())
    random.Random(seed).shuffle(all_pids)

    fold_pids: List[List[Any]] = [[] for _ in range(n_folds)]
    for idx, pid in enumerate(all_pids):
        fold_pids[idx % n_folds].append(pid)

    folds: List[List[Dict[str, Any]]] = []
    for pids in fold_pids:
        fold: List[Dict[str, Any]] = []
        for pid in pids:
            fold.extend(pid_to_episodes[pid])
        folds.append(fold)
    return folds


def earliest_per_person(episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Any, Dict[str, Any]] = {}
    for ep in episodes:
        pid = ep.get("person_id")
        t0 = parse_dt(ep.get("t0"))
        if pid is None or t0 is None:
            continue
        prev = best.get(pid)
        if prev is None or t0 < parse_dt(prev.get("t0")):  # type: ignore[operator]
            best[pid] = ep
    return list(best.values())


def summarize(name: str, eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    sex = Counter(sex_code_of(ep) for ep in eps)
    n_pos = sum(int(ep.get("label_aki") or 0) for ep in eps)
    return {
        "name": name,
        "n": len(eps),
        "n_pos": n_pos,
        "n_neg": len(eps) - n_pos,
        "pos_rate": (n_pos / len(eps)) if eps else None,
        "n_person": len({ep["person_id"] for ep in eps}),
        "sex": dict(sex),
    }


def write_jsonl(path: Path, eps: List[Dict[str, Any]], desc: str) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for ep in tqdm(eps, desc=desc):
            fout.write(json.dumps(ep, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pos_path",
        type=Path,
        default=(PROJECT_ROOT / "data/7d_window_clm_deduepi.jsonl"),
    )
    p.add_argument(
        "--neg_path",
        type=Path,
        default=(PROJECT_ROOT / "data/7d_window_clm_no_poi_deduepi.jsonl"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=(PROJECT_ROOT / "data/five_folds_deduplicate4"),
    )
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--fold_seed", type=int, default=42)
    p.add_argument("--val_shuffle_seed", type=int, default=2023)
    p.add_argument("--fold_shuffle_seed", type=int, default=10086)
    p.add_argument(
        "--drop_sex",
        nargs="+",
        default=["SEX|0"],
        help="SEX codes to exclude (default: SEX|0).",
    )
    args = p.parse_args()

    drop_sex_codes = tuple(args.drop_sex)
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pos_data, pos_reason = load_filtered(args.pos_path, drop_sex_codes, "filter pos")
    neg_data, neg_reason = load_filtered(args.neg_path, drop_sex_codes, "filter neg")

    all_data = pos_data + neg_data
    train_set, val_set = time_split_by_episode(all_data, val_ratio=args.val_ratio)

    train_pids = {ep["person_id"] for ep in train_set}
    val_pids = {ep["person_id"] for ep in val_set}
    assert not (train_pids & val_pids), "person_id leakage between train and held-out"

    random.Random(args.val_shuffle_seed).shuffle(val_set)
    val_set = earliest_per_person(val_set)
    random.Random(args.val_shuffle_seed).shuffle(val_set)

    train_folds = split_folds_by_pid_all_labels(
        train_set, n_folds=args.n_folds, seed=args.fold_seed
    )

    for i, fold in enumerate(train_folds):
        random.Random(args.fold_shuffle_seed).shuffle(fold)
        write_jsonl(out_dir / f"7d_window_clm.mixed.fold{i + 1}.jsonl", fold, f"write fold{i+1}")

    write_jsonl(out_dir / "7d_window_clm.mixed.test.jsonl", val_set, "write test")

    summary = {
        "output_dir": str(out_dir),
        "pos_path": str(args.pos_path),
        "neg_path": str(args.neg_path),
        "drop_sex": list(drop_sex_codes),
        "val_ratio": args.val_ratio,
        "n_folds": args.n_folds,
        "fold_seed": args.fold_seed,
        "val_shuffle_seed": args.val_shuffle_seed,
        "fold_shuffle_seed": args.fold_shuffle_seed,
        "filter_pos_reasons": dict(pos_reason),
        "filter_neg_reasons": dict(neg_reason),
        "after_filter": {
            "pos": summarize("pos", pos_data),
            "neg": summarize("neg", neg_data),
            "all": summarize("all", all_data),
        },
        "splits": {
            **{f"fold{i+1}": summarize(f"fold{i+1}", fold) for i, fold in enumerate(train_folds)},
            "test": summarize("test", val_set),
        },
        "notes": [
            "Time split by episode t0 (earliest 80% train pool, latest 20% held-out).",
            "No shared person_id across train pool and held-out.",
            "5 folds by person_id on train pool.",
            "Test keeps earliest t0 per person_id.",
            "Excluded SEX|0 (and any --drop_sex codes) before split.",
        ],
    }
    summary_path = out_dir / "split_manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary["splits"], indent=2))
    print(f"Wrote {out_dir}")
    print(f"Manifest: {summary_path}")


if __name__ == "__main__":
    main()
