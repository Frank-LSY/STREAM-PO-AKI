#!/usr/bin/env python3
"""Build static preoperative tabular datasets for XGBoost from five_folds JSONL."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from common.fold_data import default_data_dir, default_xgb_data_dir, load_jsonl, resolve_fold_paths
from common.utils import save_json

EPS = 1e-6
PREOP_TIME_CUTOFF = 0.0  # time_hour < 0 for cond/lab/med


def _sanitize_token(token: str) -> str:
    return (
        str(token)
        .replace("|", "_")
        .replace("/", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def _lab_col_code(lab_code: int) -> str:
    return f"LAB_{int(lab_code)}"


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

        for ev in ep.get("condition_events", []):
            th = ev.get("time_hour")
            if th is not None and float(th) < PREOP_TIME_CUTOFF:
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


def fit_preop_lab_zscore(episodes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_code: Dict[str, List[float]] = defaultdict(list)
    all_vals: List[float] = []

    for ep in episodes:
        for ev in ep.get("lab_events", []):
            th = ev.get("time_hour")
            if th is None or float(th) >= PREOP_TIME_CUTOFF:
                continue
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
        mean = float(arr.mean())
        std = float(arr.std())
        if std < EPS or not np.isfinite(std):
            std = 1.0
        per_code[code] = {"mean": mean, "std": std}

    if all_vals:
        arr = np.asarray(all_vals, dtype=np.float64)
        g_mean = float(arr.mean())
        g_std = float(arr.std())
        if g_std < EPS or not np.isfinite(g_std):
            g_std = 1.0
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
    if s < EPS or not np.isfinite(s):
        s = 1.0
    z = (float(raw) - m) / s
    return float(z) if np.isfinite(z) else 0.0


def build_feature_columns(vocab: Dict[str, Any]) -> List[str]:
    cols = ["age"]
    cols.extend(f"sex_{_sanitize_token(c)}" for c in vocab["sex_codes"])
    cols.extend(f"race_{_sanitize_token(c)}" for c in vocab["race_codes"])
    cols.extend(f"proc_{_sanitize_token(c)}" for c in vocab["procedure_codes"])
    cols.extend(f"ccs_{_sanitize_token(c)}" for c in vocab["ccs_codes"])
    for lab_code in vocab["lab_codes"]:
        base = _lab_col_code(lab_code)
        cols.append(f"{base}_value")
        cols.append(f"{base}_missing")
    cols.extend(f"med_{_sanitize_token(c)}" for c in vocab["rxcui_codes"])
    return cols


def _static_age(ep: Dict[str, Any]) -> float:
    for tok in ep.get("static_tokens", []):
        if str(tok.get("code")) == "AGE":
            v = tok.get("value")
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
    return 0.0


def _static_one_hot(ep: Dict[str, Any], prefix: str, vocab_codes: Sequence[str], col_prefix: str) -> Dict[str, float]:
    found = None
    for tok in ep.get("static_tokens", []):
        code = str(tok.get("code", ""))
        if code.startswith(prefix):
            found = code
            break
    out: Dict[str, float] = {f"{col_prefix}_{_sanitize_token(c)}": 0.0 for c in vocab_codes}
    if found is not None and found in vocab_codes:
        out[f"{col_prefix}_{_sanitize_token(found)}"] = 1.0
    return out


def _preop_med_max(ep: Dict[str, Any], num_preop_bins: int) -> List[float]:
    mat = ep.get("med_day_vector") or []
    if not mat:
        return []
    n_meds = len(mat[0]) if mat else 0
    if n_meds == 0:
        return []
    preop_rows = mat[: min(num_preop_bins, len(mat))]
    if not preop_rows:
        return [0.0] * n_meds
    arr = np.asarray(preop_rows, dtype=np.float64)
    return arr.max(axis=0).tolist()


def featurize_episode(
    ep: Dict[str, Any],
    vocab: Dict[str, Any],
    feature_columns: Sequence[str],
    lab_stats: Dict[str, Any],
) -> Dict[str, float]:
    row: Dict[str, float] = {c: 0.0 for c in feature_columns}
    row["age"] = _static_age(ep)

    row.update(_static_one_hot(ep, "SEX|", vocab["sex_codes"], "sex"))
    row.update(_static_one_hot(ep, "RACE|", vocab["race_codes"], "race"))

    for tok in ep.get("procedure_tokens", []):
        code = str(tok.get("code", "PROC_OTHERS"))
        col = f"proc_{_sanitize_token(code)}"
        if col in row:
            row[col] = 1.0

    for ev in ep.get("condition_events", []):
        th = ev.get("time_hour")
        if th is None or float(th) >= PREOP_TIME_CUTOFF:
            continue
        code = str(ev.get("code", "COND_OTHERS"))
        col = f"ccs_{_sanitize_token(code)}"
        if col in row:
            row[col] = 1.0

    last_preop_lab: Dict[int, float] = {}
    for ev in ep.get("lab_events", []):
        th = ev.get("time_hour")
        if th is None or float(th) >= PREOP_TIME_CUTOFF:
            continue
        code = str(ev.get("code", ""))
        if not code.startswith("LAB|"):
            continue
        try:
            lab_code = int(code.split("|", 1)[1])
        except (IndexError, ValueError):
            continue
        v = ev.get("value")
        if v is None:
            continue
        last_preop_lab[lab_code] = float(v)

    for lab_code in vocab["lab_codes"]:
        base = _lab_col_code(lab_code)
        val_col = f"{base}_value"
        miss_col = f"{base}_missing"
        if lab_code in last_preop_lab:
            row[miss_col] = 1.0
            row[val_col] = apply_lab_zscore(
                last_preop_lab[lab_code], _lab_event_code(lab_code), lab_stats
            )
        else:
            row[miss_col] = 0.0
            row[val_col] = 0.0

    preop_days = int(ep.get("preop_days", 7))
    med_max = _preop_med_max(ep, preop_days)
    rxcui_list = vocab["rxcui_codes"]
    for i, rxcui in enumerate(rxcui_list):
        col = f"med_{_sanitize_token(rxcui)}"
        if col in row and i < len(med_max):
            row[col] = float(med_max[i])

    return row


def episodes_to_dataframe(
    episodes: Sequence[Dict[str, Any]],
    vocab: Dict[str, Any],
    feature_columns: Sequence[str],
    lab_stats: Dict[str, Any],
    source_fold: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for ep in episodes:
        feats = featurize_episode(ep, vocab, feature_columns, lab_stats)
        rec = {
            "person_id": str(ep.get("person_id", "")),
            "label_aki": int(ep.get("label_aki", 0)),
        }
        if source_fold is not None:
            rec["source_fold"] = int(source_fold)
        rec.update(feats)
        rows.append(rec)

    df = pd.DataFrame(rows)
    col_order = ["person_id", "label_aki"]
    if source_fold is not None:
        col_order.append("source_fold")
    col_order.extend(feature_columns)
    return df[col_order]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build five_folds_xgboost Parquet splits from JSONL.")
    p.add_argument(
        "--folds_dir",
        type=str,
        default=None,
        help="JSONL folds root (default: DATA_DIR / common.fold_data.default_data_dir).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Parquet output root (default: XGB_DATA_DIR / five_folds_xgboost).",
    )
    p.add_argument("--fold_filename_template", type=str, default="7d_window_clm.mixed.fold{}.jsonl")
    p.add_argument("--test_filename", type=str, default="7d_window_clm.mixed.test.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folds_dir = Path(args.folds_dir or default_data_dir())
    output_dir = Path(args.output_dir or default_xgb_data_dir())
    fold_paths = resolve_fold_paths(folds_dir, args.fold_filename_template)
    test_path = folds_dir / args.test_filename

    print("Loading folds 1-5 for vocab and lab z-score...")
    all_train_episodes: List[Dict[str, Any]] = []
    fold_episodes: Dict[int, List[Dict[str, Any]]] = {}
    for k in range(1, 6):
        eps = load_jsonl(fold_paths[k])
        fold_episodes[k] = eps
        all_train_episodes.extend(eps)
        print(f"  fold{k}: {len(eps)} episodes")

    vocab = collect_vocab(all_train_episodes)
    lab_stats = fit_preop_lab_zscore(all_train_episodes)
    feature_columns = build_feature_columns(vocab)

    vocab_dir = output_dir / "vocab"
    save_json(vocab["ccs_codes"], vocab_dir / "ccs_codes.json")
    save_json(vocab["procedure_codes"], vocab_dir / "procedure_codes.json")
    save_json(vocab["sex_codes"], vocab_dir / "sex_codes.json")
    save_json(vocab["race_codes"], vocab_dir / "race_codes.json")
    save_json(vocab["lab_codes"], vocab_dir / "lab_codes.json")
    save_json(vocab["rxcui_codes"], vocab_dir / "rxcui_codes.json")
    save_json(lab_stats, vocab_dir / "lab_zscore.json")
    save_json(feature_columns, vocab_dir / "feature_columns.json")

    splits_meta: Dict[str, Any] = {}
    for val_fold in range(1, 6):
        train_eps: List[Dict[str, Any]] = []
        for k in range(1, 6):
            if k == val_fold:
                continue
            train_eps.extend(fold_episodes[k])
        val_eps = fold_episodes[val_fold]

        train_df = episodes_to_dataframe(train_eps, vocab, feature_columns, lab_stats)
        val_df = episodes_to_dataframe(val_eps, vocab, feature_columns, lab_stats, source_fold=val_fold)

        split_dir = output_dir / "splits" / f"val_fold_{val_fold}"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_path = split_dir / "train.parquet"
        val_path = split_dir / "val.parquet"
        train_df.to_parquet(train_path, index=False)
        val_df.to_parquet(val_path, index=False)

        train_folds = [k for k in range(1, 6) if k != val_fold]
        splits_meta[f"val_fold_{val_fold}"] = {
            "val_fold": val_fold,
            "train_folds": train_folds,
            "train_paths": [str(fold_paths[k]) for k in train_folds],
            "val_path": str(fold_paths[val_fold]),
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "train_parquet": str(train_path),
            "val_parquet": str(val_path),
        }
        print(f"val_fold_{val_fold}: train={len(train_df)} val={len(val_df)}")

    test_eps = load_jsonl(test_path)
    test_df = episodes_to_dataframe(test_eps, vocab, feature_columns, lab_stats)
    test_out = output_dir / "test.parquet"
    test_df.to_parquet(test_out, index=False)
    print(f"test: {len(test_df)} -> {test_out}")

    pos_union = sum(int(ep.get("label_aki", 0)) for ep in all_train_episodes)
    manifest = {
        "source_dir": str(folds_dir),
        "output_dir": str(output_dir),
        "protocol": {
            "preop_days": 7,
            "preop_time_rule": "time_hour < 0 for condition/lab/med",
            "static_and_procedure": "always included",
            "lab_value": "last preoperative value, z-scored",
            "lab_zscore_fit": "all folds 1-5 union (preop labs only)",
            "med_value": "max daily exposure over preoperative med_day_vector bins",
            "vocab_scope": "folds 1-5 union (test excluded)",
            "split_protocol": "same as model2 train_online --val_fold",
        },
        "n_features": len(feature_columns),
        "feature_columns_path": str(vocab_dir / "feature_columns.json"),
        "episode_counts": {
            **{f"fold{k}": len(fold_episodes[k]) for k in range(1, 6)},
            "test": len(test_eps),
        },
        "label_aki_positive_rate_folds_1_5": pos_union / max(len(all_train_episodes), 1),
        "splits": splits_meta,
        "test_parquet": str(test_out),
    }
    save_json(manifest, output_dir / "manifest.json")
    save_json(splits_meta, output_dir / "splits.json")
    print(f"Done. {len(feature_columns)} features -> {output_dir}")


if __name__ == "__main__":
    main()
