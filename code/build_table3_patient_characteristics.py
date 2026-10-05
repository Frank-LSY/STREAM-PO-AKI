#!/usr/bin/env python3
"""Build Table 3 (Patient Characteristics) from five_folds JSONL splits.

Uses train/validation = fold1..fold5, test = mixed.test.jsonl.
Stratifies by label_aki (Negative / Positive).
"""
from __future__ import annotations

import os

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

MODEL2_ROOT = Path(__file__).resolve().parent / "model2"
if str(MODEL2_ROOT) not in sys.path:
    sys.path.insert(0, str(MODEL2_ROOT))

from common.surgery_category_map import surgery_categories_for_episode

PROJECT_ROOT = Path(os.environ.get("PERIOP_ROOT", Path(__file__).resolve().parents[1]))

# OMOP gender / race concept ids (as encoded in SEX|id / RACE|id).
SEX_FEMALE = "SEX|8532"
SEX_MALE = "SEX|8507"

RACE_WHITE = {"RACE|8527"}
RACE_AA = {"RACE|8516"}
# Asian + Asian subtypes (Chinese/Japanese/Korean/Filipino/Vietnamese/Asian Indian, etc.)
RACE_ASIAN = {
    "RACE|8515",
    "RACE|38003574",
    "RACE|38003579",
    "RACE|38003581",
    "RACE|38003584",
    "RACE|38003585",
    "RACE|38003592",
}
# AIAN / NHPI / Polynesian / Other Pacific Islander
RACE_OTHERS = {"RACE|8657", "RACE|8557", "RACE|38003610", "RACE|38003613"}

SURGERY_DISPLAY = {
    "Abdomen": "Abdominal",
    "Intracranial": "Intracranial",
    "Thoracic": "Thoracic",
    "Joint": "Joint",
    "Spine": "Spinal",
    "Skin": "Skin",
}
SURGERY_ORDER = ["Abdominal", "Intracranial", "Thoracic", "Joint", "Spinal", "Skin"]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def static_code(ep: Dict[str, Any], prefix: str) -> Optional[str]:
    for tok in ep.get("static_tokens") or []:
        code = str(tok.get("code", ""))
        if code.startswith(prefix):
            return code
    return None


def age_of(ep: Dict[str, Any]) -> Optional[float]:
    for tok in ep.get("static_tokens") or []:
        if tok.get("code") == "AGE" and tok.get("value") is not None:
            try:
                return float(tok["value"])
            except (TypeError, ValueError):
                return None
    return None


def gender_bucket(sex: Optional[str]) -> str:
    if sex == SEX_FEMALE:
        return "Female"
    if sex == SEX_MALE:
        return "Male"
    return "Missing"


def race_bucket(race: Optional[str]) -> str:
    if race in RACE_WHITE:
        return "White"
    if race in RACE_AA:
        return "African American"
    if race in RACE_ASIAN:
        return "Asian"
    if race in RACE_OTHERS:
        return "Others"
    return "Unknown"


def surgery_bucket(ep: Dict[str, Any]) -> str:
    """Assign one surgery type for table counting (prefer listed categories)."""
    cats = surgery_categories_for_episode(ep)
    priority = ["Abdomen", "Intracranial", "Thoracic", "Joint", "Spine", "Skin"]
    for key in priority:
        if key in cats:
            return SURGERY_DISPLAY[key]
    return "Other/Unknown"


def event_counts(ep: Dict[str, Any]) -> Tuple[int, int, int]:
    n_cond = len(ep.get("condition_events") or [])
    n_med = len(ep.get("med_events") or [])
    n_lab = len(ep.get("lab_events") or [])
    return n_cond, n_med, n_lab


def fmt_int(n: int) -> str:
    return f"{n:,}"


def fmt_age_iqr(ages: List[float]) -> str:
    if not ages:
        return "—"
    a = np.asarray(ages, dtype=np.float64)
    med, q1, q3 = np.percentile(a, [50, 25, 75])
    return f"{med:.0f} ({q1:.0f}, {q3:.0f})"


def fmt_mean(xs: List[float]) -> str:
    if not xs:
        return "—"
    return f"{float(np.mean(xs)):.2f}"


def summarize_group(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    gender: Dict[str, int] = defaultdict(int)
    race: Dict[str, int] = defaultdict(int)
    surgery: Dict[str, int] = defaultdict(int)
    ages: List[float] = []
    conds: List[float] = []
    meds: List[float] = []
    labs: List[float] = []

    for ep in eps:
        gender[gender_bucket(static_code(ep, "SEX|"))] += 1
        race[race_bucket(static_code(ep, "RACE|"))] += 1
        surgery[surgery_bucket(ep)] += 1
        age = age_of(ep)
        if age is not None:
            ages.append(age)
        c, m, l = event_counts(ep)
        conds.append(float(c))
        meds.append(float(m))
        labs.append(float(l))

    return {
        "n": len(eps),
        "gender": dict(gender),
        "race": dict(race),
        "surgery": dict(surgery),
        "age_iqr": fmt_age_iqr(ages),
        "mean_conditions": fmt_mean(conds),
        "mean_medications": fmt_mean(meds),
        "mean_labs": fmt_mean(labs),
    }


def load_split(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    train_val: List[Dict[str, Any]] = []
    for k in range(1, 6):
        path = data_dir / f"7d_window_clm.mixed.fold{k}.jsonl"
        print(f"Loading {path} ...")
        train_val.extend(iter_jsonl(path))
    test_path = data_dir / "7d_window_clm.mixed.test.jsonl"
    print(f"Loading {test_path} ...")
    test = list(iter_jsonl(test_path))
    return train_val, test


def split_by_label(eps: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    neg = [e for e in eps if int(e.get("label_aki") or 0) == 0]
    pos = [e for e in eps if int(e.get("label_aki") or 0) == 1]
    return neg, pos


def build_table_rows(
    tv_neg: Dict[str, Any],
    tv_pos: Dict[str, Any],
    te_neg: Dict[str, Any],
    te_pos: Dict[str, Any],
) -> List[Dict[str, str]]:
    groups = (tv_neg, tv_pos, te_neg, te_pos)

    def row(characteristic: str, values: Tuple[str, str, str, str]) -> Dict[str, str]:
        return {
            "characteristic": characteristic,
            "train_val_negative": values[0],
            "train_val_positive": values[1],
            "test_negative": values[2],
            "test_positive": values[3],
        }

    rows: List[Dict[str, str]] = []
    rows.append(row("N", tuple(fmt_int(int(g["n"])) for g in groups)))

    for gname in ("Female", "Male", "Missing"):
        rows.append(
            row(
                f"Gender — {gname}",
                tuple(fmt_int(int(g["gender"].get(gname, 0))) for g in groups),
            )
        )

    rows.append(row("Age at Surgery", tuple(str(g["age_iqr"]) for g in groups)))

    for rname in ("White", "African American", "Asian", "Others", "Unknown"):
        rows.append(
            row(
                f"Race — {rname}",
                tuple(fmt_int(int(g["race"].get(rname, 0))) for g in groups),
            )
        )

    for sname in SURGERY_ORDER:
        rows.append(
            row(
                f"Surgery Types — {sname}",
                tuple(fmt_int(int(g["surgery"].get(sname, 0))) for g in groups),
            )
        )
    rows.append(
        row(
            "Surgery Types — Other/Unknown",
            tuple(fmt_int(int(g["surgery"].get("Other/Unknown", 0))) for g in groups),
        )
    )

    rows.append(
        row(
            "Average Events — Conditions",
            tuple(str(g["mean_conditions"]) for g in groups),
        )
    )
    rows.append(
        row(
            "Average Events — Medications",
            tuple(str(g["mean_medications"]) for g in groups),
        )
    )
    rows.append(
        row(
            "Average Events — Lab Tests",
            tuple(str(g["mean_labs"]) for g in groups),
        )
    )
    return rows


def write_markdown(
    path: Path,
    n_tv: int,
    n_te: int,
    tv_neg_n: int,
    tv_pos_n: int,
    te_neg_n: int,
    te_pos_n: int,
    rows: List[Dict[str, str]],
) -> None:
    lines = [
        "# Table 3. Patient Characteristics",
        "",
        f"**Training/Validation Set ({n_tv:,})** | **Test Set ({n_te:,})**",
        "",
        f"Negative ({tv_neg_n:,}) | Positive ({tv_pos_n:,}) | Negative ({te_neg_n:,}) | Positive ({te_pos_n:,})",
        "",
        "| Characteristic | Train/Val Negative | Train/Val Positive | Test Negative | Test Positive |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        if r["characteristic"] == "N":
            continue
        lines.append(
            f"| {r['characteristic']} | {r['train_val_negative']} | {r['train_val_positive']} | "
            f"{r['test_negative']} | {r['test_positive']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "- Asian includes OMOP Asian + Asian subtypes (e.g. Chinese, Japanese, Korean, Filipino, Vietnamese, Asian Indian).",
            "- Others includes American Indian/Alaska Native, Native Hawaiian/Other Pacific Islander.",
            "- Gender Missing should be near zero after excluding SEX|0.",
            "- Surgery type: one primary category per episode (priority Abdominal > Intracranial > Thoracic > Joint > Spinal > Skin).",
            "- Source: five_folds_deduplicate4 (SEX|0 removed).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data_dir",
        type=Path,
        default=(PROJECT_ROOT / "data/five_folds_deduplicate4"),
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=(PROJECT_ROOT / "result/tables"),
    )
    args = p.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    train_val, test = load_split(args.data_dir)
    tv_neg_eps, tv_pos_eps = split_by_label(train_val)
    te_neg_eps, te_pos_eps = split_by_label(test)

    print("Summarizing groups ...")
    tv_neg = summarize_group(tv_neg_eps)
    tv_pos = summarize_group(tv_pos_eps)
    te_neg = summarize_group(te_neg_eps)
    te_pos = summarize_group(te_pos_eps)

    rows = build_table_rows(tv_neg, tv_pos, te_neg, te_pos)

    csv_path = out_dir / "table3_patient_characteristics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "characteristic",
                "train_val_negative",
                "train_val_positive",
                "test_negative",
                "test_positive",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    md_path = out_dir / "table3_patient_characteristics.md"
    write_markdown(
        md_path,
        n_tv=len(train_val),
        n_te=len(test),
        tv_neg_n=int(tv_neg["n"]),
        tv_pos_n=int(tv_pos["n"]),
        te_neg_n=int(te_neg["n"]),
        te_pos_n=int(te_pos["n"]),
        rows=rows,
    )

    summary = {
        "data_dir": str(args.data_dir),
        "train_val_n": len(train_val),
        "test_n": len(test),
        "train_val_negative": tv_neg,
        "train_val_positive": tv_pos,
        "test_negative": te_neg,
        "test_positive": te_pos,
    }
    json_path = out_dir / "table3_patient_characteristics.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
