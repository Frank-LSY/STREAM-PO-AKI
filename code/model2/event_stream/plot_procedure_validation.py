#!/usr/bin/env python3
"""Plot procedure counterfactual Δlogit vs observed AKI rate (validation figure)."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranking_csv", type=str, required=True,
                   help="procedure_counterfactual_ranking.csv (or pooled).")
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--procedure_table", type=str,
                   default=str(PROJECT_ROOT / "data/cleaned_procedure_condition.csv"))
    p.add_argument("--output_png", type=str,
                   default=str(PROJECT_ROOT / "result/figures/procedure_counterfactual_validation.png"))
    p.add_argument("--min_obs_n", type=int, default=20)
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def load_region_names(path: Path) -> dict:
    d = pd.read_csv(path, low_memory=False, usecols=["procedure_source_value", "procedure_type"])
    return (
        d.dropna(subset=["procedure_source_value"])
        .groupby("procedure_source_value")["procedure_type"]
        .agg(lambda s: s.mode().iat[0] if len(s.mode()) else "")
        .to_dict()
    )


def observed_aki_rates(episode_path: Path) -> dict:
    obs: dict = {}
    with episode_path.open() as f:
        for line in f:
            ep = json.loads(line)
            pr = [e for e in ep.get("events", []) if str(e.get("modality", "")).lower() == "procedure"]
            if not pr:
                continue
            c = str(pr[0]["code"])
            a = obs.setdefault(c, [0, 0])
            a[0] += int(ep.get("label_aki") or 0)
            a[1] += 1
    return obs


def main() -> None:
    args = parse_args()
    rk = list(csv.DictReader(open(args.ranking_csv)))
    name = load_region_names(Path(args.procedure_table))
    obs = observed_aki_rates(Path(args.episode_path))

    pts = []
    for r in rk:
        c = r["procedure_code"]
        k, n = obs.get(c, [0, 0])
        if n < args.min_obs_n:
            continue
        pts.append(
            (
                float(r["mean_delta_logit_vs_reference"]),
                k / n,
                n,
                name.get(c, "other") or "other",
                c,
            )
        )
    if len(pts) < 5:
        raise SystemExit(f"too few procedures with n>={args.min_obs_n}: {len(pts)}")

    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts]) * 100.0
    rho = spearmanr(x, y)

    cols = {
        "abdomen": "#c0392b",
        "thoracic": "#e67e22",
        "intracranial": "#8e44ad",
        "joint": "#2471a3",
        "spine": "#16a085",
        "skin": "#7f8c8d",
    }
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    for reg, color in cols.items():
        m = [i for i, p in enumerate(pts) if p[3] == reg]
        if not m:
            continue
        ax.scatter(
            x[m], y[m],
            s=[np.sqrt(pts[i][2]) * 3.6 for i in m],
            c=color, alpha=0.78, edgecolors="white", linewidths=0.6,
            label=reg,
        )
    other = [i for i, p in enumerate(pts) if p[3] not in cols]
    if other:
        ax.scatter(
            x[other], y[other],
            s=[np.sqrt(pts[i][2]) * 3.6 for i in other],
            c="#95a5a6", alpha=0.7, edgecolors="white", linewidths=0.6,
            label="other",
        )

    b, a0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, b * xs + a0, "--", color="0.4", lw=1.15, zorder=0)

    snomed_path = (PROJECT_ROOT / "data/snomed_procedure_names.json")
    snomed = json.loads(snomed_path.read_text()) if snomed_path.exists() else {}
    ref_path = Path(args.ranking_csv).resolve().parent / "reference.txt"
    ref_code = ref_path.read_text().strip() if ref_path.exists() else ""

    def proc_label(code: str, width: int = 28) -> str:
        s = snomed.get(code, code.split("|")[-1])
        s = s.replace(" (procedure)", "").strip()
        s = " ".join(s.split())
        return s if len(s) <= width else s[: width - 1] + "…"

    ref_i = next((i for i, p in enumerate(pts) if p[4] == ref_code), None)
    if ref_i is not None:
        ax.annotate(
            f"ref: {proc_label(ref_code, 34)}",
            (x[ref_i], y[ref_i]), fontsize=8.0, style="italic",
            xytext=(5, -10), textcoords="offset points", color="0.2",
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7),
        )

    by_delta = sorted(range(len(pts)), key=lambda i: -abs(pts[i][0]))
    by_rate = sorted(range(len(pts)), key=lambda i: -pts[i][1])
    label_idx: list[int] = []
    for i in by_delta[:3] + by_rate[:3]:
        if i == ref_i:
            continue
        if i not in label_idx:
            label_idx.append(i)
        if len(label_idx) >= 5:
            break
    for i in label_idx:
        ax.annotate(
            proc_label(pts[i][4]), (x[i], y[i]), fontsize=7.5,
            xytext=(3, 3), textcoords="offset points", color="0.3",
        )
    ax.set_xlabel("Counterfactual Δlogit of operation (vs reference)", fontsize=9.0)
    ax.set_ylabel("Observed AKI rate (%)", fontsize=9.0)
    ax.set_title(
        f"Index-procedure substitution  ·  Spearman ρ = {rho.statistic:.3f}",
        fontsize=11.0, weight="bold", loc="left",
    )
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.axvline(0, color="0.7", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.0)
    fig.tight_layout()
    out = Path(args.output_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    print(f"Wrote {out}  (n={len(pts)}  Spearman={rho.statistic:.3f})")


if __name__ == "__main__":
    main()
