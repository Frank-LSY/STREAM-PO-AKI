#!/usr/bin/env python3
"""Composite interpretability figure: age + procedure validation + event matrix.

Layout
  A  age dose-response
  B  procedure counterfactual validation scatter (drawn natively)
  C  event-contribution dot matrix (time × event); columns ranked by
     postoperative |peak Δ| (preoperative cells still shown)
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import fig_style
from event_stream.plot_contribution_main import (
    BLUE,
    EVENT,
    POST,
    PRE,
    RED,
    nm,
)
from event_stream.plot_procedure_validation import (
    load_region_names,
    observed_aki_rates,
)
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "result/figures/interpretability_main.png"
AGE_CSV = ROOT / "result/event_stream/transformer/age_doseresponse/age_doseresponse_pooled.csv"
PROC_RANK = (
    ROOT / "result/event_stream/transformer"
    / "procedure_effects_pooled"
    / "procedure_counterfactual_ranking.csv"
)
PROC_REF = (
    ROOT / "result/event_stream/transformer"
    / "procedure_effects_pooled"
    / "reference.txt"
)
PROC_TABLE = ROOT / "data/cleaned_procedure_condition.csv"
EPISODE = ROOT / "data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl"
CF = ROOT / "result/event_stream/transformer/contribution_5fold_full"
REF_AGE = 48.0
PROC_NAMES = ROOT / "data/snomed_procedure_names.json"
MIN_OBS_N = 20
MAX_PROC_LABELS = 5

# Hand-placed labels for the crowded upper-right cluster in B: (x, y, ha, va) in data units.
PROC_DISPLAY_NAMES = {
    "SNOMED|235159007": "Percutaneous endoscopic insertion\nof gastrostomy tube",
}
PROC_LABEL_POS = {
    "SNOMED|58380004": (0.92, 41.3, "right", "center"),
    "SNOMED|235159007": (1.36, 45.0, "right", "bottom"),
    "SNOMED|107944001": (1.38, 33.0, "right", "top"),
    "SNOMED|74770008": (1.82, 25.0, "right", "top"),
    "SNOMED|309431009": (-1.30, 29.0, "left", "bottom", (0.0, 0.0)),
    "SNOMED|699120002": (-1.04, 24.5, "left", "bottom", (0.0, 0.0)),
    "SNOMED|77543007": (-0.98, 20.0, "left", "bottom", (0.0, 0.0)),
    "SNOMED|11466000": (-0.62, -4.0, "left", "top", (0.0, 1.0)),
}
# Low-incidence procedures labelled in addition to the extremes.
PROC_LOW_LABELS = ["SNOMED|309431009", "SNOMED|699120002", "SNOMED|77543007", "SNOMED|11466000"]

# Column selection for panel C: top-K by postoperative |peak|; pre cells still drawn.
MAX_EVENT_COLS = 20

# Official HCUP CCS wording (punctuation is stripped in the lookup table).
C_DISPLAY_NAMES = {
    "CCS|131": "Respiratory failure; insufficiency; arrest (adult)",
    "CCS|115": "Aortic, peripheral, and visceral artery aneurysms",
    "CCS|205": "Spondylosis; intervertebral disc disorders; other back problems",
    "CCS|182": "Hemorrhage during pregnancy; abruptio placenta; placenta previa",
    "CCS|108": "Congestive heart failure; nonhypertensive",
}

# Shared typography / chrome across A, B, C
FS_AXIS = 9.0
FS_TICK = 8.0
FS_NOTE = 7.5
FS_LEGEND = 9.5
FS_C_NAME, FS_C_CODE = 6.8, 5.8
LW_MAIN = 2.0
GRID = dict(alpha=0.22, linewidth=0.6, color="0.75")

REGION_COLS = {
    "abdomen": "#c0392b",
    "thoracic": "#e67e22",
    "intracranial": "#8e44ad",
    "joint": "#2471a3",
    "spine": "#16a085",
    "skin": "#7f8c8d",
}

YLAB = [
    "\u22127 d to \u221248 h",
    "\u221248 h to 0",
    "0\u20136 h",
    "6\u201324 h",
    "24\u201372 h",
    "72 h\u20137 d",
]
YPOS = [0.0, 1.0, 2.5, 3.5, 4.5, 5.5]


def _style_axes(ax, *, grid: bool = True) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("0.35")
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(labelsize=FS_TICK, length=3.0, color="0.45", labelcolor="0.2")
    if grid:
        ax.grid(**GRID)
        ax.set_axisbelow(True)


def draw_age(ax) -> None:
    pooled = pd.read_csv(AGE_CSV)
    if {"fold_min", "fold_max"}.issubset(pooled.columns):
        ax.fill_between(
            pooled.age, pooled.fold_min, pooled.fold_max,
            color="0.75", alpha=0.40, lw=0, zorder=1,
        )
    ax.plot(pooled.age, pooled.mean_delta_logit, color=RED, lw=LW_MAIN, marker="o",
            ms=4.2, zorder=3)
    ax.axhline(0, color="0.4", lw=0.85)
    ax.axvline(REF_AGE, color="0.4", lw=0.85, ls=(0, (4, 3)))
    ymin = float(min(pooled.fold_min.min() if "fold_min" in pooled else pooled.ci_lo.min(),
                      pooled.ci_lo.min()))
    ax.text(REF_AGE, ymin, f"  ref: median {REF_AGE:.0f}", fontsize=FS_NOTE, style="italic",
            color="0.4", va="bottom")
    ax.set_xlabel("Substituted age (years)", fontsize=FS_AXIS)
    ax.set_ylabel(f"Change in baseline PO-AKI logit vs age {REF_AGE:.0f}", fontsize=FS_AXIS)
    fig_style.panel_title(ax, "A", "Age substitution", pad=7)
    _style_axes(ax)
    ax.legend(
        handles=[
            Line2D([], [], color=RED, lw=LW_MAIN, marker="o", ms=4.5,
                   label="Mean \u0394logit (5-fold pooled)"),
            Patch(facecolor="0.70", edgecolor="0.55", alpha=0.55, linewidth=0.6,
                  label="Fold minimum\u2013maximum"),
        ],
        loc="upper left", frameon=True, fancybox=False,
        edgecolor="0.85", framealpha=0.92, fontsize=FS_NOTE + 0.5,
        handlelength=1.8, handletextpad=0.55, borderaxespad=0.5, labelspacing=0.35,
    )


def draw_procedure_validation(ax) -> None:
    """Native scatter so fonts / spines match A and C (no embedded PNG)."""
    if not PROC_RANK.exists():
        ax.text(0.5, 0.5, "missing procedure ranking CSV",
                ha="center", va="center", color="0.5", fontsize=FS_AXIS)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    rk = pd.read_csv(PROC_RANK)
    name = load_region_names(PROC_TABLE)
    obs = observed_aki_rates(EPISODE)

    pts = []
    for _, r in rk.iterrows():
        c = str(r["procedure_code"])
        k, n = obs.get(c, [0, 0])
        if n < MIN_OBS_N:
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
    x = np.array([p[0] for p in pts])
    y = np.array([p[1] for p in pts]) * 100.0
    rho = spearmanr(x, y)

    for reg, color in REGION_COLS.items():
        m = [i for i, p in enumerate(pts) if p[3] == reg]
        if not m:
            continue
        ax.scatter(
            x[m], y[m],
            s=[np.sqrt(pts[i][2]) * 3.6 for i in m],
            c=color, alpha=0.78, edgecolors="white", linewidths=0.6,
            label=reg, zorder=3,
        )
    other = [i for i, p in enumerate(pts) if p[3] not in REGION_COLS]
    if other:
        ax.scatter(
            x[other], y[other],
            s=[np.sqrt(pts[i][2]) * 3.6 for i in other],
            c="#95a5a6", alpha=0.7, edgecolors="white", linewidths=0.6,
            label="other", zorder=3,
        )

    b, a0 = np.polyfit(x, y, 1)
    xs = np.linspace(float(x.min()), float(x.max()), 50)
    ax.plot(xs, b * xs + a0, "--", color="0.4", lw=1.15, zorder=1)
    ax.axvline(0, color="0.7", lw=0.7, zorder=1)

    # Sparse labels: a few extremes, using SNOMED CT preferred names.
    snomed = {}
    if PROC_NAMES.exists():
        snomed = json.loads(PROC_NAMES.read_text())

    def proc_label(code: str, width: int = 200) -> str:
        s = snomed.get(code, code.split("|")[-1])
        s = s.replace(" (procedure)", "").strip()
        s = " ".join(s.split())
        return s if len(s) <= width else s[: width - 1] + "\u2026"

    ref_code = PROC_REF.read_text().strip() if PROC_REF.exists() else ""
    ref_i = next((i for i, p in enumerate(pts) if p[4] == ref_code), None)
    if ref_i is not None:
        ax.annotate(
            f"ref: {proc_label(ref_code)}",
            (x[ref_i], y[ref_i]),
            fontsize=FS_NOTE,
            style="italic",
            xytext=(0.08, -2.6), textcoords="data", va="center",
            color="0.2",
            bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4),
            arrowprops=dict(arrowstyle="-", color="0.55", lw=0.7, relpos=(0.0, 0.5), shrinkB=3.0),
            zorder=5,
        )

    by_delta = sorted(range(len(pts)), key=lambda i: -abs(pts[i][0]))
    by_rate = sorted(range(len(pts)), key=lambda i: -pts[i][1])
    label_idx: list[int] = []
    for i in by_delta[:3] + by_rate[:3]:
        if i == ref_i:
            continue
        if i not in label_idx:
            label_idx.append(i)
        if len(label_idx) >= MAX_PROC_LABELS:
            break
    for c in PROC_LOW_LABELS:
        j = next((i for i, p in enumerate(pts) if p[4] == c), None)
        if j is not None and j != ref_i and j not in label_idx:
            label_idx.append(j)
    for i in label_idx:
        code = pts[i][4]
        if code in PROC_LABEL_POS:
            tx, ty, ha, va, *rel = PROC_LABEL_POS[code]
            ax.annotate(
                PROC_DISPLAY_NAMES.get(code, proc_label(code)),
                (x[i], y[i]), xytext=(tx, ty), textcoords="data",
                fontsize=FS_NOTE - 0.3, color="0.25", ha=ha, va=va,
                linespacing=1.05,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=0.4),
                arrowprops=dict(arrowstyle="-", color="0.55", lw=0.6,
                                shrinkA=1.5, shrinkB=3.0,
                                relpos=rel[0] if rel else (0.5, 0.5)),
                zorder=5,
            )
        else:
            ax.annotate(
                proc_label(code),
                (x[i], y[i]),
                fontsize=FS_NOTE - 0.3,
                xytext=(4, 4), textcoords="offset points",
                color="0.25",
            )

    ax.text(
        0.98, 0.02, f"Spearman \u03c1 = {rho.statistic:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=FS_NOTE, color="0.35", style="italic",
    )
    ax.set_ylim(top=max(ax.get_ylim()[1], 50.0))
    ax.set_xlabel("Procedure-substitution \u0394logit (vs reference)", fontsize=FS_AXIS)
    ax.set_ylabel("Observed coded PO-AKI incidence (%)", fontsize=FS_AXIS)
    fig_style.panel_title(ax, "B", "Procedure substitution", pad=7)
    ax.legend(
        fontsize=FS_NOTE, frameon=False, loc="upper left",
        handletextpad=0.4, borderaxespad=0.3, labelspacing=0.25, markerscale=0.85,
    )
    _style_axes(ax)


def build_matrix() -> tuple[list[str], dict]:
    """Select / order columns by postoperative |peak Δ|; keep pre cells for display."""
    base = pd.read_csv(CF / "crossfold_baseline.csv")
    online = pd.read_csv(CF / "crossfold_online.csv")
    pre = base[base.modality.isin(EVENT) & base.time_bin.isin(PRE)]
    post = online[online.modality.isin(EVENT) & online.time_bin.isin(POST)]
    cell: dict = {}
    for df in (pre, post):
        for _, r in df.iterrows():
            cell[(r.raw_code, r.time_bin)] = (float(r.pooled_delta_logit), bool(r.replicated))

    def peak(code: str, bins: list[str]) -> float:
        v = [cell[(code, b)][0] for b in bins if (code, b) in cell and cell[(code, b)][1]]
        return max(v, key=abs) if v else 0.0

    # Rank by postoperative streaming-head |peak|; keep top-K (no abs gate).
    codes = sorted({*post.raw_code.astype(str)})
    keep = [c for c in codes if abs(peak(c, POST)) > 0.0]
    keep.sort(key=lambda c: -abs(peak(c, POST)))
    keep = keep[:MAX_EVENT_COLS]
    return keep, cell


def draw_event_matrix(ax) -> None:
    """Dot matrix: events sorted by |peak|; vertical two-line x labels (code + name)."""
    events, cell = build_matrix()

    x: dict[str, float] = {c: float(i) for i, c in enumerate(events)}
    W = float(len(events))
    vals = [abs(v) for (v, ok) in cell.values() if ok]
    smax = float(max(vals + [0.5]))

    def area(v: float) -> float:
        return 10.0 + 320.0 * (abs(v) / smax) ** 1.55

    ty = ax.get_yaxis_transform()

    for c in events:
        ax.plot([x[c]] * 2, [YPOS[0] - 0.35, YPOS[-1] + 0.35], color="0.90", lw=0.5, zorder=1)

    for c in events:
        for b, y in zip(PRE + POST, YPOS):
            if (c, b) not in cell:
                continue
            v, ok = cell[(c, b)]
            if ok:
                ax.scatter(x[c], y, s=area(v), color=RED if v > 0 else BLUE,
                           edgecolor="white", lw=0.55, zorder=3)

    ax.axhline((YPOS[1] + YPOS[2]) / 2, color="0.2", lw=1.4, zorder=2)
    ax.text(-0.012, (YPOS[1] + YPOS[2]) / 2, "surgery\nindex", transform=ty,
            fontsize=FS_NOTE, style="italic", color="0.35",
            ha="right", va="center", linespacing=1.05)

    ax.set_xlim(-0.65, W - 0.35)
    ax.set_ylim(YPOS[-1] + 0.55, YPOS[0] - 0.55)
    ax.set_yticks(YPOS)
    ax.set_yticklabels(YLAB, fontsize=FS_TICK)
    ax.set_xticks([x[c] for c in events])
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=2.5, color="0.45", pad=2)
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    # Full names, tilted; wrapped to at most two lines.
    trans = ax.get_xaxis_transform()
    rot = 35.0
    for c in events:
        name = C_DISPLAY_NAMES.get(c, nm(c, 200))
        ax.text(
            x[c], -0.03, textwrap.fill(name, 34, break_long_words=False),
            transform=trans, rotation=rot, rotation_mode="anchor",
            ha="right", va="top", multialignment="right",
            fontsize=FS_C_NAME + 0.2, color="0.15", linespacing=1.05,
            clip_on=False, zorder=5,
        )

    fig_style.panel_title(ax, "C", "Temporal grouped-deletion attribution", pad=7)
    draw_event_matrix._area = area  # type: ignore[attr-defined]


def main() -> None:
    # Balanced A/B vs C: top row gains height; C no longer dominates.
    fig = plt.figure(figsize=(14.8, 11.2))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.08], hspace=0.36,
        left=0.075, right=0.985, top=0.905, bottom=0.18,
    )
    gs_top = gs[0].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.22)
    ax_age = fig.add_subplot(gs_top[0])
    ax_proc = fig.add_subplot(gs_top[1])
    draw_age(ax_age)
    draw_procedure_validation(ax_proc)

    ax_c = fig.add_subplot(gs[1])
    draw_event_matrix(ax_c)

    area = draw_event_matrix._area  # type: ignore[attr-defined]
    leg = [
        Line2D([], [], marker="o", lw=0, color=RED, markersize=7, mec="white",
               label="Higher model-estimated risk"),
        Line2D([], [], marker="o", lw=0, color=BLUE, markersize=7, mec="white",
               label="Lower model-estimated risk"),
    ]
    leg += [
        Line2D([], [], marker="o", lw=0, color="0.45", mec="white",
               markersize=np.sqrt(area(v)), label=f"|\u0394logit| = {v:.1f}")
        for v in (0.1, 0.3, 0.5)
    ]
    fig.legend(
        handles=leg, loc="lower center", ncol=5, frameon=False, fontsize=FS_LEGEND,
        bbox_to_anchor=(0.55, 0.02), handletextpad=0.5, columnspacing=1.15,
    )
    fig_style.add_suptitle(
        fig, "Model attribution across age, index procedure, and perioperative events", y=0.985,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight", pad_inches=0.15)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
