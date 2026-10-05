"""Main event-contribution figure: ranked pre/post effects by sign.

Why this form (vs the wide period×sign dot matrix)
  The scientific question is ranked: which preoperative events raise / lower risk,
  and which postoperative events raise / lower risk. A two-panel tornado
  (bidirectional bar) answers that directly — magnitude is bar length, sign is
  side/colour, uncertainty is a CI whisker. A compact time-bin sparkline beside
  each name preserves *when* within the period the effect peaks, without forcing
  every code into its own full-height column.

  The same code may appear on both panels (e.g. shock) whenever it has a
  replicated peak preoperatively and postoperatively.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
CF = ROOT / "result/event_stream/transformer/contribution_5fold_full"
OUT = ROOT / "result/figures/event_contribution_main.png"

TOP_POS, TOP_NEG = 5, 5
DISPLAY_CAP = 0.55  # bars longer than this are truncated with the true value labelled
RED, BLUE = "#c0392b", "#1f6cb0"
PRE = ["preop_7d_48h", "preop_48h_0h"]
POST = ["postop_0_6h", "postop_6_24h", "postop_24_72h", "postop_72h_7d"]
PRE_LAB = ["\u22127d\u201348h", "\u221248h\u20130"]
POST_LAB = ["0\u20136h", "6\u201324h", "24\u201372h", "72h\u20137d"]
EVENT = ["condition", "med", "lab"]

ccs = {r["ccs_code"]: r["definition"] for r in csv.DictReader(open(ROOT / "data/ccs_codes.csv"))}
rx = dict(l.strip().split(",", 1) for l in open(ROOT / "data/rxcui_to_name.csv") if "," in l)
labn = json.load(open(ROOT / "data/lab_concept_names.json"))
SEX = {"8532": "female", "8507": "male", "0": "sex not recorded"}
RACE = {"8515": "Asian", "8516": "Black or African American", "8527": "White",
        "8657": "American Indian / Alaska Native",
        "8557": "Native Hawaiian / Pacific Islander", "0": "race not recorded"}


def nm(code: str, width: int = 34) -> str:
    p = code.split("|")
    if code.startswith("CCS|"):
        s = ccs.get(p[1], code)
    elif code.startswith("RXCUI|"):
        s = rx.get(code, code)
    elif code.startswith("LAB|"):
        s = labn.get(p[1], code)
    elif code.startswith("SEX|"):
        s = SEX.get(p[1], code)
    elif code.startswith("RACE|"):
        s = RACE.get(p[1], code)
    elif code == "AGE":
        s = "age"
    else:
        s = code
    s = " ".join(s.split())
    return s if len(s) <= width else s[: width - 1] + "\u2026"


def peak_table(df: pd.DataFrame, bins: list[str]) -> pd.DataFrame:
    sub = df[df.modality.isin(EVENT) & df.time_bin.isin(bins) & df.replicated].copy()
    if sub.empty:
        return sub
    rows = []
    for code, g in sub.groupby("raw_code"):
        i = g.pooled_delta_logit.abs().idxmax()
        r = g.loc[i]
        rows.append({
            "raw_code": code,
            "peak_bin": r.time_bin,
            "delta": float(r.pooled_delta_logit),
            "ci_lo": float(r.ci_lo),
            "ci_hi": float(r.ci_hi),
            "n_patients": int(r.n_patients),
            "traj": {b: float(gg.pooled_delta_logit.iloc[0])
                     for b, gg in g.groupby("time_bin")},
        })
    out = pd.DataFrame(rows)
    out["absd"] = out.delta.abs()
    return out


def pick(peak: pd.DataFrame) -> pd.DataFrame:
    pos = peak[peak.delta > 0].nlargest(TOP_POS, "absd")
    neg = peak[peak.delta < 0].nlargest(TOP_NEG, "absd")
    sel = pd.concat([pos, neg], ignore_index=True)
    return sel.sort_values("delta", ascending=True).reset_index(drop=True)


def draw_panel(ax, ax_spark, sel: pd.DataFrame, bins: list[str], bin_labs: list[str],
               title: str) -> None:
    y = np.arange(len(sel))
    colors = [RED if d > 0 else BLUE for d in sel.delta]
    shown = sel.delta.clip(-DISPLAY_CAP, DISPLAY_CAP)
    ax.axvline(0, color="0.35", lw=0.9, zorder=1)
    ax.barh(y, shown, color=colors, height=0.70, alpha=0.88, zorder=2,
            edgecolor="white", linewidth=0.4)
    for yi, lo, hi, d, sh, col in zip(y, sel.ci_lo, sel.ci_hi, sel.delta, shown, colors):
        lo_s, hi_s = max(lo, -DISPLAY_CAP), min(hi, DISPLAY_CAP)
        ax.plot([lo_s, hi_s], [yi, yi], color="0.2", lw=1.0, solid_capstyle="round", zorder=3)
        ax.plot([lo_s, lo_s], [yi - 0.12, yi + 0.12], color="0.2", lw=1.0, zorder=3)
        ax.plot([hi_s, hi_s], [yi - 0.12, yi + 0.12], color="0.2", lw=1.0, zorder=3)
        # only annotate bars that exceed the display cap
        if abs(d) > DISPLAY_CAP + 1e-9:
            side = 1 if d > 0 else -1
            ax.text(DISPLAY_CAP * side + side * 0.03, yi, f"{d:+.2f}", fontsize=8.0,
                    color="0.15", va="center", ha="left" if d > 0 else "right",
                    fontweight="bold", zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{nm(c, 36)}\n({c}, n={int(n)})" for c, n in zip(sel.raw_code, sel.n_patients)],
        fontsize=8.5, linespacing=1.05,
    )

    xmax = DISPLAY_CAP * 1.32
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(-0.65, len(sel) - 0.35)
    ax.set_xlabel("\u0394logit (with group \u2212 without group)", fontsize=9.5)
    ax.set_title(title, fontsize=11.5, weight="bold", loc="left", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.75")
    ax.spines["bottom"].set_color("0.55")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="0.9", lw=0.7, zorder=0)

    smax = max(float(sel.absd.max()), 1e-6)
    xs = np.arange(len(bins), dtype=float)
    for yi, row in enumerate(sel.itertuples()):
        ax_spark.plot([xs[0] - 0.25, xs[-1] + 0.25], [yi, yi], color="0.88", lw=0.7, zorder=1)
        for x, b in zip(xs, bins):
            if b not in row.traj:
                ax_spark.plot(x, yi, marker="o", ms=2.6, color="0.82", zorder=2)
                continue
            v = row.traj[b]
            ms = 2.8 + 6.5 * (abs(v) / smax)
            ax_spark.plot(x, yi, marker="o", ms=ms, color=RED if v > 0 else BLUE,
                          markeredgecolor="white", markeredgewidth=0.35, zorder=3)
    ax_spark.set_xlim(xs[0] - 0.55, xs[-1] + 0.55)
    ax_spark.set_ylim(ax.get_ylim())
    ax_spark.set_xticks(xs)
    ax_spark.set_xticklabels(bin_labs, fontsize=7.5, rotation=90)
    ax_spark.set_yticks([])
    ax_spark.set_title("when", fontsize=9.0, color="0.35", pad=4)
    for s in ax_spark.spines.values():
        s.set_visible(False)
    ax_spark.tick_params(length=0)


def draw_demographics(ax, demo: pd.DataFrame) -> None:
    demo = demo.sort_values("pooled_delta_logit")
    y = np.arange(len(demo))
    colors = [RED if d > 0 else BLUE for d in demo.pooled_delta_logit]
    ax.axvline(0, color="0.35", lw=0.9)
    ax.barh(y, demo.pooled_delta_logit, color=colors, height=0.62, alpha=0.9,
            edgecolor="white", lw=0.4)
    for yi, lo, hi in zip(y, demo.ci_lo, demo.ci_hi):
        ax.plot([lo, hi], [yi, yi], color="0.2", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"{nm(c, 28)}\n({c})" for c in demo.raw_code],
        fontsize=8.5, linespacing=1.05,
    )
    ax.set_xlabel("\u0394logit", fontsize=9.5)
    ax.set_title("Time-invariant attributes", fontsize=11.5, weight="bold", loc="left", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("0.75")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", color="0.9", lw=0.7)
    lim = max(0.85, float(demo.pooled_delta_logit.abs().max()) * 1.15)
    ax.set_xlim(-lim, lim)


def main() -> None:
    base = pd.read_csv(CF / "crossfold_baseline.csv")
    online = pd.read_csv(CF / "crossfold_online.csv")

    pre = pick(peak_table(base, PRE))
    post = pick(peak_table(online, POST))
    demo = base[(base.modality == "static") & base.replicated & (base.raw_code != "AGE")].copy()

    fig = plt.figure(figsize=(14.5, 10.8))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 3.2], hspace=0.34,
        left=0.24, right=0.985, top=0.88, bottom=0.09,
    )
    ax_demo = fig.add_subplot(gs[0])
    gs_bot = gs[1].subgridspec(1, 2, wspace=0.48, width_ratios=[1.0, 1.0])

    def panel_axes(spec):
        g = spec.subgridspec(1, 2, width_ratios=[4.0, 1.05], wspace=0.10)
        return fig.add_subplot(g[0]), fig.add_subplot(g[1])

    ax_pre, ax_pre_s = panel_axes(gs_bot[0])
    ax_post, ax_post_s = panel_axes(gs_bot[1])

    draw_demographics(ax_demo, demo)
    draw_panel(ax_pre, ax_pre_s, pre, PRE, PRE_LAB, "Preoperative contributions")
    draw_panel(ax_post, ax_post_s, post, POST, POST_LAB, "Postoperative contributions")

    leg = [
        Line2D([], [], color=RED, lw=6, label="raises predicted AKI risk", solid_capstyle="butt"),
        Line2D([], [], color=BLUE, lw=6, label="lowers predicted AKI risk", solid_capstyle="butt"),
        Line2D([], [], color="0.2", lw=1.2, label="patient-bootstrap 95% CI"),
        Line2D([], [], marker="o", color="0.35", lw=0, markersize=5,
               label="replicated bin in time profile"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=4, frameon=False, fontsize=9.0,
               bbox_to_anchor=(0.58, 0.005), handletextpad=0.6, columnspacing=1.4)

    fig.suptitle(
        "Which events move predicted AKI risk \u2014 before vs after surgery",
        fontsize=14.0, y=0.980,
    )
    fig.text(
        0.58, 0.945,
        "full held-out test set (n=23,538); peak replicated \u0394logit within each period; "
        f"bars clipped at |\u0394|={DISPLAY_CAP:.2f}; AGE shown in the dose\u2013response figure",
        ha="center", fontsize=8.5, color="0.35",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"pre rows={len(pre)} post rows={len(post)} demo={len(demo)}")
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
