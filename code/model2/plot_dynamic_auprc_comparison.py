#!/usr/bin/env python3
"""Dynamic AUPRC comparison on the postoperative 6h grid, anchored at the surgery index.

All four models are scored on identical pools at every point on the x-axis. The surgery-index
baseline and the first postoperative bucket [0,6) both cover all 23,538 test episodes (2,371
positive); later buckets shrink only because AKI-positive episodes stop at the bucket holding
their last event.

The event-stream model's token-level predictions are read off the fixed 6h grid
(``eval_fixedgrid_*``), which applies the same follow-up truncation as the time-series
``eval_postop6h_*`` protocol, so the curves are directly comparable.

The x-axis is broken at 48h: the first two postoperative days are expanded so the early
milestones stay legible, and the remainder of the 30-day horizon is compressed beside it.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from common import fig_style

FOLDS = [1, 2, 3, 4, 5]
BUCKET_H = 6
N_BUCKETS = 120
HORIZON_H = BUCKET_H * N_BUCKETS  # 720

SPLIT_H = 48  # boundary between the expanded early panel and the compressed remainder
# The 48h bucket belongs to the left panel only; the right panel picks up at the next bucket, so
# every time point — and therefore every tick and every label — appears exactly once.
EARLY_MILESTONES = [6, 12, 24, 48]
LATE_MILESTONES = [168, 336]

MODELS: List[str] = ["lstm", "grud", "transformer", "eventstream"]

LABEL: Dict[str, str] = {
    "eventstream": "STREAM (proposed)",
    "transformer": "Time-series Transformer",
    "grud": "GRU-D",
    "lstm": "LSTM",
}

STYLE: Dict[str, Dict[str, object]] = {
    "eventstream": {"color": "#C4453C", "marker": "o", "lw": 1.6, "ms": 4.0, "z": 5},
    "transformer": {"color": "#3A6FB0", "marker": "s", "lw": 1.2, "ms": 3.3, "z": 4},
    "grud": {"color": "#3F8F5B", "marker": "^", "lw": 1.2, "ms": 3.5, "z": 3},
    "lstm": {"color": "#7B62A8", "marker": "D", "lw": 1.2, "ms": 3.0, "z": 2},
}

# Episode-level prediction at the surgery index ([BASE] node), one row per episode.
BASELINE_JSON: Dict[str, str] = {
    "eventstream": "event_stream/transformer/eval_baseline_fold{f}_test/baseline_metrics.json",
    "transformer": "timeseries/transformer/eval_fold{f}_test/baseline_metrics.json",
    "grud": "timeseries/grud/eval_fold{f}_test/baseline_metrics.json",
    "lstm": "timeseries/lstm/eval_fold{f}_test/baseline_metrics.json",
}

# Streaming prediction on the postoperative 6h grid.
GRID_JSON: Dict[str, str] = {
    "eventstream": "event_stream/transformer/eval_fixedgrid_fold{f}_test/fixed_time_grid_metrics.json",
    "transformer": "timeseries/transformer/eval_postop6h_fold{f}_test/postop_hour_bucket_metrics.json",
    "grud": "timeseries/grud/eval_postop6h_fold{f}_test/postop_hour_bucket_metrics.json",
    "lstm": "timeseries/lstm/eval_postop6h_fold{f}_test/postop_hour_bucket_metrics.json",
}

# (outer key, inner key, hours per unit of the bucket key). The event-stream report keys buckets
# by left-edge hour, the time-series one by bucket index.
GRID_BLOCK: Dict[str, Tuple[str, str, int]] = {
    "eventstream": ("by_bucket_hour", "", 1),
    "transformer": ("full_metrics", "by_postop_hour_bucket", BUCKET_H),
    "grud": ("full_metrics", "by_postop_hour_bucket", BUCKET_H),
    "lstm": ("full_metrics", "by_postop_hour_bucket", BUCKET_H),
}

X_BASE = -10.0  # where the surgery-index baseline sits on the left panel
X_SPLIT = -4.0  # divider between the baseline panel and the postoperative grid
X_LEFT = -15.0
Y_MAX = 0.92

# Secondary axis: share of AKI+ episodes still at risk in the pool scored at each bucket. The
# range is set wide enough that this curve stays clear of the model curves; 16% also keeps it
# readable given the pool tops out at 10.07%.
PREV_COLOR = "#57575F"
PREV_MAX = 16.0

# Past this bucket fewer than 100 AKI+ episodes remain at risk and the per-bucket AUPRC becomes
# too unstable to interpret (fold-to-fold SD climbs above 0.08).
SPARSE_FROM_H = 606


def mean_sd(vals: Sequence[float]) -> Tuple[float, float]:
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def mean_range(vals: Sequence[float]) -> Tuple[float, float, float, float]:
    """(mean, fold min, fold max, SD)."""
    m, sd = mean_sd(vals)
    return m, min(vals), max(vals), sd


def load_baseline(root: Path, model: str) -> Tuple[float, float, float, float]:
    vals = []
    for f in FOLDS:
        p = root / BASELINE_JSON[model].format(f=f)
        vals.append(float(json.loads(p.read_text())["metrics"]["auprc"]))
    return mean_range(vals)


def load_series(root: Path, model: str) -> Tuple[List[int], List[float], List[float], List[float], List[float]]:
    """Per 6h bucket: AUPRC mean, fold min, fold max and SD across the five folds, keyed by left-edge hour."""
    outer, inner, hours_per_key = GRID_BLOCK[model]
    per_fold: List[Dict[int, float]] = []
    for f in FOLDS:
        data = json.loads((root / GRID_JSON[model].format(f=f)).read_text())
        block = data[outer][inner] if inner else data[outer]
        vals: Dict[int, float] = {}
        for key, stats in block.items():
            auprc = stats.get("auprc") if isinstance(stats, dict) else None
            if auprc is None:
                continue
            vals[int(key) * hours_per_key] = float(auprc)
        per_fold.append(vals)

    xs = sorted({h for fold in per_fold for h in fold})
    means: List[float] = []
    los: List[float] = []
    his: List[float] = []
    sds: List[float] = []
    for h in xs:
        fold_vals = [fold[h] for fold in per_fold if h in fold]
        m, lo, hi, sd = mean_range(fold_vals)
        means.append(m)
        los.append(lo)
        his.append(hi)
        sds.append(sd)
    return xs, means, los, his, sds


def load_prevalence(root: Path) -> Tuple[List[int], List[float], float]:
    """Positive rate (%) per bucket, plus the surgery-index value.

    The scored pool is a property of the protocol, not of any model: every model and every fold
    sees the same 2,621,894 rows, and the event-stream and time-series reports agree on
    ``pos_rate`` in all 120 buckets. Reading one report is therefore sufficient.
    """
    grid = json.loads((root / GRID_JSON["eventstream"].format(f=1)).read_text())["by_bucket_hour"]
    xs = sorted(int(k) for k in grid)
    rates = [float(grid[str(h)]["pos_rate"]) * 100.0 for h in xs]
    base = json.loads(
        (root / BASELINE_JSON["eventstream"].format(f=1)).read_text()
    )["metrics"]["pos_rate"]
    return xs, rates, float(base) * 100.0


def annotate_column(
    ax,
    x: float,
    items: Sequence[Tuple[float, str]],
    dx: float,
    fontsize: float,
    min_gap: float,
) -> None:
    """Label every point at this x, nudging labels apart vertically where curves converge.

    Labels are stacked top-down with at least ``min_gap`` (data units) between them, then the
    whole stack is recentred on the points it describes so it stays visually attached.
    """
    for y, label_y, color in stack_labels(items, min_gap):
        ax.annotate(
            f"{y:.3f}",
            xy=(x, label_y),
            xytext=(dx, 0),
            textcoords="offset points",
            ha="left" if dx >= 0 else "right",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="medium",
            zorder=12,
            bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85),
        )


def stack_labels(items: Sequence[Tuple[float, str]], min_gap: float) -> List[Tuple[float, float, str]]:
    """Return (value, label_y, color), spread vertically by >= min_gap and recentred on the values."""
    order = sorted(items, key=lambda t: -t[0])
    placed: List[float] = []
    for y, _ in order:
        placed.append(y if not placed else min(y, placed[-1] - min_gap))
    shift = (sum(y for y, _ in order) - sum(placed)) / len(placed)
    return [(y, ty + shift, color) for (y, color), ty in zip(order, placed)]


def draw_model(ax, model: str, xs: Sequence[int], means: Sequence[float], band_lo: Sequence[float],
               band_hi: Sequence[float], lo: float, hi: float, label: Optional[str]) -> None:
    """Plot one model's curve restricted to the x-window [lo, hi]."""
    s = STYLE[model]
    idx = [i for i, x in enumerate(xs) if lo <= x <= hi]
    if not idx:
        return
    xv = [xs[i] for i in idx]
    mv = [means[i] for i in idx]
    color = str(s["color"])
    ax.fill_between(
        xv,
        [band_lo[i] for i in idx],
        [band_hi[i] for i in idx],
        color=color,
        alpha=0.10,
        lw=0,
        zorder=int(s["z"]),
    )
    ax.plot(
        xv,
        mv,
        marker=str(s["marker"]),
        color=color,
        lw=float(s["lw"]),
        ms=float(s["ms"]),
        mfc="white",
        mew=1.0,
        label=label,
        zorder=int(s["z"]) + 5,
    )


def build_figure(root: Path, out_png: Path, out_csv: Optional[Path], dpi: int) -> None:
    baseline = {mdl: load_baseline(root, mdl) for mdl in MODELS}
    series = {mdl: load_series(root, mdl) for mdl in MODELS}
    prev_x, prev_y, prev_base = load_prevalence(root)

    fig, (axl, axr) = plt.subplots(
        1,
        2,
        figsize=(20.0, 8.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.4, 1.75], "wspace": 0.035},
    )

    # Prevalence lives on twin axes drawn *beneath* the model curves.
    axl2, axr2 = axl.twinx(), axr.twinx()
    for main, twin in ((axl, axl2), (axr, axr2)):
        main.set_zorder(twin.get_zorder() + 1)
        main.patch.set_visible(False)
        twin.set_ylim(0.0, PREV_MAX)

    axl2.axvspan(X_LEFT, X_SPLIT, color="#F2F2F4", zorder=0)
    axl.axvline(X_SPLIT, color="#8A8A93", lw=1.0, ls=(0, (5, 4)), zorder=1)

    for twin, lo, hi, lbl in (
        (axl2, 0, SPLIT_H, None),
        (axr2, SPLIT_H, HORIZON_H, "PO-AKI prevalence in the remaining risk set (right axis)"),
    ):
        pts = [(x, y) for x, y in zip(prev_x, prev_y) if lo <= x <= hi]
        twin.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            color=PREV_COLOR,
            lw=1.3,
            ls=(0, (4, 2)),
            marker=".",
            ms=3.4,
            label=lbl,
            zorder=3,
        )
    axl2.plot(
        [X_BASE], [prev_base], marker="*", ms=11, color=PREV_COLOR, ls="none", zorder=4
    )

    axr2.axvspan(SPARSE_FROM_H, HORIZON_H, color="#EFEFF2", zorder=0)
    axr2.text(
        SPARSE_FROM_H + 6,
        PREV_MAX * 0.985,
        "fewer than 100\nPO-AKI-positive\nepisodes at risk",
        ha="left",
        va="top",
        fontsize=7.8,
        color="#7A7A84",
        style="italic",
    )

    for mdl in MODELS:
        xs, means, los, his, _ = series[mdl]
        draw_model(axl, mdl, xs, means, los, his, 0, SPLIT_H, None)
        draw_model(axr, mdl, xs, means, los, his, SPLIT_H, HORIZON_H, LABEL[mdl])
        s = STYLE[mdl]
        b_mean, b_lo, b_hi, _ = baseline[mdl]
        axl.errorbar(
            [X_BASE],
            [b_mean],
            yerr=[[b_mean - b_lo], [b_hi - b_mean]],
            fmt=str(s["marker"]),
            color=str(s["color"]),
            ms=float(s["ms"]) + 3.2,
            mfc=str(s["color"]),
            mew=1.35,
            capsize=3.5,
            elinewidth=1.2,
            zorder=int(s["z"]) + 5,
        )

    lookup = {m: dict(zip(series[m][0], series[m][1])) for m in MODELS}
    annotate_column(
        axl,
        X_BASE,
        [(baseline[m][0], str(STYLE[m]["color"])) for m in MODELS],
        dx=9,
        fontsize=8.8,
        min_gap=0.028,
    )
    # The 48h bucket is drawn on both panels, so it is labelled on both — reading inward on each
    # side of the break (leftward on the left panel, rightward on the right panel).
    for x in EARLY_MILESTONES:
        annotate_column(
            axl,
            float(x),
            [(lookup[m][x], str(STYLE[m]["color"])) for m in MODELS if x in lookup[m]],
            dx=-5 if x == SPLIT_H else 5,
            fontsize=7.8,
            min_gap=0.027,
        )
    for x in LATE_MILESTONES + [SPLIT_H]:
        annotate_column(
            axr,
            float(x),
            [(lookup[m][x], str(STYLE[m]["color"])) for m in MODELS if x in lookup[m]],
            dx=8,
            fontsize=7.8 if x == SPLIT_H else 8.2,
            min_gap=0.026,
        )

    # Prevalence values at the same milestones, tucked under its own curve.
    prev_lookup = dict(zip(prev_x, prev_y))
    for twin, marks, inward_left in (
        (axl2, EARLY_MILESTONES, True),
        (axr2, LATE_MILESTONES + [SPLIT_H], False),
    ):
        for x in marks:
            if x not in prev_lookup:
                continue
            at_break = x == SPLIT_H
            twin.annotate(
                f"{prev_lookup[x]:.1f}%",
                xy=(x, prev_lookup[x]),
                xytext=((-6 if inward_left else 6), -11) if at_break else (0, -12),
                textcoords="offset points",
                ha=("right" if inward_left else "left") if at_break else "center",
                va="top",
                fontsize=8.0,
                color=PREV_COLOR,
                zorder=12,
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.85),
            )
    axl2.annotate(
        f"{prev_base:.1f}%",
        xy=(X_BASE, prev_base),
        xytext=(0, -13),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.4,
        color=PREV_COLOR,
        zorder=12,
    )

    # Light guides at the annotated milestones so the x positions are unambiguous. The boundary
    # bucket gets one on each panel, so the pair brackets the gap holding its labels.
    for ax, marks in ((axl, EARLY_MILESTONES + [SPLIT_H]), (axr, LATE_MILESTONES + [SPLIT_H])):
        for x in marks:
            ax.axvline(x, color="#B4B4BC", lw=0.7, ls=(0, (2, 3)), zorder=1)

    axl.set_xlim(X_LEFT, SPLIT_H + 4)
    axr.set_xlim(SPLIT_H - 4, HORIZON_H - 2)
    axl.set_ylim(0.0, Y_MAX)

    # Both panels index the same hour-valued data; the units differ only in how they are read,
    # so each tick and each panel label carries its unit explicitly.
    left_ticks = list(range(0, SPLIT_H + 1, 6))
    axl.set_xticks([X_BASE] + [float(t) for t in left_ticks])
    axl.set_xticklabels(["Surgery index\n(baseline)"] + [f"{t} h" for t in left_ticks], fontsize=8.4)
    right_ticks = list(range(SPLIT_H, HORIZON_H, 24))
    axr.set_xticks([float(t) for t in right_ticks])
    axr.set_xticklabels([f"{t // 24} d" for t in right_ticks], fontsize=7.6)
    axr.xaxis.set_minor_locator(MultipleLocator(BUCKET_H))

    axl.set_xlabel("Postoperative hour (6-h windows)", fontsize=11, labelpad=7)
    axr.set_xlabel("Postoperative day (6-h windows)", fontsize=11, labelpad=7)

    for ax in (axl, axr):
        ax.grid(True, axis="y", alpha=0.28, lw=0.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
    axl.spines["right"].set_visible(False)
    axr.spines["left"].set_visible(False)
    axr.spines["right"].set_visible(False)
    axr.tick_params(axis="y", left=False)

    for twin in (axl2, axr2):
        twin.spines["top"].set_visible(False)
        twin.spines["left"].set_visible(False)
    axl2.spines["right"].set_visible(False)
    axl2.tick_params(axis="y", right=False, labelright=False)
    axr2.spines["right"].set_color(PREV_COLOR)
    axr2.tick_params(axis="y", colors=PREV_COLOR, labelsize=9)
    axr2.set_ylabel(
        "PO-AKI prevalence in the remaining risk set (%)",
        fontsize=11,
        color=PREV_COLOR,
    )

    # Axis-break marks on the facing edges.
    kw = dict(transform=axl.transAxes, color="#8A8A93", clip_on=False, lw=1.0)
    axl.plot([1.0, 1.0], [-0.012, 0.012], **kw)
    kw["transform"] = axr.transAxes
    axr.plot([0.0, 0.0], [-0.012, 0.012], **kw)

    axl.set_ylabel("AUPRC", fontsize=11)
    axl.text(
        (X_LEFT + X_SPLIT) / 2,
        0.905,
        "Episode-level\nat surgery index",
        ha="center",
        va="top",
        fontsize=8.6,
        color="#55555E",
        style="italic",
    )
    axl.text(
        1.0,
        0.905,
        "Postoperative hours 0–48, expanded",
        ha="left",
        va="top",
        fontsize=8.6,
        color="#55555E",
        style="italic",
    )
    axr.text(
        SPLIT_H + 4,
        0.905,
        "Postoperative days 2–30, compressed",
        ha="left",
        va="top",
        fontsize=8.6,
        color="#55555E",
        style="italic",
    )

    handles, labels = axr.get_legend_handles_labels()
    h2, l2 = axr2.get_legend_handles_labels()
    axr.legend(
        handles + h2,
        labels + l2,
        loc="upper right",
        bbox_to_anchor=(0.995, 0.86),
        fontsize=10,
        frameon=True,
        framealpha=0.94,
        edgecolor="#D0D0D6",
    )

    fig_style.add_suptitle(
        fig,
        "Dynamic PO-AKI discrimination over postoperative follow-up",
        y=0.975,
    )
    # tight_layout cannot account for the twinned axes, so pin the margins explicitly.
    fig.subplots_adjust(left=0.043, right=0.947, top=0.885, bottom=0.085)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    print(f"Wrote {out_png}")

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(
                ["point", "hour_start", "hour_end"]
                + [f"{m}_{stat}" for m in MODELS for stat in ("mean", "min", "max", "sd")]
            )
            row: List[object] = ["index0_baseline", "", ""]
            for m in MODELS:
                row += [f"{v:.6f}" for v in baseline[m]]
            w.writerow(row)
            for i, h in enumerate(series[MODELS[0]][0]):
                row = [f"postop_bucket_{h}h", h, h + BUCKET_H]
                for m in MODELS:
                    row += [f"{series[m][k][i]:.6f}" for k in (1, 2, 3, 4)]
                w.writerow(row)
        print(f"Wrote {out_csv}")


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--result_root", type=str, default=str(here / ".." / ".." / "result"))
    p.add_argument("--out_png", type=str, default=None)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.result_root).resolve()
    out_png = Path(args.out_png) if args.out_png else root / "figures" / "dynamic_auprc_comparison.png"
    out_csv = Path(args.out_csv) if args.out_csv else root / "figures" / "dynamic_auprc_comparison.csv"
    build_figure(root, out_png.resolve(), out_csv.resolve(), args.dpi)


if __name__ == "__main__":
    main()
