#!/usr/bin/env python3
"""Study cohort and perioperative temporal landscape (3 panels).

A: cohort construction flowchart (workflow.png)
B: timing of first coded PO-AKI diagnosis
C: perioperative EHR event density, PO-AKI vs no PO-AKI
"""
from __future__ import annotations

import os

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from PIL import Image

import sys

import run_descriptive_analysis_dedup4 as R

sys.path.insert(0, str(Path(__file__).resolve().parent / "model2"))
from common import fig_style  # noqa: E402

PROJECT_ROOT = Path(os.environ.get("PERIOP_ROOT", Path(__file__).resolve().parents[1]))

OUT = (PROJECT_ROOT / "result/tables/descriptive_analysis_dedup4")
C_POS = "#C44E52"
C_NEG = "#4C72B0"
PREOP_DAYS, POSTOP_DAYS, FINE_HALF = 7, 30, 2.0


def _flat_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGBA")
    bg = Image.new("RGBA", im.size, "white")
    return np.asarray(Image.alpha_composite(bg, im).convert("RGB"))


def panel_timing(ax, aki_days: np.ndarray) -> None:
    ax.hist(aki_days, bins=np.arange(0, POSTOP_DAYS + 1, 1), color="skyblue",
            edgecolor="k", linewidth=0.7, density=True)
    ax.set_xlim(0, POSTOP_DAYS)
    ax.set_xticks(np.arange(0, POSTOP_DAYS + 1, 2))
    ax.set_xlabel("Days after index surgery", fontsize=11.5)
    ax.set_ylabel("Proportion of PO-AKI episodes", fontsize=11.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.45)
    ax.spines[["top", "right"]].set_visible(False)


def panel_density(ax, pos: dict, neg: dict, bins: np.ndarray) -> None:
    x = (bins[:-1] + bins[1:]) / 2
    m = (x >= -PREOP_DAYS) & (x <= POSTOP_DAYS)
    x, yp, yn = x[m], np.asarray(pos["total"])[m], np.asarray(neg["total"])[m]
    ax.axvspan(-FINE_HALF, FINE_HALF, color="#f0e6d2", alpha=0.35, zorder=0)
    ax.fill_between(x, yn, 0, color=C_NEG, alpha=0.10, linewidth=0)
    ax.fill_between(x, yp, 0, color=C_POS, alpha=0.10, linewidth=0)
    ax.plot(x, yp, color=C_POS, lw=2.0, marker="s", ms=4.2, mec="white", mew=0.6, label="PO-AKI")
    ax.plot(x, yn, color=C_NEG, lw=2.0, marker="o", ms=4.5, mec="white", mew=0.6, label="No PO-AKI")
    ax.axvline(0, color="#333333", ls="--", lw=1.2, alpha=0.65)
    ax.set_xlim(-PREOP_DAYS, POSTOP_DAYS)
    ax.set_ylim(0, max(yp.max(), yn.max()) * 1.08)
    ax.set_xticks(np.arange(-PREOP_DAYS + 1, POSTOP_DAYS + 1, 2))
    ax.set_xlabel("Days relative to index surgery", fontsize=11.5)
    ax.set_ylabel("Mean events per patient-day", fontsize=11.5)
    ax.legend(fontsize=10.5, loc="upper right", frameon=False)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    cache = R.load_density_cache(OUT / "densities.npz")
    flow = _flat_rgb(OUT / "workflow.png")

    fig = plt.figure(figsize=(15.6, 9.1))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[0.74, 1.0], wspace=0.12, hspace=0.42,
                  left=0.01, right=0.985, top=0.875, bottom=0.07)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    ax_a.imshow(flow, interpolation="lanczos")
    ax_a.set_axis_off()
    ax_a.set_anchor("N")
    panel_timing(ax_b, np.asarray(cache["aki_deltas"]))
    panel_density(ax_c, cache["pos_overlay"], cache["neg_overlay"], cache["overlay_bins"])
    fig.align_ylabels([ax_b, ax_c])

    fig_style.add_suptitle(fig, "Study cohort and perioperative temporal landscape", y=0.985)
    fig.canvas.draw()
    pa = ax_a.get_position()
    fig.text(pa.x0 + 0.005, 0.90, fig_style.panel_label("A", "Cohort construction"),
             va="bottom", **fig_style.panel_kw(fig))
    lab_x = ax_b.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted()).x0
    for ax, letter, title in ((ax_b, "B", "Timing of first coded PO-AKI diagnosis"),
                              (ax_c, "C", "Perioperative EHR event density")):
        y = ax.get_position().y1 + 0.012
        fig.text(lab_x, y, fig_style.panel_label(letter, title), va="bottom", **fig_style.panel_kw(fig))

    out = OUT / "fig_cohort_temporal_landscape.png"
    fig.savefig(out, dpi=300, facecolor="white")
    fig.savefig(out.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    print("Wrote", out)


if __name__ == "__main__":
    main()
