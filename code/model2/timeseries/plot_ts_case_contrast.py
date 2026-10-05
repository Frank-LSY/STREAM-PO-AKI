#!/usr/bin/env python3
"""Contrast figure: time-series Transformer 6h risk trajectories on curated cases.

Same patients as the event-arrival case studies (IDs from STREAM_CASE_POS_ID / STREAM_CASE_NEG_ID).
Fixed 6h postoperative bins → piecewise-constant risk staircase.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import fig_style
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
TS_ROOT = ROOT / "result/timeseries/transformer"
NPZ = ROOT / "data/five_folds_ts6h4/test.npz"
FIG = ROOT / "result/figures/timeseries"
CACHE = FIG / "curated_case_trajectories_5fold.json"

# Case-study person_ids are site data and are not distributed; set them via environment variables.
CURATED_POS = int(os.environ.get("STREAM_CASE_POS_ID", "-1"))
CURATED_NEG = int(os.environ.get("STREAM_CASE_NEG_ID", "-2"))
POS_AKI_H = 15.434444444444445
RED, BLUE = "#c0392b", "#1f6cb0"
RISK_THR = 0.5
RISK_YLIM = (-0.02, 1.05)
BUCKET_H = 6.0
# Full 30-day postoperative monitoring window for the AKI− case.
NEG_MONITOR_END_H = 720.0
PLOT_HORIZON_H = NEG_MONITOR_END_H

_PANEL_ROLE = {
    CURATED_POS: "PO-AKI positive episode",
    CURATED_NEG: "PO-AKI negative episode",
}
_PANEL_SUBTITLE_STEM = {
    CURATED_POS: ("Laparoscopic appendectomy", "PO-AKI at 15.4 h"),
    CURATED_NEG: ("Exploratory laparotomy", "No PO-AKI within 30 days"),
}


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@torch.no_grad()
def _infer_cases(pids: list[int], horizon_h: float = PLOT_HORIZON_H) -> dict[int, dict]:
    """Five-fold mean baseline + postop bucket probs for selected person_ids."""
    import sys

    sys.path.insert(0, str(ROOT / "code/model2"))
    from timeseries.checkpoint import load_bucket_checkpoint
    from timeseries.dataset import Ts6hEpisodeDataset, _collate_indices
    from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, BUCKET_SIZE_H, NUM_POSTOP_BUCKETS

    ds = Ts6hEpisodeDataset(NPZ)
    pid_to_idx = {int(str(p)): i for i, p in enumerate(ds.person_ids)}
    missing = [p for p in pids if p not in pid_to_idx]
    if missing:
        raise RuntimeError(f"person_ids not in test.npz: {missing}")

    indices = [pid_to_idx[p] for p in pids]
    batch = _collate_indices(ds, indices)
    device = _device()
    max_bucket = int(np.ceil(horizon_h / BUCKET_SIZE_H)) - 1  # inclusive index

    # Accumulators: baseline + per-postop-bucket
    base_sum = np.zeros(len(pids), dtype=np.float64)
    # Shape: (n_pids, NUM_POSTOP_BUCKETS)
    bucket_sum = np.zeros((len(pids), NUM_POSTOP_BUCKETS), dtype=np.float64)
    n_folds = 0

    for fold in range(1, 6):
        ckpt = TS_ROOT / f"fold{fold}/best_model.pt"
        if not ckpt.exists():
            raise FileNotFoundError(ckpt)
        model, _, _, _ = load_bucket_checkpoint(ckpt, device)
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        out = model(
            bucket_features=batch_t["bucket_features"],
            attention_mask=batch_t["attention_mask"],
            base_mask=batch_t["base_mask"],
        )
        token_probs = torch.sigmoid(out["token_logits"]).cpu().numpy()
        base_probs = torch.sigmoid(out["baseline_logits_all"]).cpu().numpy()
        for i in range(len(pids)):
            base_sum[i] += float(base_probs[i, BASE_BUCKET_IDX])
            last_j = min(int(batch["max_postop_bucket"][i]), NUM_POSTOP_BUCKETS - 1, max_bucket)
            for j in range(last_j + 1):
                idx = BASE_BUCKET_IDX + 1 + j
                bucket_sum[i, j] += float(token_probs[i, idx])
        n_folds += 1
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out_recs: dict[int, dict] = {}
    for i, pid in enumerate(pids):
        last_j = min(int(batch["max_postop_bucket"][i]), NUM_POSTOP_BUCKETS - 1, max_bucket)
        buckets = []
        for j in range(last_j + 1):
            bh = float(j * BUCKET_SIZE_H)
            buckets.append(
                {
                    "bucket_hour": int(bh),
                    "bucket_hour_end": float(bh + BUCKET_SIZE_H),
                    "aki_risk_prob": float(bucket_sum[i, j] / n_folds),
                }
            )
        out_recs[int(pid)] = {
            "person_id": int(pid),
            "label_aki": int(batch["labels"][i].item()),
            "baseline_aki_risk_prob": float(base_sum[i] / n_folds),
            "max_postop_bucket": int(batch["max_postop_bucket"][i].item()),
            "n_folds": n_folds,
            "bucket_predictions": buckets,
        }
    return out_recs


def _load_or_infer(pids: list[int], force: bool = False) -> dict[int, dict]:
    FIG.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and not force:
        raw = json.loads(CACHE.read_text())
        by_pid = {int(k): v for k, v in raw.items()}
        if all(p in by_pid for p in pids):
            return {p: by_pid[p] for p in pids}
    recs = _infer_cases(pids)
    CACHE.write_text(json.dumps({str(k): v for k, v in recs.items()}, indent=2))
    return recs


def _staircase_xy(
    baseline: float,
    buckets: list[dict],
    truncate_at: float | None = None,
) -> tuple[list[float], list[float]]:
    """Build step coordinates: hold risk constant within each 6h bin."""
    if not buckets:
        return [0.0], [baseline]
    xs = [0.0]
    ys = [baseline]
    for b in buckets:
        t0 = float(b["bucket_hour"])
        t1 = float(b["bucket_hour_end"])
        p = float(b["aki_risk_prob"])
        if truncate_at is not None and t0 >= truncate_at - 1e-9:
            break
        if truncate_at is not None:
            t1 = min(t1, float(truncate_at))
        if t1 <= t0 + 1e-12:
            continue
        if abs(xs[-1] - t0) > 1e-9:
            xs.append(t0)
            ys.append(ys[-1])
        xs.append(t0)
        ys.append(p)
        xs.append(t1)
        ys.append(p)
        if truncate_at is not None and abs(t1 - float(truncate_at)) < 1e-9:
            break
    return xs, ys


def _risk_color(p: float) -> str:
    return RED if p >= RISK_THR else BLUE


def _fmt_hour_tick(t: float) -> str:
    """Axis tick like ``0 h``, ``8 h``, ``12 h``."""
    if abs(t - round(t)) < 1e-6:
        return f"{int(round(t))} h"
    return f"{t:.1f} h"


def _long_xticks(xmax: float) -> list[float]:
    """Sparse hour ticks plus monitor end; drop a tick that crowds the end label."""
    # ~weekly spacing in hours, but labels stay in h.
    step = 168.0
    mids = [float(t) for t in np.arange(step, xmax, step)]
    mids = [t for t in mids if xmax - t >= 80.0]
    return sorted({0.0, *mids, float(xmax)})


def _key_markers_long(
    baseline: float,
    buckets: list[dict],
    end_t: float,
    final: float,
) -> tuple[list[float], list[float]]:
    """Only start, first threshold crossing, and monitor end."""
    mark_t = [0.0]
    mark_p = [baseline]
    prev_p = baseline
    for b in buckets:
        t0 = float(b["bucket_hour"])
        p = float(b["aki_risk_prob"])
        if (prev_p - RISK_THR) * (p - RISK_THR) <= 0 and abs(prev_p - p) > 1e-9:
            if abs(t0 - mark_t[-1]) > 1e-9:
                mark_t.append(t0)
                mark_p.append(p)
            break
        prev_p = p
    if abs(end_t - mark_t[-1]) > 1e-9:
        mark_t.append(end_t)
        mark_p.append(final)
    else:
        mark_t[-1], mark_p[-1] = end_t, final
    return mark_t, mark_p


def _draw_panel(
    ax,
    rec: dict,
    panel: str,
    pid: int,
    aki_hours: float | None = None,
    x_max: float | None = None,
) -> None:
    baseline = float(rec["baseline_aki_risk_prob"])
    buckets = list(rec["bucket_predictions"])
    truncate_at = float(aki_hours) if aki_hours is not None and np.isfinite(aki_hours) else None
    if x_max is not None:
        buckets = [b for b in buckets if float(b["bucket_hour"]) < float(x_max) + 1e-9]
    if truncate_at is not None:
        buckets = [b for b in buckets if float(b["bucket_hour"]) < truncate_at + 1e-9]

    stair_end = truncate_at if truncate_at is not None else (float(x_max) if x_max is not None else None)
    xs, ys = _staircase_xy(baseline, buckets, truncate_at=stair_end)
    final = float(ys[-1]) if ys else baseline
    end_t = float(xs[-1]) if xs else 0.0
    xmax_plot = float(
        truncate_at if truncate_at is not None
        else (x_max if x_max is not None else end_t)
    )
    long_panel = xmax_plot >= 72.0

    # Soft bands + decision threshold
    ax.axhspan(RISK_YLIM[0], RISK_THR, facecolor=BLUE, alpha=0.06, zorder=0, lw=0)
    ax.axhspan(RISK_THR, RISK_YLIM[1], facecolor=RED, alpha=0.06, zorder=0, lw=0)
    ax.axhline(
        RISK_THR, color="#4a5568", lw=1.15, linestyle=(0, (3.2, 2.0)),
        zorder=1.6, alpha=0.90,
    )

    # Trajectory: thin continuous line on long panels; chunky on short.
    line_w = 1.35 if long_panel else 2.0
    for i in range(len(xs) - 1):
        c = _risk_color(0.5 * (ys[i] + ys[i + 1]))
        ax.plot(
            [xs[i], xs[i + 1]], [ys[i], ys[i + 1]],
            color=c, lw=line_w, solid_capstyle="butt", zorder=3, alpha=0.95,
        )

    if long_panel:
        mark_t, mark_p = _key_markers_long(baseline, buckets, end_t, final)
        msize = 5.0
    else:
        mark_t = [0.0]
        mark_p = [baseline]
        for b in buckets:
            t0 = float(b["bucket_hour"])
            if truncate_at is not None and t0 >= truncate_at - 1e-9:
                break
            mark_t.append(t0)
            mark_p.append(float(b["aki_risk_prob"]))
        mark_t.append(end_t)
        mark_p.append(final)
        mt, mp = [mark_t[0]], [mark_p[0]]
        for t, p in zip(mark_t[1:], mark_p[1:]):
            if abs(t - mt[-1]) < 1e-9 and abs(p - mp[-1]) < 1e-9:
                continue
            mt.append(t)
            mp.append(p)
        mark_t, mark_p = mt, mp
        msize = 5.5

    for t, p in zip(mark_t, mark_p):
        ax.plot(
            [t], [p],
            marker="o", markersize=msize, color=_risk_color(p),
            markeredgecolor="white", markeredgewidth=0.65, zorder=5,
        )

    ax.annotate(
        f"{baseline:.2f}",
        xy=(0.0, baseline),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left", va="center",
        fontsize=7.2, fontweight="bold", color="0.20", zorder=8,
    )
    if long_panel:
        ax.annotate(
            f"{final:.2f}",
            xy=(mark_t[-1], mark_p[-1]),
            xytext=(-2, 8),
            textcoords="offset points",
            ha="right", va="bottom",
            fontsize=7.2, fontweight="bold", color="0.20", zorder=8,
        )
    elif pid == CURATED_NEG:
        ax.annotate(
            f"{final:.2f}",
            xy=(mark_t[-1], mark_p[-1]),
            xytext=(0, -9),
            textcoords="offset points",
            ha="center", va="top",
            fontsize=7.2, fontweight="bold", color="0.20", zorder=8,
        )
    else:
        ax.annotate(
            f"{final:.2f}",
            xy=(mark_t[-1], mark_p[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left", va="center",
            fontsize=7.2, fontweight="bold", color="0.20", zorder=8,
        )

    # Guides: per-bin only on short panels; week guides on long ones.
    if not long_panel:
        for b in buckets:
            ax.axvline(
                float(b["bucket_hour"]), color="0.75", lw=0.55,
                linestyle=(0, (2.0, 2.0)), alpha=0.55, zorder=1,
            )
    else:
        for week in np.arange(168.0, xmax_plot + 1e-9, 168.0):
            ax.axvline(
                float(week), color="0.88", lw=0.5,
                linestyle=(0, (2.5, 2.5)), alpha=0.5, zorder=1,
            )

    if aki_hours is not None and np.isfinite(aki_hours):
        ax.axvline(
            float(aki_hours), color=RED, lw=1.05, linestyle=(0, (3.2, 2.0)),
            alpha=0.85, zorder=3.5,
        )
        ax.text(
            float(aki_hours) - 0.15, RISK_YLIM[1] - 0.03, "PO-AKI diagnosis",
            ha="right", va="top",
            fontsize=7.0, fontweight="bold", color=RED,
            clip_on=False, zorder=8,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.88),
        )

    if long_panel:
        # Axes coords: pin to the right margin (data-x was hard to read on long span).
        ax.text(
            0.985, RISK_THR + 0.018, f"Decision threshold: {RISK_THR:.2f}",
            transform=ax.get_yaxis_transform(),  # x=axes, y=data
            ha="right", va="bottom",
            fontsize=6.8, fontweight="bold", color="#4a5568",
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white", edgecolor="#4a5568", linewidth=0.65, alpha=0.92,
            ),
            zorder=7,
        )
    else:
        thr_x = min(xmax_plot - 0.15, max(8.0, 0.42 * xmax_plot))
        ax.text(
            thr_x, RISK_THR + 0.018, f"Decision threshold: {RISK_THR:.2f}",
            ha="right", va="bottom",
            fontsize=6.8, fontweight="bold", color="#4a5568",
            clip_on=False,
            bbox=dict(
                boxstyle="round,pad=0.15",
                facecolor="white", edgecolor="#4a5568", linewidth=0.65, alpha=0.92,
            ),
            zorder=7,
        )

    xpad = max(1.2, 0.012 * xmax_plot)
    ax.set_xlim(-xpad, xmax_plot + xpad)
    ax.set_ylim(*RISK_YLIM)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    if xmax_plot <= 24:
        xticks = [0.0] + [float(b["bucket_hour"]) for b in buckets]
        if truncate_at is not None:
            xticks.append(float(truncate_at))
        else:
            xticks.append(end_t)
        xticks = sorted({round(t, 6) for t in xticks})
    elif long_panel:
        xticks = _long_xticks(xmax_plot)
    else:
        step = 12.0
        xticks = [0.0] + [float(t) for t in np.arange(step, xmax_plot, step)]
        xticks.append(float(xmax_plot))
        xticks = sorted({round(t, 6) for t in xticks})
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [_fmt_hour_tick(t) for t in xticks],
        fontsize=7.0 if long_panel else 8.0,
    )
    if long_panel and len(xticks) >= 2:
        labs = ax.get_xticklabels()
        labs[-1].set_fontweight("bold")
        labs[-1].set_color("0.15")
    ax.set_xlabel("Postoperative time (h)")
    ax.set_ylabel("Predicted PO-AKI probability")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.22)

    head = fig_style.panel_label(panel, _PANEL_ROLE[pid])
    surg, outcome = _PANEL_SUBTITLE_STEM[pid]
    risk = f"Predicted PO-AKI probability: {baseline:.2f} \u2192 {final:.2f}"
    ax.set_title("")
    ax.text(
        0.0, 1.108, head,
        transform=ax.transAxes, ha="left", va="bottom",
        **fig_style.panel_kw(ax.figure), clip_on=False, zorder=30,
    )
    ax.text(
        0.0, 1.052, surg,
        transform=ax.transAxes, ha="left", va="bottom",
        **fig_style.subtitle_kw(ax.figure), clip_on=False, zorder=30,
    )
    ax.text(
        0.0, 1.002, f"{outcome} | {risk}",
        transform=ax.transAxes, ha="left", va="bottom",
        **fig_style.subtitle_kw(ax.figure), clip_on=False, zorder=30,
    )


def plot_case_contrast(force_infer: bool = False) -> Path:
    recs = _load_or_infer([CURATED_POS, CURATED_NEG], force=force_infer)
    FIG.mkdir(parents=True, exist_ok=True)

    # Compact side-by-side layout: less content than event-arrival force bars.
    fig_w, fig_h = 11.2, 4.95
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = fig.add_gridspec(
        1, 2,
        width_ratios=[1, 1.35],
        wspace=0.22,
        top=0.76,
        bottom=0.16,
        left=0.07,
        right=0.985,
    )

    ax_a = fig.add_subplot(outer[0])
    # AKI+ truncated at diagnosis time, not the full 6 h bin end.
    _draw_panel(
        ax_a, recs[CURATED_POS], panel="A", pid=CURATED_POS,
        aki_hours=POS_AKI_H, x_max=POS_AKI_H,
    )

    ax_b = fig.add_subplot(outer[1])
    _draw_panel(
        ax_b, recs[CURATED_NEG], panel="B", pid=CURATED_NEG,
        aki_hours=None, x_max=PLOT_HORIZON_H,
    )

    fig_style.add_suptitle(
        fig, "Time-series Transformer trajectories for the same representative episodes", y=0.975,
    )

    out = FIG / "ts_transformer_case_studies.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force-infer", action="store_true")
    args = ap.parse_args()
    out = plot_case_contrast(force_infer=args.force_infer)
    print(out)


if __name__ == "__main__":
    main()
