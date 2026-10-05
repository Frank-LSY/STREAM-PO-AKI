#!/usr/bin/env python3
"""Landmark calibration evaluation for the event-stream Transformer.

Builds risk-set outcomes at L∈{0,6,12,24,48,168}h with carry-forward predictions,
computes Brier / BSS / calibration intercept & slope, and writes a 5-panel
equal-frequency calibration figure plus a summary table.

Smoke example::

    python evaluate_landmark_calibration.py \\
      --checkpoint .../fold1/best_model.pt \\
      --episode_path .../7d_window_clm.mixed.test.jsonl \\
      --output_dir .../landmark_calibration_smoke \\
      --max_episodes 400 --n_boot 200 --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

from common.landmark_calibration import (
    DEFAULT_HORIZON_H,
    DEFAULT_LANDMARKS_H,
    aggregate_fold_summaries,
    build_landmark_table,
    rows_to_records,
    summarize_all_landmarks,
    summary_table_records,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json
from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import (
    build_eval_dataloader,
    build_model_from_checkpoint,
    load_checkpoint_bundle,
)
from event_stream.evaluate_clinical import collect_episode_trajectories


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, default="", help="Single fold checkpoint.")
    p.add_argument(
        "--checkpoints",
        type=str,
        default="",
        help="Comma-separated fold checkpoints (overrides --checkpoint).",
    )
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_episodes", type=int, default=0, help="0 = all.")
    p.add_argument("--sample_n", type=int, default=0, help="Random subsample after load.")
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--landmarks_h", type=str, default="0,6,12,24,48,168")
    p.add_argument("--horizon_h", type=float, default=DEFAULT_HORIZON_H)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=160)
    return p.parse_args()


def _parse_landmarks(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _landmark_label(h: float) -> str:
    if abs(h) < 1e-9:
        return "Baseline (0 h)"
    if abs(h - 168.0) < 1e-9:
        return "7 d"
    if abs(h - int(h)) < 1e-9:
        return f"{int(h)} h"
    return f"{h:g} h"


def plot_landmark_calibration(
    summaries: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Landmark calibration — Event-stream Transformer",
    dpi: int = 160,
    show_fold_curves: bool = False,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(summaries)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n + 0.6, 3.55), sharey=True)
    if n == 1:
        axes = [axes]
    letters = "ABCDEF"
    color = "#1f6cb0"
    for i, (ax, s) in enumerate(zip(axes, summaries)):
        curve = s.get("curve") or s.get("curve_mean") or {}
        # Legacy: faint per-fold traces only if requested AND no fold CI band.
        has_fold_ci = (
            curve.get("bin_obs_lo") is not None
            and curve.get("bin_obs_hi") is not None
            and curve.get("ci") == "fold_minmax"
        )
        if show_fold_curves and s.get("curves_by_fold") and not has_fold_ci:
            for c in s["curves_by_fold"]:
                xp = c.get("bin_pred") or []
                yp = c.get("bin_obs") or []
                if xp and yp:
                    ax.plot(xp, yp, color="0.75", lw=1.0, alpha=0.85, zorder=2)
            curve = s.get("curve_mean") or curve

        xp = list(curve.get("bin_pred") or [])
        yp = list(curve.get("bin_obs") or [])
        ylo = curve.get("bin_obs_lo")
        yhi = curve.get("bin_obs_hi")
        ax.plot([0, 1], [0, 1], color="0.55", ls=(0, (3, 2)), lw=1.0, zorder=1)
        if xp and yp:
            if ylo is not None and yhi is not None and len(ylo) == len(yp):
                # 5-fold: solid = mean, shade = fold min–max CI band.
                # Single-fold bootstrap: keep light vertical whiskers.
                if has_fold_ci or curve.get("ci") == "fold_minmax":
                    ax.fill_between(
                        xp, ylo, yhi, color=color, alpha=0.22, linewidth=0, zorder=2,
                    )
                else:
                    ax.vlines(
                        xp, ylo, yhi, colors=color, lw=1.2, alpha=0.55, zorder=3,
                    )
            ax.plot(
                xp, yp, color=color, lw=1.7, marker="o", markersize=4.2, zorder=4,
            )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        lab = _landmark_label(float(s["landmark_h"]))
        letter = letters[i] if i < len(letters) else str(i + 1)
        ax.set_title(f"{letter}  {lab}", fontsize=10.0, fontweight="bold", loc="left")
        ax.set_xlabel("Predicted probability", fontsize=8.5)
        if i == 0:
            ax.set_ylabel("Observed proportion", fontsize=8.5)
        n_risk = s.get("n_at_risk", s.get("n_at_risk_mean", "?"))
        n_aki = s.get("n_future_aki", "?")
        rate = s.get("future_aki_rate")
        rate_s = f"{100 * rate:.1f}%" if rate is not None and np.isfinite(rate) else "?"
        ax.text(
            0.03, 0.97,
            f"N={n_risk}\nAKI={n_aki} ({rate_s})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.2, color="0.35",
        )
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.22)

    fig.suptitle(title, fontsize=12.0, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_landmark_calibration_with_hist(
    rows_by_landmark: Dict[float, Any],
    summaries: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Landmark calibration — Event-stream Transformer",
    dpi: int = 160,
) -> None:
    """5-panel reliability curves with a light predicted-risk histogram under each."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(summaries)
    fig = plt.figure(figsize=(2.6 * n + 0.5, 4.15))
    gs = GridSpec(
        2, n,
        height_ratios=[3.2, 0.85],
        hspace=0.08,
        wspace=0.28,
        left=0.06, right=0.99, top=0.88, bottom=0.10,
    )
    letters = "ABCDEF"
    color = "#1f6cb0"
    for i, s in enumerate(summaries):
        ax = fig.add_subplot(gs[0, i])
        ax_h = fig.add_subplot(gs[1, i], sharex=ax)
        curve = s.get("curve") or s.get("curve_mean") or {}
        xp = list(curve.get("bin_pred") or [])
        yp = list(curve.get("bin_obs") or [])
        ylo = curve.get("bin_obs_lo")
        yhi = curve.get("bin_obs_hi")
        fold_ci = curve.get("ci") == "fold_minmax"
        ax.plot([0, 1], [0, 1], color="0.55", ls=(0, (3, 2)), lw=1.0, zorder=1)
        if xp and yp:
            if ylo is not None and yhi is not None and len(ylo) == len(yp):
                if fold_ci:
                    ax.fill_between(
                        xp, ylo, yhi, color=color, alpha=0.22, linewidth=0, zorder=2,
                    )
                else:
                    ax.vlines(xp, ylo, yhi, colors=color, lw=1.15, alpha=0.5, zorder=3)
            ax.plot(xp, yp, color=color, lw=1.7, marker="o", markersize=4.0, zorder=4)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        lab = _landmark_label(float(s["landmark_h"]))
        letter = letters[i] if i < len(letters) else str(i + 1)
        ax.set_title(f"{letter}  {lab}", fontsize=10.0, fontweight="bold", loc="left")
        if i == 0:
            ax.set_ylabel("Observed PO-AKI proportion", fontsize=8.2)
        n_risk = s.get("n_at_risk", "?")
        n_aki = s.get("n_future_aki", "?")
        rate = s.get("future_aki_rate")
        rate_s = f"{100 * rate:.1f}%" if rate is not None and np.isfinite(rate) else "?"
        ax.text(
            0.03, 0.97,
            f"N={n_risk}  ·  AKI {n_aki} ({rate_s})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.0, color="0.35",
        )
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.22)
        ax.tick_params(labelbottom=False)

        L = float(s["landmark_h"])
        ps = rows_by_landmark.get(L)
        if ps is not None and len(ps):
            ax_h.hist(
                np.asarray(ps, dtype=float),
                bins=20, range=(0, 1),
                color="0.55", alpha=0.55, edgecolor="none",
            )
        ax_h.set_xlim(0, 1)
        ax_h.set_xlabel("Predicted probability", fontsize=8.0)
        if i == 0:
            ax_h.set_ylabel("Count", fontsize=7.5)
        for sp in ("top", "right"):
            ax_h.spines[sp].set_visible(False)
        ax_h.tick_params(labelsize=7.0)

    fig.suptitle(title, fontsize=12.0, fontweight="bold", y=0.98)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_summary_csv(summaries: Sequence[Dict[str, Any]], path: Path) -> None:
    recs = summary_table_records(summaries)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not recs:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        w.writeheader()
        w.writerows(recs)


def write_summary_markdown(summaries: Sequence[Dict[str, Any]], path: Path) -> None:
    lines = [
        "| Landmark | N at risk | Future AKI n (%) | Brier | BSS | Intercept | Slope | % BASE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        lab = _landmark_label(float(s["landmark_h"]))
        n = s.get("n_at_risk", float("nan"))
        n_aki = s.get("n_future_aki", float("nan"))
        rate = s.get("future_aki_rate")
        rate_s = f"{100 * rate:.1f}" if rate is not None and np.isfinite(rate) else "?"
        def fmt(x, nd=3):
            return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "—"
        frac_base = s.get("frac_used_base")
        lines.append(
            f"| {lab} | {int(n) if np.isfinite(n) else '—'} | "
            f"{int(n_aki) if np.isfinite(n_aki) else '—'} ({rate_s}%) | "
            f"{fmt(s.get('brier'))} | {fmt(s.get('brier_skill'))} | "
            f"{fmt(s.get('calibration_intercept'))} | {fmt(s.get('calibration_slope'))} | "
            f"{fmt(100 * frac_base, 1) if frac_base is not None and np.isfinite(frac_base) else '—'} |"
        )
    path.write_text("\n".join(lines) + "\n")


def run_one_checkpoint(
    checkpoint: str,
    episodes: List[Dict[str, Any]],
    *,
    device_str: str,
    batch_size: int,
    landmarks_h: Sequence[float],
    horizon_h: float,
    n_bins: int,
    n_boot: int,
    seed: int,
    fold: int,
) -> Dict[str, Any]:
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(checkpoint)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    # Write a temporary view via dataset loader: build dataloader from in-memory list
    # by saving is heavy; instead use PeriopEpisodeDataset directly.
    from torch.utils.data import DataLoader
    from event_stream.dataset import PeriopEpisodeDataset, collate_fn

    dataset = PeriopEpisodeDataset(
        episodes, code_vocab, lab_stats=lab_stats, age_stats=age_stats, num_med_exposure=num_med
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(device_str)
    model.to(device)
    model.eval()

    print(f"  fold={fold}  collecting trajectories on {len(episodes)} episodes @ {device}", flush=True)
    trajs = collect_episode_trajectories(model, loader, device, episodes)
    rows, meta = build_landmark_table(
        trajs, landmarks_h=landmarks_h, horizon_h=horizon_h, fold=fold
    )
    summaries = summarize_all_landmarks(
        rows, landmarks_h=landmarks_h, n_bins=n_bins, n_boot=n_boot, seed=seed + fold
    )
    return {
        "fold": fold,
        "checkpoint": checkpoint,
        "meta": meta,
        "rows": rows,
        "summaries": summaries,
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    landmarks_h = _parse_landmarks(args.landmarks_h)

    episodes = load_episode_jsonl(args.episode_path)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    if 0 < args.sample_n < len(episodes):
        rng = np.random.default_rng(args.sample_seed)
        pick = rng.choice(len(episodes), size=args.sample_n, replace=False)
        episodes = [episodes[i] for i in sorted(pick)]
    print(f"episodes={len(episodes)}  landmarks={landmarks_h}", flush=True)

    ckpts: List[str] = []
    if args.checkpoints.strip():
        ckpts = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    elif args.checkpoint.strip():
        ckpts = [args.checkpoint.strip()]
    else:
        raise SystemExit("Provide --checkpoint or --checkpoints")

    fold_results = []
    all_rows = []
    for fi, ck in enumerate(ckpts, start=1):
        fold_id = fi if len(ckpts) > 1 else 1
        # Try to parse fold number from path
        for part in Path(ck).parts:
            if part.startswith("fold") and part[4:].isdigit():
                fold_id = int(part[4:])
                break
        res = run_one_checkpoint(
            ck,
            episodes,
            device_str=args.device,
            batch_size=args.batch_size,
            landmarks_h=landmarks_h,
            horizon_h=args.horizon_h,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            seed=args.seed,
            fold=fold_id,
        )
        fold_results.append(res)
        all_rows.extend(res["rows"])
        # per-fold outputs
        fold_dir = out / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        with (fold_dir / "landmark_rows.csv").open("w", newline="", encoding="utf-8") as f:
            recs = rows_to_records(res["rows"])
            if recs:
                w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
                w.writeheader()
                w.writerows(recs)
        write_summary_csv(res["summaries"], fold_dir / "landmark_calibration_table.csv")
        # strip curves for JSON size
        slim = []
        for s in res["summaries"]:
            d = {k: v for k, v in s.items() if k != "curve"}
            d["curve"] = {
                kk: s["curve"][kk]
                for kk in (
                    "n_bins", "bin_pred", "bin_obs", "bin_n",
                    "bin_obs_lo", "bin_obs_hi", "n_boot",
                )
                if kk in s["curve"]
            }
            slim.append(d)
        save_json(
            {"meta": res["meta"], "summaries": slim, "checkpoint": ck},
            fold_dir / "landmark_calibration_summary.json",
        )

    if len(fold_results) == 1:
        summaries = fold_results[0]["summaries"]
        title = "Landmark calibration (smoke) — Event-stream Transformer"
        show_folds = False
        agg = summaries
    else:
        summaries = [r["summaries"] for r in fold_results]
        agg = aggregate_fold_summaries(summaries)
        title = "Landmark calibration — Event-stream Transformer (5-fold)"
        show_folds = True
        # For table, prefer aggregated mean(min,max)
        write_summary_csv(agg, out / "landmark_calibration_table.csv")
        write_summary_markdown(agg, out / "landmark_calibration_table.md")

    if len(fold_results) == 1:
        write_summary_csv(summaries, out / "landmark_calibration_table.csv")
        write_summary_markdown(summaries, out / "landmark_calibration_table.md")
        plot_src = summaries
    else:
        plot_src = agg

    # Histogram data from rows (first fold or pooled — use all_rows)
    rows_by_L: Dict[float, List[float]] = {float(L): [] for L in landmarks_h}
    for r in all_rows:
        rows_by_L[float(r.landmark_h)].append(float(r.p))

    fig_path = out / "landmark_calibration_curves.png"
    plot_landmark_calibration_with_hist(
        rows_by_L,
        # For multi-fold agg, curve bootstrap CIs may be absent; plot mean curve.
        [
            {
                **s,
                "curve": s.get("curve") or s.get("curve_mean") or {},
                "n_at_risk": s.get("n_at_risk"),
                "n_future_aki": s.get("n_future_aki"),
                "future_aki_rate": s.get("future_aki_rate"),
            }
            for s in plot_src
        ],
        fig_path,
        title=title,
        dpi=args.dpi,
    )
    # Also keep a simple equal-panel figure without hist for paper variants
    plot_landmark_calibration(
        plot_src,
        out / "landmark_calibration_curves_simple.png",
        title=title,
        dpi=args.dpi,
        show_fold_curves=show_folds,
    )

    save_json(
        {
            "n_episodes": len(episodes),
            "n_checkpoints": len(ckpts),
            "landmarks_h": landmarks_h,
            "horizon_h": args.horizon_h,
            "n_bins": args.n_bins,
            "n_boot": args.n_boot,
            "figure": str(fig_path),
            "table_csv": str(out / "landmark_calibration_table.csv"),
            "table_md": str(out / "landmark_calibration_table.md"),
        },
        out / "run_meta.json",
    )
    print("\n=== landmark calibration table ===")
    print((out / "landmark_calibration_table.md").read_text())
    print(f"wrote {fig_path}")
    print(f"wrote {out / 'landmark_calibration_table.csv'}")


if __name__ == "__main__":
    main()
