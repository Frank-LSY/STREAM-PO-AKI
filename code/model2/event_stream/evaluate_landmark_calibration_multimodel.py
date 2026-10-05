#!/usr/bin/env python3
"""Multi-model landmark calibration smoke (ES Transformer + 3 TS models).

Overlays calibration curves for:
  - Event-stream Transformer
  - Time-series Transformer / LSTM / GRU-D

TS landmark mapping (6 h bins, prediction at bin end):
  L=0  → BASE
  L=6  → bucket [0, 6)
  L=12 → bucket [6, 12)
  L=24 → bucket [18, 24)
  L=48 → bucket [42, 48)
  L=168 → bucket [162, 168)
Missing later bins carry forward the last available bucket risk (else BASE).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from common import fig_style
from common.clinical_metrics import EpisodeTrajectory, hours_after_anchor, _parse_dt
from common.landmark_calibration import (
    DEFAULT_HORIZON_H,
    DEFAULT_LANDMARKS_H,
    LandmarkRow,
    aggregate_fold_summaries,
    build_landmark_table,
    summarize_all_landmarks,
    summarize_landmark,
    summary_table_records,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json
from event_stream.dataset import PeriopEpisodeDataset, collate_fn, load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.evaluate_clinical import collect_episode_trajectories
from event_stream.evaluate_landmark_calibration import (
    _landmark_label,
    _parse_landmarks,
    write_summary_markdown,
)
from timeseries.checkpoint import load_bucket_checkpoint
from timeseries.dataset import Ts6hEpisodeDataset
from timeseries.utils_eval import forward_bucket_model
from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, BUCKET_SIZE_H, NUM_POSTOP_BUCKETS
from common.paths import PROJECT_ROOT


MODEL_SPECS = (
    {
        "key": "static_logistic",
        "label": "Logistic Regression",
        "color": "#6E6E78",
        "family": "static",
        "pred_parquet": str(PROJECT_ROOT / "result/static/logistic/eval_fold1_test/test_predictions.parquet"),
    },
    {
        "key": "static_xgboost",
        "label": "XGBoost",
        "color": "#C98A2E",
        "family": "static",
        "pred_parquet": str(PROJECT_ROOT / "result/static/xgboost/eval_fold1_test/test_predictions.parquet"),
    },
    {
        "key": "ts_lstm",
        "label": "LSTM",
        "color": "#7B62A8",
        "family": "timeseries",
        "arch_dir": "lstm",
    },
    {
        "key": "ts_grud",
        "label": "GRU-D",
        "color": "#3F8F5B",
        "family": "timeseries",
        "arch_dir": "grud",
    },
    {
        "key": "ts_transformer",
        "label": "Time-series Transformer",
        "color": "#3A6FB0",
        "family": "timeseries",
        "arch_dir": "transformer",
    },
    {
        "key": "es_transformer",
        "label": "STREAM (proposed)",
        "color": "#C4453C",
        "family": "event_stream",
    },
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--es_checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "result/event_stream/transformer/fold1/best_model.pt"),
    )
    p.add_argument(
        "--ts_result_root",
        type=str,
        default=str(PROJECT_ROOT / "result/timeseries"),
    )
    p.add_argument("--fold", type=int, default=1)
    p.add_argument(
        "--folds",
        type=str,
        default="",
        help="Comma-separated folds for 5-fold aggregation (e.g. 1,2,3,4,5). "
        "Overrides --fold when set; solid=mean, shade=fold min–max.",
    )
    p.add_argument(
        "--es_result_root",
        type=str,
        default=str(PROJECT_ROOT / "result/event_stream/transformer"),
        help="ES checkpoints at {es_result_root}/fold{k}/best_model.pt when --folds is set.",
    )
    p.add_argument(
        "--episode_path",
        type=str,
        default=str(PROJECT_ROOT / "data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl"),
    )
    p.add_argument(
        "--ts_npz",
        type=str,
        default=str(PROJECT_ROOT / "data/five_folds_ts6h4/test.npz"),
    )
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--ts_batch_size", type=int, default=64)
    p.add_argument("--max_episodes", type=int, default=400)
    p.add_argument("--landmarks_h", type=str, default="0,6,12,24,48,168")
    p.add_argument("--horizon_h", type=float, default=DEFAULT_HORIZON_H)
    p.add_argument("--n_bins", type=int, default=10)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument(
        "--reuse_summary_json",
        type=str,
        default="",
        help="Reuse prior multimodel_summary.json for ES/TS curves; only recompute static + replot.",
    )
    p.add_argument(
        "--static_root",
        type=str,
        default=str(PROJECT_ROOT / "result/static"),
        help="Root with xgboost/logistic eval_fold{k}_test/test_predictions.parquet",
    )
    return p.parse_args()


def _t_aki_hours_from_meta(label_aki: int, t0: Any, aki_time: Any) -> Optional[float]:
    if int(label_aki) == 0:
        return float("inf")
    h = hours_after_anchor(_parse_dt(t0), _parse_dt(aki_time))
    if h is None or not np.isfinite(h):
        return None
    return float(h)


def collect_es_trajectories(
    checkpoint: str,
    episodes: List[Dict[str, Any]],
    *,
    device_str: str,
    batch_size: int,
) -> List[EpisodeTrajectory]:
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(checkpoint)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    dataset = PeriopEpisodeDataset(
        episodes, code_vocab, lab_stats=lab_stats, age_stats=age_stats, num_med_exposure=num_med
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(device_str)
    model.to(device)
    model.eval()
    return collect_episode_trajectories(model, loader, device, episodes)


def ts_record_to_trajectory(
    *,
    person_id: str,
    label_aki: int,
    t_aki_hours: Optional[float],
    baseline_prob: float,
    token_probs: np.ndarray,
    max_postop_bucket: int,
) -> EpisodeTrajectory:
    """Map TS bucket probs to a trajectory with hours = bin end times."""
    last_j = int(min(int(max_postop_bucket), NUM_POSTOP_BUCKETS - 1))
    hours: List[float] = []
    probs: List[float] = []
    if last_j >= 0 and np.isfinite(last_j):
        for j in range(last_j + 1):
            idx = BASE_BUCKET_IDX + 1 + j
            hours.append(float((j + 1) * BUCKET_SIZE_H))  # bin end
            probs.append(float(token_probs[idx]))
    return EpisodeTrajectory(
        person_id=str(person_id),
        label_aki=int(label_aki),
        t_aki_hours=None if (t_aki_hours is None or not np.isfinite(t_aki_hours)) else float(t_aki_hours),
        postop_hours=np.asarray(hours, dtype=np.float64),
        postop_probs=np.asarray(probs, dtype=np.float64),
        baseline_prob=float(baseline_prob),
        postop_patient_days=float(hours[-1] / 24.0) if hours else 0.0,
    )


@torch.no_grad()
def collect_ts_trajectories(
    checkpoint: str,
    ts_npz: Path,
    episode_meta: Dict[str, Dict[str, Any]],
    keep_pids: Sequence[str],
    *,
    device_str: str,
    batch_size: int,
) -> List[EpisodeTrajectory]:
    device = resolve_torch_device(device_str)
    model, _ckpt, _args, _layout = load_bucket_checkpoint(checkpoint, device)
    ds = Ts6hEpisodeDataset(ts_npz, load_obs=bool(getattr(model, "uses_obs_mask", False)))
    pid_to_idx = {str(p): i for i, p in enumerate(ds.person_ids)}
    indices = [pid_to_idx[p] for p in keep_pids if p in pid_to_idx]
    missing = [p for p in keep_pids if p not in pid_to_idx]
    if missing:
        print(f"  warn: {len(missing)} person_ids missing from TS npz (e.g. {missing[:3]})", flush=True)
    subset = Subset(ds, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ds.collate_fn,
    )
    out_trajs: List[EpisodeTrajectory] = []
    for batch in loader:
        batch_t = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()
        }
        out = forward_bucket_model(model, batch_t)
        token_probs = torch.sigmoid(out["token_logits"]).cpu().numpy()
        base_probs = torch.sigmoid(out["baseline_logits_all"]).cpu().numpy()
        labels = batch_t["labels"].cpu().numpy()
        max_postop = batch_t["max_postop_bucket"].cpu().numpy()
        person_ids = batch["person_ids"]
        for i, pid in enumerate(person_ids):
            meta = episode_meta.get(str(pid), {})
            label = int(labels[i] >= 0.5)
            if label == 0:
                t_aki: Optional[float] = None  # resolve_t_aki_hours → +inf via label
            else:
                t_aki = _t_aki_hours_from_meta(1, meta.get("t0"), meta.get("aki_time"))
            out_trajs.append(
                ts_record_to_trajectory(
                    person_id=str(pid),
                    label_aki=label,
                    t_aki_hours=t_aki,
                    baseline_prob=float(base_probs[i, BASE_BUCKET_IDX]),
                    token_probs=token_probs[i],
                    max_postop_bucket=int(max_postop[i]),
                )
            )
    # Preserve keep_pids order where possible
    by_pid = {t.person_id: t for t in out_trajs}
    ordered = [by_pid[p] for p in keep_pids if p in by_pid]
    return ordered


def run_model_summaries(
    trajs: List[EpisodeTrajectory],
    *,
    landmarks_h: Sequence[float],
    horizon_h: float,
    n_bins: int,
    n_boot: int,
    seed: int,
    fold: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[float, List[float]]]:
    rows, meta = build_landmark_table(
        trajs, landmarks_h=landmarks_h, horizon_h=horizon_h, fold=fold
    )
    summaries = summarize_all_landmarks(
        rows, landmarks_h=landmarks_h, n_bins=n_bins, n_boot=n_boot, seed=seed
    )
    by_L: Dict[float, List[float]] = {float(L): [] for L in landmarks_h}
    for r in rows:
        by_L[float(r.landmark_h)].append(float(r.p))
    return summaries, meta, by_L


def static_baseline_summaries_from_parquet(
    pred_parquet: Path,
    episode_meta: Dict[str, Dict[str, Any]],
    keep_pids: Sequence[str],
    *,
    landmarks_h: Sequence[float],
    horizon_h: float,
    n_bins: int,
    n_boot: int,
    seed: int,
    fold: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Static models only contribute to the L=0 landmark (preop baseline)."""
    import pandas as pd

    df = pd.read_parquet(pred_parquet)
    df["person_id"] = df["person_id"].astype(str)
    keep = set(str(p) for p in keep_pids)
    df = df[df["person_id"].isin(keep)].copy()
    # Preserve keep_pids order
    order = {p: i for i, p in enumerate(keep_pids)}
    df["_ord"] = df["person_id"].map(order)
    df = df.sort_values("_ord")

    rows: List[LandmarkRow] = []
    n_skip = 0
    for _, r in df.iterrows():
        pid = str(r["person_id"])
        label = int(float(r["label_aki"]) >= 0.5)
        meta = episode_meta.get(pid, {})
        if label == 0:
            t_aki = float("inf")
        else:
            t_aki_opt = _t_aki_hours_from_meta(1, meta.get("t0"), meta.get("aki_time"))
            if t_aki_opt is None:
                n_skip += 1
                continue
            t_aki = float(t_aki_opt)
        # Baseline risk set: T_AKI > 0
        if not (t_aki > 0.0):
            continue
        y = int(0.0 < t_aki <= float(horizon_h))
        p = float(np.clip(float(r["pred_prob"]), 1e-6, 1.0 - 1e-6))
        rows.append(
            LandmarkRow(
                person_id=pid,
                landmark_h=0.0,
                y=y,
                p=p,
                t_aki_hours=None if not np.isfinite(t_aki) else float(t_aki),
                used_base=1,
                fold=int(fold),
            )
        )

    # Build placeholder summaries for non-zero landmarks (empty / nan)
    base = summarize_landmark(rows, 0.0, n_bins=n_bins, n_boot=n_boot, seed=seed)
    summaries: List[Dict[str, Any]] = []
    for L in landmarks_h:
        if abs(float(L)) < 1e-12:
            summaries.append(base)
        else:
            summaries.append(
                {
                    "landmark_h": float(L),
                    "n_at_risk": 0,
                    "n_future_aki": 0,
                    "future_aki_rate": float("nan"),
                    "frac_used_base": float("nan"),
                    "brier": float("nan"),
                    "brier_null": float("nan"),
                    "brier_skill": float("nan"),
                    "calibration_intercept": float("nan"),
                    "calibration_slope": float("nan"),
                    "curve": {"n_bins": 0, "bin_pred": [], "bin_obs": [], "bin_n": []},
                    "pred_hist": {"edges": [], "counts": []},
                    "static_baseline_only": True,
                }
            )
    meta = {
        "n_rows_baseline": len(rows),
        "n_skipped_episodes": n_skip,
        "pred_parquet": str(pred_parquet),
        "baseline_only": True,
    }
    return summaries, meta


def _panel_short_label(h: float) -> str:
    """Compact panel subtitle: Baseline / 6 h / … / 7 d."""
    if abs(h) < 1e-9:
        return "Surgery index (baseline)"
    if abs(h - 168.0) < 1e-9:
        return "7 d"
    if abs(h - int(h)) < 1e-9:
        return f"{int(h)} h"
    return f"{h:g} h"


def plot_multimodel_curves(
    model_summaries: Dict[str, List[Dict[str, Any]]],
    model_specs: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Calibration across postoperative time points",
    dpi: int = 160,
    zoom_max: float = 0.20,
    ncols: int = 3,
) -> None:
    """Full [0,1] calibration + zoom inset [0, zoom_max] + pred-prob histograms.

    Landmark panels are arranged in a ``nrows × ncols`` grid (default 2×3 for
    six landmarks); each panel stacks calibration over the count histogram.
    """
    import math

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    first_key = next(s["key"] for s in model_specs if s["key"] in model_summaries)
    landmarks = [float(s["landmark_h"]) for s in model_summaries[first_key]]
    n = len(landmarks)
    ncols = max(1, int(ncols))
    nrows = int(math.ceil(n / ncols))
    fig = plt.figure(figsize=(3.35 * ncols + 0.55, 4.25 * nrows + 0.55))
    outer = GridSpec(
        nrows, ncols,
        hspace=0.30,
        wspace=0.28,
        left=0.07, right=0.99, top=0.88, bottom=0.055,
    )
    letters = "ABCDEF"
    zmax = float(zoom_max)
    legend_handles = None
    legend_labels = None

    for i, L in enumerate(landmarks):
        r, c = divmod(i, ncols)
        inner = GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=outer[r, c],
            # Histogram ~25% shorter than previous 1.05 ratio.
            height_ratios=[3.15, 0.78],
            hspace=0.08,
        )
        # Main = full [0,1]; inset (upper left) = zoomed [0, zmax].
        ax = fig.add_subplot(inner[0])
        ax_h = fig.add_subplot(inner[1])

        ax.plot([0, 1], [0, 1], color="0.55", ls=(0, (3, 2)), lw=1.0, zorder=1)
        axins = inset_axes(
            ax,
            width="38%",
            height="38%",
            loc="upper left",
            # Mild left inset; y-ticks are on the right so they won't hit the main axis.
            bbox_to_anchor=(0.06, 0.0, 1.0, 1.0),
            bbox_transform=ax.transAxes,
            borderpad=0.25,
        )
        axins.plot([0, zmax], [0, zmax], color="0.55", ls=(0, (3, 2)), lw=0.8, zorder=1)

        for spec in model_specs:
            key = spec["key"]
            if key not in model_summaries:
                continue
            if spec.get("family") == "static" and abs(float(L)) > 1e-12:
                continue
            s = model_summaries[key][i]
            curve = s.get("curve") or s.get("curve_mean") or {}
            xp = [float(v) for v in (curve.get("bin_pred") or [])]
            yp = [float(v) for v in (curve.get("bin_obs") or [])]
            ylo = curve.get("bin_obs_lo")
            yhi = curve.get("bin_obs_hi")
            fold_ci = curve.get("ci") == "fold_minmax"
            if xp and yp:
                if (
                    fold_ci
                    and ylo is not None
                    and yhi is not None
                    and len(ylo) == len(yp)
                ):
                    ax.fill_between(
                        xp, ylo, yhi,
                        color=spec["color"], alpha=0.18, linewidth=0, zorder=2,
                    )
                    axins.fill_between(
                        xp, ylo, yhi,
                        color=spec["color"], alpha=0.15, linewidth=0, zorder=2,
                    )
                (line,) = ax.plot(
                    xp, yp,
                    color=spec["color"], lw=1.55, marker="o", markersize=3.5,
                    label=spec["label"], zorder=3, alpha=0.95,
                )
                axins.plot(
                    xp, yp,
                    color=spec["color"], lw=1.0, marker="o", markersize=2.0,
                    zorder=3, alpha=0.9,
                )
                if i == 0:
                    if legend_handles is None:
                        legend_handles, legend_labels = [], []
                    legend_handles.append(line)
                    legend_labels.append(spec["label"])

            # Same risk-set episodes, but each model has its own p̂ → different
            # score histograms. Use thin step outlines to de-emphasize vs curves.
            hist = s.get("pred_hist") or {}
            edges = hist.get("edges") or []
            counts = hist.get("counts") or []
            if len(edges) >= 2 and len(counts) == len(edges) - 1:
                ax_h.stairs(
                    counts, edges,
                    color=spec["color"], lw=0.85, alpha=0.75,
                    label=spec["label"],
                )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        letter = letters[i] if i < len(letters) else str(i + 1)
        fig_style.panel_title(ax, letter, _panel_short_label(L), pad=5.0)
        if c == 0:
            ax.set_ylabel("Observed PO-AKI proportion", fontsize=8.0)
        if i == 0:
            # Small in-panel cue for the identity line (also in figure legend).
            ax.text(
                0.62, 0.70, "Perfect calibration",
                transform=ax.transAxes,
                fontsize=6.2, color="0.45", rotation=45,
                rotation_mode="anchor", ha="left", va="bottom",
            )
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(alpha=0.22)
        ax.tick_params(labelbottom=False)

        axins.set_xlim(0, zmax)
        axins.set_ylim(0, zmax)
        axins.set_xticks([0.0, 0.1, 0.2])
        axins.set_yticks([0.0, 0.1, 0.2])
        axins.yaxis.tick_right()
        axins.yaxis.set_label_position("right")
        axins.tick_params(labelsize=5.4, pad=1.0)
        axins.set_title(f"0–{zmax:.2f}", fontsize=5.8, color="0.4", pad=1.2)
        for sp in axins.spines.values():
            sp.set_color("0.65")
            sp.set_linewidth(0.7)
        axins.grid(alpha=0.15)

        ax_h.set_xlim(0, 1)
        ax_h.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        # Shared x-label only on the bottom row of panels.
        if r == nrows - 1:
            ax_h.set_xlabel("Predicted PO-AKI probability", fontsize=7.0)
        else:
            ax_h.set_xlabel("")
        if c == 0:
            ax_h.set_ylabel("Count", fontsize=7.0)
        else:
            ax_h.set_ylabel("")
            ax_h.tick_params(labelleft=False)
        for sp in ("top", "right"):
            ax_h.spines[sp].set_visible(False)
        ax_h.tick_params(labelsize=6.2)
        ax_h.grid(axis="y", alpha=0.15)

    if legend_handles and legend_labels:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=len(legend_labels),
            frameon=False,
            fontsize=6.8,
            handlelength=1.6,
            columnspacing=1.00,
            handletextpad=0.32,
        )
    fig_style.add_suptitle(fig, title, y=0.995)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_multimodel_table(
    model_summaries: Dict[str, List[Dict[str, Any]]],
    model_specs: Sequence[Dict[str, Any]],
    out_csv: Path,
    out_md: Path,
) -> None:
    rows: List[Dict[str, Any]] = []
    md = [
        "| Model | Landmark | N at risk | Future AKI n (%) | Brier | BSS | Intercept | Slope |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for spec in model_specs:
        key = spec["key"]
        if key not in model_summaries:
            continue
        for s in model_summaries[key]:
            # Skip empty static placeholders for L>0
            if spec.get("family") == "static" and abs(float(s["landmark_h"])) > 1e-12:
                continue
            if int(s.get("n_at_risk") or 0) <= 0 and not np.isfinite(s.get("brier", np.nan)):
                continue
            rec = summary_table_records([s])[0]
            rec["model"] = spec["label"]
            rows.append(rec)
            rate = rec["future_aki_pct"]
            rate_s = f"{rate:.1f}" if np.isfinite(rate) else "?"

            def fmt(x, nd=3):
                return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "—"

            md.append(
                f"| {spec['label']} | {_landmark_label(float(rec['landmark_h']))} | "
                f"{int(rec['n_at_risk'])} | {int(rec['n_future_aki'])} ({rate_s}%) | "
                f"{fmt(rec['brier'])} | {fmt(rec['brier_skill'])} | "
                f"{fmt(rec['calibration_intercept'])} | {fmt(rec['calibration_slope'])} |"
            )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fields = ["model"] + [k for k in rows[0].keys() if k != "model"]
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    out_md.write_text("\n".join(md) + "\n")


def _parse_folds(folds_s: str, default_fold: int) -> List[int]:
    if folds_s.strip():
        return [int(x.strip()) for x in folds_s.split(",") if x.strip()]
    return [int(default_fold)]


def _collect_one_fold_summaries(
    args: argparse.Namespace,
    fold: int,
    episodes: List[Any],
    keep_pids: Sequence[str],
    episode_meta: Dict[str, Dict[str, Any]],
    landmarks_h: Sequence[float],
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    """Run ES + TS + static summaries for a single fold checkpoint."""
    model_summaries: Dict[str, List[Dict[str, Any]]] = {}
    model_meta: Dict[str, Any] = {}

    es_root = Path(args.es_result_root)
    es_ck = es_root / f"fold{fold}" / "best_model.pt"
    if not es_ck.exists():
        es_ck = Path(args.es_checkpoint)
    es_spec = next(s for s in MODEL_SPECS if s["key"] == "es_transformer")
    print(f"\n[fold{fold}][{es_spec['label']}] {es_ck}", flush=True)
    es_trajs = collect_es_trajectories(
        str(es_ck), episodes, device_str=args.device, batch_size=args.batch_size
    )
    summ, meta, _ = run_model_summaries(
        es_trajs,
        landmarks_h=landmarks_h,
        horizon_h=args.horizon_h,
        n_bins=args.n_bins,
        n_boot=args.n_boot,
        seed=args.seed + fold,
        fold=fold,
    )
    model_summaries[es_spec["key"]] = summ
    model_meta[es_spec["key"]] = meta

    ts_root = Path(args.ts_result_root)
    ts_npz = Path(args.ts_npz)
    for spec in MODEL_SPECS:
        if spec.get("family") != "timeseries":
            continue
        ck = ts_root / spec["arch_dir"] / f"fold{fold}" / "best_model.pt"
        if not ck.exists():
            print(f"  skip {spec['label']}: missing {ck}", flush=True)
            continue
        print(f"[fold{fold}][{spec['label']}] {ck}", flush=True)
        trajs = collect_ts_trajectories(
            str(ck),
            ts_npz,
            episode_meta,
            keep_pids,
            device_str=args.device,
            batch_size=args.ts_batch_size,
        )
        print(f"  trajectories={len(trajs)}", flush=True)
        summ, meta, _ = run_model_summaries(
            trajs,
            landmarks_h=landmarks_h,
            horizon_h=args.horizon_h,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            seed=args.seed + 17 + fold,
            fold=fold,
        )
        model_summaries[spec["key"]] = summ
        model_meta[spec["key"]] = meta

    static_root = Path(args.static_root)
    for spec in MODEL_SPECS:
        if spec.get("family") != "static":
            continue
        fold_pred = (
            static_root
            / spec["key"].replace("static_", "")
            / f"eval_fold{fold}_test"
            / "test_predictions.parquet"
        )
        pred_p = fold_pred if fold_pred.exists() else Path(spec.get("pred_parquet", ""))
        if not pred_p.exists():
            print(f"  skip {spec['label']}: missing {pred_p}", flush=True)
            continue
        print(f"[fold{fold}][{spec['label']}] {pred_p}", flush=True)
        summ, meta = static_baseline_summaries_from_parquet(
            pred_p,
            episode_meta,
            keep_pids,
            landmarks_h=landmarks_h,
            horizon_h=args.horizon_h,
            n_bins=args.n_bins,
            n_boot=args.n_boot,
            seed=args.seed + 31 + fold,
            fold=fold,
        )
        model_summaries[spec["key"]] = summ
        model_meta[spec["key"]] = meta
        print(f"  baseline N={summ[0]['n_at_risk']}  Brier={summ[0]['brier']:.3f}", flush=True)

    return model_summaries, model_meta


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    landmarks_h = _parse_landmarks(args.landmarks_h)
    folds = _parse_folds(args.folds, args.fold)
    multi_fold = len(folds) > 1

    episodes = load_episode_jsonl(args.episode_path)
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    keep_pids = [str(ep.get("person_id", "")) for ep in episodes]
    episode_meta = {
        str(ep.get("person_id", "")): {"aki_time": ep.get("aki_time"), "t0": ep.get("t0")}
        for ep in episodes
    }
    print(
        f"episodes={len(episodes)}  landmarks={landmarks_h}  folds={folds}",
        flush=True,
    )

    model_summaries: Dict[str, List[Dict[str, Any]]] = {}
    model_meta: Dict[str, Any] = {}

    reuse_path = Path(args.reuse_summary_json) if args.reuse_summary_json.strip() else None
    if reuse_path and reuse_path.exists():
        prev = json.loads(reuse_path.read_text())
        print(f"reusing summaries from {reuse_path}", flush=True)
        for k, summ in (prev.get("summaries") or {}).items():
            model_summaries[k] = summ
        model_meta.update(prev.get("meta") or {})
        # If reuse already has fold CI, keep; else leave as single-fold curves.
    else:
        per_fold: Dict[str, List[List[Dict[str, Any]]]] = {}
        for fold in folds:
            fold_summ, fold_meta = _collect_one_fold_summaries(
                args, fold, episodes, keep_pids, episode_meta, landmarks_h
            )
            for k, summ in fold_summ.items():
                per_fold.setdefault(k, []).append(summ)
            for k, meta in fold_meta.items():
                model_meta.setdefault(k, {})[f"fold{fold}"] = meta

        if multi_fold:
            for k, fold_list in per_fold.items():
                model_summaries[k] = aggregate_fold_summaries(fold_list)
        else:
            for k, fold_list in per_fold.items():
                model_summaries[k] = fold_list[0]

    used_specs = [s for s in MODEL_SPECS if s["key"] in model_summaries]
    fig_path = out / "landmark_calibration_curves_multimodel.png"
    plot_multimodel_curves(
        model_summaries,
        used_specs,
        fig_path,
        title="Calibration across postoperative time points",
        dpi=args.dpi,
    )
    write_multimodel_table(
        model_summaries,
        used_specs,
        out / "landmark_calibration_table_multimodel.csv",
        out / "landmark_calibration_table_multimodel.md",
    )

    # Slim JSON (no huge bootstrap arrays beyond needed)
    slim = {}
    for k, summ in model_summaries.items():
        slim[k] = []
        for s in summ:
            d = {kk: vv for kk, vv in s.items() if kk not in ("curve", "curves_by_fold")}
            c = s.get("curve") or s.get("curve_mean") or {}
            d["curve"] = {
                kk: c[kk]
                for kk in (
                    "n_bins", "bin_pred", "bin_obs", "bin_n", "n_boot",
                    "bin_obs_lo", "bin_obs_hi", "ci", "n_folds",
                )
                if kk in c
            }
            if "pred_hist" in s:
                d["pred_hist"] = s["pred_hist"]
            if "curve_mean" in s and "curve" not in s:
                d["curve_mean"] = s["curve_mean"]
            slim[k].append(d)
    save_json(
        {
            "n_episodes": len(episodes),
            "folds": folds,
            "landmarks_h": landmarks_h,
            "models": {s["key"]: s["label"] for s in used_specs},
            "meta": model_meta,
            "summaries": slim,
            "figure": str(fig_path),
        },
        out / "multimodel_summary.json",
    )
    print("\n" + (out / "landmark_calibration_table_multimodel.md").read_text())
    print(f"wrote {fig_path}")


if __name__ == "__main__":
    main()
