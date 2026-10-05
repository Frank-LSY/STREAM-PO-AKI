#!/usr/bin/env python3
"""Reproduce notebook/descriptive_analysis figures on five_folds_deduplicate4."""
from __future__ import annotations

import os

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(os.environ.get("PERIOP_ROOT", Path(__file__).resolve().parents[1]))


def log(msg: str) -> None:
    print(msg, flush=True)


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_dedup4(data_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Load fold1–5 + test; return (pos, neg) by label_aki."""
    eps: List[Dict[str, Any]] = []
    for k in range(1, 6):
        path = data_dir / f"7d_window_clm.mixed.fold{k}.jsonl"
        log(f"Loading {path.name} ...")
        eps.extend(iter_jsonl(path))
    test_path = data_dir / "7d_window_clm.mixed.test.jsonl"
    log(f"Loading {test_path.name} ...")
    eps.extend(iter_jsonl(test_path))
    pos = [e for e in eps if int(e.get("label_aki") or 0) == 1]
    neg = [e for e in eps if int(e.get("label_aki") or 0) == 0]
    log(f"Loaded N={len(eps)} pos={len(pos)} neg={len(neg)}")
    return pos, neg


def parse_dt(x: Any) -> Optional[datetime]:
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
    return None


EVENT_KEYS = {
    "condition": "condition_events",
    "medication": "med_events",
    "lab": "lab_events",
}


def get_event_day_deltas(
    ep: Dict[str, Any],
    event_type: str,
    t0_dt: datetime,
    preop_days: float,
    postop_days: float,
) -> List[float]:
    # Notebook originally used cond_events (empty); canonical field is condition_events.
    entries = ep.get(EVENT_KEYS[event_type]) or []
    out: List[float] = []
    for e in entries:
        if e.get("time_day") is not None:
            try:
                delta_day = float(e["time_day"])
            except (TypeError, ValueError):
                continue
        else:
            t = parse_dt(e.get("time"))
            if t is None:
                continue
            delta_day = (t - t0_dt).total_seconds() / 86400.0
        if -preop_days <= delta_day <= postop_days:
            out.append(delta_day)
    return out


def aki_time_deltas_days(pos_data: List[Dict[str, Any]]) -> List[float]:
    """AKI onset time relative to t0, in days (same unit as event-density x-axis)."""
    deltas: List[float] = []
    for ep in pos_data:
        t0 = parse_dt(ep.get("t0"))
        aki = parse_dt(ep.get("aki_time"))
        if t0 is None or aki is None:
            continue
        if aki > t0:
            deltas.append((aki - t0).total_seconds() / 86400.0)
    return deltas


EVENT_LABELS = {
    "condition": "Condition Events",
    "medication": "Medication Events",
    "lab": "Lab Events",
    "total": "Total Events",
}
EVENT_COLORS = {
    "condition": "tab:blue",
    "medication": "tab:orange",
    "lab": "tab:green",
    "total": "tab:red",
}


def _empty_sums(n_bins: int) -> Dict[str, np.ndarray]:
    return {k: np.zeros(n_bins, dtype=np.float64) for k in ("condition", "medication", "lab", "total")}


def make_hybrid_time_bins(
    preop_days: float,
    x_hi: float,
    *,
    fine_half: float = 2.0,
    fine_step: float = 0.25,
    coarse_step: float = 1.0,
) -> np.ndarray:
    """Daily bins outside surgery±fine_half; fine_step (default 6h) inside."""
    fine_lo, fine_hi = -fine_half, fine_half
    left = np.arange(-preop_days, fine_lo + 1e-9, coarse_step)
    fine = np.arange(fine_lo, fine_hi + 1e-9, fine_step)
    right = np.arange(fine_hi, x_hi + coarse_step + 1e-9, coarse_step)
    return np.unique(np.concatenate([left, fine, right]).astype(np.float64))


def counts_to_per_day_rate(counts: np.ndarray, time_bins: np.ndarray) -> np.ndarray:
    """Convert bin counts to events/patient/day so mixed-width bins are comparable."""
    widths = np.diff(time_bins).astype(np.float64)
    widths = np.maximum(widths, 1e-12)
    return counts / widths


def build_densities_single_pass(
    pos_data: List[Dict[str, Any]],
    neg_data: List[Dict[str, Any]],
    daily_bins: np.ndarray,
    overlay_bins: np.ndarray,
    preop_days: float,
    postop_days: float,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    List[float],
    List[float],
]:
    """One pass: daily rates + hybrid overlay rates + per-episode daily density."""
    n_daily = len(daily_bins) - 1
    n_over = len(overlay_bins) - 1
    n_days = preop_days + postop_days
    pos_daily_sums = _empty_sums(n_daily)
    neg_daily_sums = _empty_sums(n_daily)
    pos_over_sums = _empty_sums(n_over)
    neg_over_sums = _empty_sums(n_over)
    n_pos = 0
    n_neg = 0
    pos_daily: List[float] = []
    neg_daily: List[float] = []
    all_eps = [(ep, True) for ep in pos_data] + [(ep, False) for ep in neg_data]
    report_every = max(1, len(all_eps) // 20)
    for idx, (ep, is_pos) in enumerate(all_eps):
        if idx % report_every == 0:
            log(f"  density progress {idx}/{len(all_eps)}")
        t0 = parse_dt(ep.get("t0"))
        if t0 is None:
            continue
        dsums = pos_daily_sums if is_pos else neg_daily_sums
        osums = pos_over_sums if is_pos else neg_over_sums
        if is_pos:
            n_pos += 1
        else:
            n_neg += 1
        cond = get_event_day_deltas(ep, "condition", t0, preop_days, postop_days)
        med = get_event_day_deltas(ep, "medication", t0, preop_days, postop_days)
        lab = get_event_day_deltas(ep, "lab", t0, preop_days, postop_days)
        for key, times in (("condition", cond), ("medication", med), ("lab", lab)):
            if times:
                dsums[key] += np.histogram(times, bins=daily_bins)[0]
                osums[key] += np.histogram(times, bins=overlay_bins)[0]
        total_times = cond + med + lab
        if total_times:
            dsums["total"] += np.histogram(total_times, bins=daily_bins)[0]
            osums["total"] += np.histogram(total_times, bins=overlay_bins)[0]
        daily = len(total_times) / max(n_days, 1.0)
        if is_pos:
            pos_daily.append(daily)
        else:
            neg_daily.append(daily)
    log(f"  density done; pos={n_pos} neg={n_neg}")
    denom_p, denom_n = max(n_pos, 1), max(n_neg, 1)

    def _mean_rate(sums: Dict[str, np.ndarray], bins: np.ndarray, denom: int) -> Dict[str, np.ndarray]:
        return {k: counts_to_per_day_rate(v / denom, bins) for k, v in sums.items()}

    return (
        _mean_rate(pos_daily_sums, daily_bins, denom_p),
        _mean_rate(neg_daily_sums, daily_bins, denom_n),
        _mean_rate(pos_over_sums, overlay_bins, denom_p),
        _mean_rate(neg_over_sums, overlay_bins, denom_n),
        pos_daily,
        neg_daily,
    )


def _plot_density_on_ax(
    ax: Any,
    mean_densities: Dict[str, np.ndarray],
    time_bins: np.ndarray,
    preop_days: int,
    postop_days: int,
    title: str,
    show_legend: bool = True,
) -> None:
    x_mid = (time_bins[:-1] + time_bins[1:]) / 2
    for et in ("condition", "medication", "lab", "total"):
        ax.plot(
            x_mid,
            mean_densities[et],
            label=EVENT_LABELS[et],
            color=EVENT_COLORS[et],
            linewidth=2,
        )
    ax.axvline(0, linestyle="--", color="k", alpha=0.4)
    ax.set_xticks(np.arange(-preop_days, postop_days + 1, 1))
    ax.set_xlim(-preop_days, postop_days)
    ax.set_xlabel("Relative Time to Surgery (days)", fontsize=11)
    ax.set_ylabel("Mean Event Count per Patient per Day", fontsize=11)
    ax.set_title(title, fontsize=13)
    if show_legend:
        ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)


def _peak_info(
    x_mid: np.ndarray,
    y: np.ndarray,
    *,
    x_lo: Optional[float] = None,
    x_hi: Optional[float] = None,
) -> Tuple[float, float, int]:
    mask = np.ones(len(y), dtype=bool)
    if x_lo is not None:
        mask &= x_mid >= x_lo
    if x_hi is not None:
        mask &= x_mid <= x_hi
    if not mask.any():
        i = int(np.argmax(y))
    else:
        local = np.where(mask, y, -np.inf)
        i = int(np.argmax(local))
    return float(x_mid[i]), float(y[i]), i


def plot_density_pos_vs_neg(
    pos_densities: Dict[str, np.ndarray],
    neg_densities: Dict[str, np.ndarray],
    time_bins: np.ndarray,
    preop_days: int,
    postop_days: int,
    out_path: Path,
    *,
    xlim: Optional[Tuple[float, float]] = None,
    fine_half: float = 2.0,
    ax: Any = None,
) -> Any:
    """Overlay AKI+/− total event density on absolute scale (events/patient/day).

    Expects hybrid bins (6h near surgery±2d, daily elsewhere). Points are marked.
    """
    x_mid = (time_bins[:-1] + time_bins[1:]) / 2
    pos_y = np.asarray(pos_densities["total"], dtype=np.float64)
    neg_y = np.asarray(neg_densities["total"], dtype=np.float64)

    x_lo, x_hi = xlim if xlim is not None else (-float(preop_days), float(postop_days))
    # Annotate peri-op peak (avoid day-30 edge artifacts on the full axis)
    peak_lo, peak_hi = -7.0, 7.0
    pos_peak_x, pos_peak_raw, _ = _peak_info(x_mid, pos_y, x_lo=peak_lo, x_hi=peak_hi)
    neg_peak_x, neg_peak_raw, _ = _peak_info(x_mid, neg_y, x_lo=peak_lo, x_hi=peak_hi)

    mask = (x_mid >= x_lo) & (x_mid <= x_hi)
    c_pos = "#C44E52"
    c_neg = "#4C72B0"

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(11.5, 5.8))
    else:
        fig = ax.figure

    x = x_mid[mask]
    yp = pos_y[mask]
    yn = neg_y[mask]

    ax.axvspan(-fine_half, fine_half, color="#f0e6d2", alpha=0.35, zorder=0, label="_nolegend_")
    ax.fill_between(x, yn, 0, color=c_neg, alpha=0.10, linewidth=0)
    ax.fill_between(x, yp, 0, color=c_pos, alpha=0.10, linewidth=0)
    ax.plot(
        x,
        yn,
        color=c_neg,
        linewidth=2.0,
        marker="o",
        markersize=5.0,
        markerfacecolor=c_neg,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=f"AKI negative  (peak {neg_peak_raw:.1f}/day near day {neg_peak_x:+.2f})",
    )
    ax.plot(
        x,
        yp,
        color=c_pos,
        linewidth=2.0,
        marker="s",
        markersize=4.5,
        markerfacecolor=c_pos,
        markeredgecolor="white",
        markeredgewidth=0.7,
        label=f"AKI positive  (peak {pos_peak_raw:.1f}/day near day {pos_peak_x:+.2f})",
    )

    for px, py, color in (
        (neg_peak_x, neg_peak_raw, c_neg),
        (pos_peak_x, pos_peak_raw, c_pos),
    ):
        if x_lo <= px <= x_hi:
            ax.axvline(px, color=color, linestyle=":", linewidth=1.1, alpha=0.55)
            ax.scatter([px], [py], s=48, color=color, zorder=6, edgecolors="white", linewidths=0.8)

    ax.axvline(0, color="#333333", linestyle="--", linewidth=1.2, alpha=0.65)
    ax.set_xlim(x_lo, x_hi)
    y_max = max(float(yp.max()) if len(yp) else 0.0, float(yn.max()) if len(yn) else 0.0, 1.0)
    ax.set_ylim(0, y_max * 1.08)
    # Daily tick labels across the full window
    xticks = np.arange(int(np.ceil(x_lo)), int(np.floor(x_hi)) + 1, 1)
    ax.set_xticks(xticks)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Relative Time to Surgery (days)", fontsize=11)
    ax.set_ylabel("Mean event count per patient per day", fontsize=11)
    ax.set_title(
        "Event Density Relative to Surgery: AKI+ vs AKI−\n"
        f"(6h bins in day [{-fine_half:g}, {fine_half:g}]; daily bins elsewhere)",
        fontsize=12,
    )
    ax.legend(fontsize=8.5, loc="upper right", frameon=True, fancybox=False, edgecolor="#dddddd")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if own_fig:
        fig.tight_layout()
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        log(f"Wrote {out_path}")
    return ax


def plot_combined(
    aki_deltas_days: List[float],
    pos_overlay: Dict[str, np.ndarray],
    neg_overlay: Dict[str, np.ndarray],
    overlay_bins: np.ndarray,
    preop_days: int,
    postop_days: int,
    out_path: Path,
) -> None:
    """2-panel figure: AKI onset (days) + overlayed AKI+/− event density."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.2), gridspec_kw={"width_ratios": [1.0, 1.35]})

    ax1 = axes[0]
    aki_bins = np.arange(0, postop_days + 1, 1)
    ax1.hist(aki_deltas_days, bins=aki_bins, color="skyblue", edgecolor="k", density=True)
    ax1.set_xticks(np.arange(0, postop_days + 1, 1))
    ax1.set_xlim(0, postop_days)
    ax1.set_xlabel("AKI Occurrence Time Relative to Surgery Index (days)", fontsize=11)
    ax1.set_ylabel("Proportion of patients (positive AKI)", fontsize=11)
    ax1.set_title("Distribution of AKI Occurrence Time Relative to Surgery Index", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5)

    plot_density_pos_vs_neg(
        pos_overlay,
        neg_overlay,
        overlay_bins,
        preop_days,
        postop_days,
        out_path,
        xlim=(-float(preop_days), float(postop_days)),
        fine_half=2.0,
        ax=axes[1],
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log(f"Wrote {out_path}")


def plot_density_only(
    mean_densities: Dict[str, np.ndarray],
    time_bins: np.ndarray,
    preop_days: int,
    postop_days: int,
    out_path: Path,
    title: str = "Distribution of Event Density Relative to Surgery",
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_density_on_ax(ax, mean_densities, time_bins, preop_days, postop_days, title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    log(f"Wrote {out_path}")


def save_density_cache(
    path: Path,
    daily_bins: np.ndarray,
    overlay_bins: np.ndarray,
    pos_daily_d: Dict[str, np.ndarray],
    neg_daily_d: Dict[str, np.ndarray],
    pos_overlay: Dict[str, np.ndarray],
    neg_overlay: Dict[str, np.ndarray],
    pos_daily: List[float],
    neg_daily: List[float],
    aki_deltas: List[float],
) -> None:
    np.savez_compressed(
        path,
        daily_bins=daily_bins,
        overlay_bins=overlay_bins,
        aki_deltas=np.asarray(aki_deltas, dtype=np.float64),
        pos_daily=np.asarray(pos_daily, dtype=np.float64),
        neg_daily=np.asarray(neg_daily, dtype=np.float64),
        **{f"pos_d_{k}": v for k, v in pos_daily_d.items()},
        **{f"neg_d_{k}": v for k, v in neg_daily_d.items()},
        **{f"pos_o_{k}": v for k, v in pos_overlay.items()},
        **{f"neg_o_{k}": v for k, v in neg_overlay.items()},
    )
    log(f"Wrote {path}")


def load_density_cache(path: Path) -> Dict[str, Any]:
    z = np.load(path)
    keys = ("condition", "medication", "lab", "total")
    return {
        "daily_bins": z["daily_bins"],
        "overlay_bins": z["overlay_bins"],
        "pos_daily_d": {k: z[f"pos_d_{k}"] for k in keys},
        "neg_daily_d": {k: z[f"neg_d_{k}"] for k in keys},
        "pos_overlay": {k: z[f"pos_o_{k}"] for k in keys},
        "neg_overlay": {k: z[f"neg_o_{k}"] for k in keys},
        "pos_daily": z["pos_daily"].tolist(),
        "neg_daily": z["neg_daily"].tolist(),
        "aki_deltas": z["aki_deltas"].tolist(),
    }


def plot_boxplot(
    pos_d: List[float],
    neg_d: List[float],
    out_path: Path,
) -> Dict[str, float]:
    fig = plt.figure(figsize=(7, 6))
    plt.boxplot(
        [pos_d, neg_d],
        tick_labels=["AKI", "Non-AKI"],
        patch_artist=True,
        showmeans=True,
        boxprops=dict(facecolor="#65a4e3"),
        medianprops=dict(color="k"),
        meanprops=dict(marker="o", markerfacecolor="red", markeredgecolor="k"),
    )
    plt.ylabel("Mean Event Density per Day (Total Events)", fontsize=12)
    plt.title("Event Density in AKI vs Non-AKI Episodes", fontsize=14)
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    stats = {
        "pos_mean": float(np.mean(pos_d)),
        "neg_mean": float(np.mean(neg_d)),
        "pos_median": float(np.median(pos_d)),
        "neg_median": float(np.median(neg_d)),
        "n_pos_density": len(pos_d),
        "n_neg_density": len(neg_d),
    }
    log(f"Wrote {out_path}")
    log(
        f"density mean AKI={stats['pos_mean']:.3f} Non-AKI={stats['neg_mean']:.3f} "
        f"median AKI={stats['pos_median']:.3f} Non-AKI={stats['neg_median']:.3f}"
    )
    return stats


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
        default=(PROJECT_ROOT / "result/tables/descriptive_analysis_dedup4"),
    )
    p.add_argument("--preop_days", type=int, default=7)
    p.add_argument("--postop_days", type=int, default=30)
    p.add_argument("--overlay_x_hi", type=float, default=30.0)
    p.add_argument("--fine_half", type=float, default=2.0, help="Surgery ±days for 6h bins")
    p.add_argument("--fine_step", type=float, default=0.25, help="Fine bin width in days (0.25=6h)")
    p.add_argument(
        "--from_cache",
        action="store_true",
        help="Skip JSONL scan; load densities.npz from output_dir if present.",
    )
    args = p.parse_args()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "densities.npz"

    daily_bins = np.arange(-args.preop_days, args.postop_days + 1, 1, dtype=np.float64)
    overlay_bins = make_hybrid_time_bins(
        args.preop_days,
        args.overlay_x_hi,
        fine_half=args.fine_half,
        fine_step=args.fine_step,
    )

    if args.from_cache and cache_path.is_file():
        log(f"Loading cache {cache_path}")
        cache = load_density_cache(cache_path)
        if "overlay_bins" not in cache:
            raise SystemExit("Cache missing overlay_bins; rerun without --from_cache")
        daily_bins = cache["daily_bins"]
        overlay_bins = cache["overlay_bins"]
        pos_daily_d = cache["pos_daily_d"]
        neg_daily_d = cache["neg_daily_d"]
        pos_overlay = cache["pos_overlay"]
        neg_overlay = cache["neg_overlay"]
        pos_daily = cache["pos_daily"]
        neg_daily = cache["neg_daily"]
        aki_deltas = cache["aki_deltas"]
        n_pos, n_neg = len(pos_daily), len(neg_daily)
    else:
        pos_data, neg_data = load_dedup4(args.data_dir)
        n_pos, n_neg = len(pos_data), len(neg_data)
        aki_deltas = aki_time_deltas_days(pos_data)
        log(f"AKI time deltas (days) n={len(aki_deltas)}")
        log(
            f"Building densities: daily bins={len(daily_bins)-1}, "
            f"overlay hybrid bins={len(overlay_bins)-1} "
            f"(fine ±{args.fine_half}d step={args.fine_step})"
        )
        (
            pos_daily_d,
            neg_daily_d,
            pos_overlay,
            neg_overlay,
            pos_daily,
            neg_daily,
        ) = build_densities_single_pass(
            pos_data,
            neg_data,
            daily_bins,
            overlay_bins,
            args.preop_days,
            args.postop_days,
        )
        save_density_cache(
            cache_path,
            daily_bins,
            overlay_bins,
            pos_daily_d,
            neg_daily_d,
            pos_overlay,
            neg_overlay,
            pos_daily,
            neg_daily,
            aki_deltas,
        )

    plot_combined(
        aki_deltas,
        pos_overlay,
        neg_overlay,
        overlay_bins,
        args.preop_days,
        args.postop_days,
        out_dir / "descriptive_analysis.png",
    )
    plot_density_pos_vs_neg(
        pos_overlay,
        neg_overlay,
        overlay_bins,
        args.preop_days,
        args.postop_days,
        out_dir / "event_density_aki_pos_vs_neg.png",
        xlim=(-float(args.preop_days), float(args.postop_days)),
        fine_half=args.fine_half,
    )
    plot_density_only(
        pos_daily_d,
        daily_bins,
        args.preop_days,
        args.postop_days,
        out_dir / "event_density_vs_surgery_aki_pos.png",
        title="Event Density Relative to Surgery (AKI Positive)",
    )
    plot_density_only(
        neg_daily_d,
        daily_bins,
        args.preop_days,
        args.postop_days,
        out_dir / "event_density_vs_surgery_aki_neg.png",
        title="Event Density Relative to Surgery (AKI Negative)",
    )
    box_stats = plot_boxplot(
        pos_daily,
        neg_daily,
        out_dir / "event_density_aki_vs_nonaki.png",
    )

    x_mid = (overlay_bins[:-1] + overlay_bins[1:]) / 2
    pos_peak_x, pos_peak_y, _ = _peak_info(
        x_mid, pos_overlay["total"], x_lo=-args.preop_days, x_hi=args.postop_days
    )
    neg_peak_x, neg_peak_y, _ = _peak_info(
        x_mid, neg_overlay["total"], x_lo=-args.preop_days, x_hi=args.postop_days
    )
    meta = {
        "data_dir": str(args.data_dir),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_aki_time_deltas": len(aki_deltas),
        "aki_time_unit": "days",
        "preop_days": args.preop_days,
        "postop_days": args.postop_days,
        "overlay": {
            "x_hi": args.overlay_x_hi,
            "fine_half_days": args.fine_half,
            "fine_step_days": args.fine_step,
            "n_bins": int(len(overlay_bins) - 1),
            "y_scale": "absolute_events_per_patient_per_day",
        },
        "total_peak": {
            "aki_pos_day": pos_peak_x,
            "aki_pos_mean_per_day": pos_peak_y,
            "aki_neg_day": neg_peak_x,
            "aki_neg_mean_per_day": neg_peak_y,
        },
        "boxplot": box_stats,
        "note": (
            "Overlay uses absolute Total density; 6h bins in surgery±2d, daily elsewhere; "
            "x-axis -preop..+postop; rates are events/patient/day (bin counts / bin width)."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"Wrote {out_dir / 'summary.json'}")
    log("Done.")


if __name__ == "__main__":
    main()
