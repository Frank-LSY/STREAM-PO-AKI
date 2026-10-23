#!/usr/bin/env python3
"""Make expressive figures for within-bin order sensitivity (from CSV + optional path replay)."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import fig_style
from common.paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "result/figures/order_sensitivity"
ES_CKPT = ROOT / "result/event_stream/transformer/fold1/best_model.pt"
TEST_JSONL = ROOT / "data/five_folds_deduplicate4/7d_window_clm.mixed.test.jsonl"
RED, BLUE = "#c0392b", "#1f6cb0"
GRAY = "0.45"
# Case streams: avoid classic red/blue pairing (easy to misread as partisan).
ORIG_COL = "#0f766e"  # teal
REV_COL = "#a16207"   # ochre
CASE_PATH_DATA = OUT / "case_path_data.json"
T_LABS = ["0\u20136 h", "6\u201324 h", "24\u201372 h", "72 h\u20137 d"]
K_LABS = ["3\u20134", "5\u20138", "9\u201316", "17\u201332", "\u226533"]


def _ecdf(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def fig_a_es_vs_ts(es: pd.DataFrame, ts: pd.DataFrame, out: Path) -> None:
    """Panel A: ECDF + paired summary of |Δz| for ES vs TS."""
    es_abs = es["abs_delta_logit"].to_numpy(dtype=float)
    ts_abs = ts["abs_delta_logit"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.35, 1.0]})

    ax = axes[0]
    xe, ye = _ecdf(es_abs)
    xt, yt = _ecdf(ts_abs)
    ax.plot(xe, ye, color=BLUE, lw=2.0, label="Event-stream")
    ax.plot(xt, yt, color=RED, lw=2.0, label="Time-series (6 h bag)")
    ax.axvline(0.05, color="0.6", ls=":", lw=0.9)
    ax.text(0.05, 0.05, "0.05", fontsize=7.5, color="0.45", ha="left", va="bottom")
    ax.set_xlabel(r"$|\Delta z|$ at end of 6 h bin")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("A  Order perturbation changes ES, not TS", fontsize=10.5, fontweight="bold", loc="left")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    # Summary bars: median |Δz|
    med_es = float(np.median(es_abs))
    med_ts = float(np.median(ts_abs)) if len(ts_abs) else 0.0
    p90_es = float(np.percentile(es_abs, 90))
    p90_ts = float(np.percentile(ts_abs, 90)) if len(ts_abs) else 0.0
    x = np.arange(2)
    w = 0.36
    ax.bar(x - w / 2, [med_es, med_ts], width=w, color=[BLUE, RED], alpha=0.88, label="Median")
    ax.bar(x + w / 2, [p90_es, p90_ts], width=w, color=[BLUE, RED], alpha=0.45, label="P90")
    ax.set_xticks(x)
    ax.set_xticklabels(["Event-stream", "Time-series"])
    ax.set_ylabel(r"$|\Delta z|$")
    ax.set_title("Summary", fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    # annotate zeros for TS
    if med_ts == 0 and p90_ts == 0:
        ax.text(1, 0.02 * max(p90_es, med_es, 1e-3), r"$\equiv 0$", ha="center", va="bottom",
                fontsize=9, color=RED, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_b_vs_k(es: pd.DataFrame, out: Path) -> None:
    """Panel B: |Δz| vs number of events in the bin."""
    # one row per window×perm already; also show window-level mean
    win = (
        es.groupby(["person_id", "bin_j", "k", "n_distinct"], as_index=False)["abs_delta_logit"]
        .mean()
        .rename(columns={"abs_delta_logit": "mean_abs_dz"})
    )

    # bin k into groups for boxplot
    edges = [3, 5, 9, 17, 33, 10_000]
    labels = ["3–4", "5–8", "9–16", "17–32", "≥33"]
    win["k_bin"] = pd.cut(win["k"], bins=edges, right=False, labels=labels)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    # scatter all window means
    ax.scatter(
        win["k"], win["mean_abs_dz"],
        s=16, alpha=0.35, c=BLUE, edgecolors="none", zorder=2, label="Window mean",
    )
    # trend: median in k groups
    xs, meds, los, his = [], [], [], []
    for lab in labels:
        sub = win[win["k_bin"] == lab]["mean_abs_dz"]
        if sub.empty:
            continue
        # place at mid of group using median k in group
        kk = win.loc[win["k_bin"] == lab, "k"]
        xs.append(float(kk.median()))
        meds.append(float(sub.median()))
        los.append(float(sub.quantile(0.25)))
        his.append(float(sub.quantile(0.75)))
    ax.plot(xs, meds, color=RED, lw=2.0, marker="o", markersize=6, zorder=3, label="Group median")
    ax.fill_between(xs, los, his, color=RED, alpha=0.15, zorder=1)

    ax.set_xlabel("Events in 6 h bin ($k$)")
    ax.set_ylabel(r"Mean $|\Delta z_{\mathrm{ES}}|$ over permutations")
    ax.set_title("B  Larger within-bin sequences → stronger order sensitivity",
                 fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)


def fig_d_vs_postop_time(es: pd.DataFrame, out: Path) -> Dict[str, Any]:
    """Panel D: |Δz| vs hours since surgery (which 6h bin)."""
    win = (
        es.groupby(["person_id", "bin_j", "h0", "h1", "k", "n_distinct"], as_index=False)[
            "abs_delta_logit"
        ]
        .mean()
        .rename(columns={"abs_delta_logit": "mean_abs_dz"})
    )
    edges = [0, 6, 24, 72, 168]
    labels = ["0–6 h", "6–24 h", "24–72 h", "72 h–7 d"]
    win["tbin"] = pd.cut(win["h0"], bins=edges, right=False, labels=labels)

    fig, axes = plt.subplots(
        1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    ax = axes[0]
    ax.scatter(
        win["h0"], win["mean_abs_dz"],
        s=14, alpha=0.30, c=BLUE, edgecolors="none", zorder=2,
    )
    # median + IQR by coarse postop strata, placed at stratum mid
    mids = [3.0, 15.0, 48.0, 120.0]
    meds, los, his = [], [], []
    stats = {}
    for lab, mid in zip(labels, mids):
        sub = win.loc[win["tbin"] == lab, "mean_abs_dz"]
        kk = win.loc[win["tbin"] == lab, "k"]
        if sub.empty:
            meds.append(np.nan)
            los.append(np.nan)
            his.append(np.nan)
            continue
        meds.append(float(sub.median()))
        los.append(float(sub.quantile(0.25)))
        his.append(float(sub.quantile(0.75)))
        stats[lab] = {
            "n": int(len(sub)),
            "median_abs_dz": float(sub.median()),
            "mean_abs_dz": float(sub.mean()),
            "median_k": float(kk.median()),
        }
    ax.plot(mids, meds, color=RED, lw=2.0, marker="o", markersize=6.5, zorder=3, label="Stratum median")
    ax.fill_between(mids, los, his, color=RED, alpha=0.15, zorder=1, label="IQR")
    ax.set_xlabel("Postoperative time of bin start (h since surgery)")
    ax.set_ylabel(r"Mean $|\Delta z_{\mathrm{ES}}|$ over permutations")
    ax.set_title("D  Order sensitivity by time since surgery", fontsize=10.5, fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    data, pos, xticklabels = [], [], []
    for i, lab in enumerate(labels):
        sub = win.loc[win["tbin"] == lab, "mean_abs_dz"].to_numpy(dtype=float)
        if sub.size == 0:
            continue
        data.append(sub)
        pos.append(i)
        n = stats[lab]["n"]
        mk = stats[lab]["median_k"]
        xticklabels.append(f"{lab}\n(n={n}, med $k$={mk:.0f})")
    bp = ax.boxplot(
        data, positions=pos, widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color=RED, lw=1.6),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(BLUE)
        patch.set_alpha(0.35)
    ax.set_xticks(pos)
    ax.set_xticklabels(xticklabels, fontsize=7.8)
    ax.set_ylabel(r"Mean $|\Delta z_{\mathrm{ES}}|$")
    ax.set_title("By postoperative stratum", fontsize=10.5, fontweight="bold", loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return stats


def _bootstrap_median_ci(
    values: np.ndarray,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Return (median, ci_lo, ci_hi) from percentile bootstrap of the median."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(x))
    if x.size < 3:
        # Too few points for a stable CI; collapse to the point estimate.
        return med, med, med
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.median(x[idx], axis=1)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return med, lo, hi


def _window_abs_dz(es: pd.DataFrame) -> pd.DataFrame:
    """One row per eligible window: mean |Δz| across the given permutation rows."""
    return (
        es.groupby(["person_id", "bin_j", "h0", "h1", "k", "n_distinct"], as_index=False)[
            "abs_delta_logit"
        ]
        .mean()
        .rename(columns={"abs_delta_logit": "mean_abs_dz"})
    )


def _add_joint_bins(win: pd.DataFrame, t_labs: List[str], k_labs: List[str]) -> pd.DataFrame:
    t_edges = [0, 6, 24, 72, 168]
    k_edges = [3, 5, 9, 17, 33, 10_000]
    out = win.copy()
    out["tbin"] = pd.cut(out["h0"], bins=t_edges, right=False, labels=t_labs)
    out["kbin"] = pd.cut(out["k"], bins=k_edges, right=False, labels=k_labs)
    return out


def _joint_med_cnt(win: pd.DataFrame, t_labs: List[str], k_labs: List[str]):
    med = win.pivot_table(
        index="tbin", columns="kbin", values="mean_abs_dz", aggfunc="median", observed=True,
    ).reindex(index=t_labs, columns=k_labs)
    cnt = (
        win.pivot_table(
            index="tbin", columns="kbin", values="mean_abs_dz", aggfunc="count", observed=True,
        )
        .reindex(index=t_labs, columns=k_labs)
        .fillna(0)
        .astype(int)
    )
    return med, cnt


def _draw_joint_heatmap(ax, med, cnt, t_labs, k_labs, title, vmin, vmax, show_ylabel=True):
    z = med.to_numpy(dtype=float)
    im = ax.imshow(z, aspect="auto", cmap="YlOrRd", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(k_labs)))
    ax.set_xticklabels(k_labs)
    ax.set_yticks(np.arange(len(t_labs)))
    ax.set_yticklabels(t_labs)
    ax.set_xlabel("Events per 6-h window ($k$)")
    if show_ylabel:
        ax.set_ylabel("Postoperative time")
    ax.set_title(title, loc="left", pad=5, **fig_style.subtitle_kw(ax.figure))
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            n = int(cnt.iloc[i, j])
            val = z[i, j]
            if n <= 0 or not np.isfinite(val):
                txt, tcolor = "—", "0.4"
            else:
                txt = f"{val:.2f}\n(n={n:,})"
                tcolor = "white" if val >= 0.5 * (vmin + vmax) else "0.15"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.0, color=tcolor)
    return im


def _draw_joint_lines(ax, win, t_labs, k_labs, title, seed_offset=0) -> Dict[str, float]:
    colors = ["#c0392b", "#d68910", "#1f6cb0", "#1e8449"]
    k_mids = [3.5, 6.5, 12.5, 24.5, 48.0]
    corrs: Dict[str, float] = {}
    for ti, (lab, color) in enumerate(zip(t_labs, colors)):
        sub = win[win["tbin"] == lab]
        if sub.empty:
            continue
        corrs[lab] = float(sub["k"].corr(sub["mean_abs_dz"]))
        xs, ys, los, his = [], [], [], []
        for kb, mid in zip(k_labs, k_mids):
            s = sub.loc[sub["kbin"] == kb, "mean_abs_dz"].to_numpy(dtype=float)
            if s.size == 0:
                continue
            m, lo, hi = _bootstrap_median_ci(s, seed=1000 + seed_offset + ti * 17 + len(xs))
            xs.append(mid)
            ys.append(m)
            los.append(lo)
            his.append(hi)
        if xs:
            ax.fill_between(xs, los, his, color=color, alpha=0.18, linewidth=0, zorder=1)
            ax.plot(xs, ys, color=color, lw=1.8, marker="o", markersize=5.0, label=lab, zorder=3)
    ax.set_xlabel("Events per 6-h window ($k$, group midpoint)")
    ax.set_ylabel(r"Median $|\Delta\mathrm{logit}|$")
    ax.set_title(title, loc="left", pad=5, **fig_style.subtitle_kw(ax.figure))
    ax.legend(frameon=False, fontsize=7.5, title="Postoperative time", title_fontsize=7.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    return corrs


def fig_e_joint_k_time(es: pd.DataFrame, out: Path) -> Dict[str, Any]:
    """Panel E: random within-bin order sensitivity by k and time since surgery."""
    t_labs, k_labs = T_LABS, K_LABS

    rnd_mask = es["perm"].astype(str).str.startswith("random")
    n_random = int(es.loc[rnd_mask, "perm"].nunique()) if rnd_mask.any() else 0
    rnd = _add_joint_bins(_window_abs_dz(es[rnd_mask]), t_labs, k_labs)
    med_n, cnt_n = _joint_med_cnt(rnd, t_labs, k_labs)

    finite = med_n.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size:
        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            vmax = vmin + 1e-6
    else:
        vmin, vmax = 0.0, 1.0

    r_lab = f"mean of {n_random}" if n_random else "random"
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    fig, axes = plt.subplots(
        1, 2, figsize=(11.8, 4.0),
        gridspec_kw={"width_ratios": [1.18, 1.05], "wspace": 0.30},
    )
    im = _draw_joint_heatmap(
        axes[0], med_n, cnt_n, t_labs, k_labs,
        r"Median $|\Delta\mathrm{logit}|$ by postoperative time and $k$", vmin, vmax,
    )
    corrs_n = _draw_joint_lines(
        axes[1], rnd, t_labs, k_labs,
        r"Median $|\Delta\mathrm{logit}|$ vs $k$ (95% bootstrap CI)", seed_offset=200,
    )
    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="3.8%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label(r"Median $|\Delta\mathrm{logit}|$", fontsize=8.0)

    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_random": n_random,
        "random": {
            "heatmap_median": med_n.where(cnt_n > 0).to_dict(),
            "heatmap_n": cnt_n.to_dict(),
            "within_time_corr_k": corrs_n,
        },
        # Backward-compatible aggregate key = random (primary analysis).
        "within_time_corr_k": corrs_n,
    }


def _load_episode(pid: str) -> Dict[str, Any]:
    return _load_episodes([pid])[str(pid)]


def _load_episodes(pids: Sequence[Any]) -> Dict[str, Dict[str, Any]]:
    import sys
    sys.path.insert(0, str(ROOT / "code/model2"))
    from event_stream.dataset import load_episode_jsonl

    want = {str(p) for p in pids}
    # Also accept integer-normalized forms.
    want |= {str(int(float(p))) for p in pids if str(p).replace(".", "", 1).isdigit()}
    out: Dict[str, Dict[str, Any]] = {}
    for ep in load_episode_jsonl(TEST_JSONL):
        key = str(ep.get("person_id"))
        keys = {key}
        if key.replace(".", "", 1).isdigit():
            keys.add(str(int(float(key))))
        hit = keys & want
        if not hit:
            continue
        for k in hit:
            out[k] = ep
        if len(out) >= len(want):
            break
    missing = want - set(out)
    if missing:
        raise KeyError(f"episodes not found: {sorted(missing)[:5]}")
    return out


def _postop_hours(episode: Dict[str, Any]) -> List[float]:
    hrs = []
    for ev in episode.get("events") or []:
        th = ev.get("time_hour")
        if th is None:
            continue
        thf = float(th)
        if thf < 0 or str(ev.get("modality", "")).lower() == "procedure":
            continue
        hrs.append(thf)
    return hrs


def _aki_hours_postop(episode: Dict[str, Any]) -> Optional[float]:
    """Hours from index surgery (t0) to real AKI onset, or None if unlabeled/missing."""
    from datetime import datetime

    if int(episode.get("label_aki", 0) or 0) != 1:
        return None
    t0, aki = episode.get("t0"), episode.get("aki_time")
    if not t0 or not aki:
        return None
    try:
        t0d = datetime.fromisoformat(str(t0).replace(" ", "T"))
        akid = datetime.fromisoformat(str(aki).replace(" ", "T"))
    except Exception:
        return None
    return (akid - t0d).total_seconds() / 3600.0


AKI_MARK = "#2d6a4f"


def _truncate_path(path: Dict[str, Any], t_max: float) -> Dict[str, Any]:
    keep = [i for i, th in enumerate(path["t"]) if th <= t_max]
    return {
        "t": [path["t"][i] for i in keep],
        "z": [path["z"][i] for i in keep],
        "p": [path["p"][i] for i in keep],
        "codes": [path["codes"][i] for i in keep],
        "z_base": path["z_base"],
        "p_base": path["p_base"],
    }


def _bin_indices(episode: Dict[str, Any], h0: float, h1: float) -> List[int]:
    idxs = []
    for i, ev in enumerate(episode.get("events") or []):
        th = ev.get("time_hour")
        if th is None:
            continue
        thf = float(th)
        if thf < 0 or str(ev.get("modality", "")).lower() == "procedure":
            continue
        if h0 <= thf < h1:
            idxs.append(i)
    return idxs


def _permute_event_list(events, indices, order):
    """Chronology-preserving: keep destination timestamps; move identity+value."""
    out = list(events)
    block = [copy.deepcopy(events[i]) for i in indices]
    slot_times = [
        {"time_hour": events[i].get("time_hour"), "time_day": events[i].get("time_day")}
        for i in indices
    ]
    for loc, src in enumerate(order):
        ev = copy.deepcopy(block[src])
        for k, v in slot_times[loc].items():
            if v is not None:
                ev[k] = v
        out[indices[loc]] = ev
    return out


def _path_in_bin(model, episode, code_vocab, lab_stats, num_med, device, h0, h1, age_stats=None):
    path = _path_postop(
        model, episode, code_vocab, lab_stats, num_med, device, age_stats=age_stats,
    )
    keep = [i for i, th in enumerate(path["t"]) if h0 <= th < h1]
    return {
        "t": [path["t"][i] for i in keep],
        "z": [path["z"][i] for i in keep],
        "p": [path["p"][i] for i in keep],
        "codes": [path["codes"][i] for i in keep],
        "z_base": path["z_base"],
        "p_base": path["p_base"],
    }


def _path_postop(model, episode, code_vocab, lab_stats, num_med, device, age_stats=None):
    """Full postoperative risk path; x is hours since index surgery."""
    from event_stream.event_attribution import episode_to_batch, forward_logits, logit_to_prob

    batch, seq, _ = episode_to_batch(
        episode, code_vocab, lab_stats, num_med, device, age_stats=age_stats
    )
    z_base, _base_idx, postop_idx, z_postop = forward_logits(model, batch)
    ts, zs, ps, codes = [], [], [], []
    for pos, z in zip(postop_idx, z_postop):
        th = float(seq[pos].get("time_hour", -1))
        if th < 0:
            continue
        ts.append(th)
        zs.append(float(z))
        ps.append(logit_to_prob(z))
        codes.append(str(seq[pos].get("raw_code", "")))
    return {
        "t": ts, "z": zs, "p": ps, "codes": codes,
        "z_base": float(z_base), "p_base": logit_to_prob(z_base),
    }


def _reverse_within_all_bins(episode: Dict[str, Any], bucket_h: float = 6.0) -> Dict[str, Any]:
    """Chronology-preserving reverse of event identity within every postoperative 6-h bin."""
    ep = copy.deepcopy(episode)
    events = list(ep.get("events") or [])
    # Collect bin starts that actually contain eligible postoperative events.
    bin_starts = sorted({
        float(bucket_h) * int(float(ev["time_hour"]) // float(bucket_h))
        for ev in events
        if ev.get("time_hour") is not None
        and float(ev["time_hour"]) >= 0
        and str(ev.get("modality", "")).lower() != "procedure"
    })
    for h0 in bin_starts:
        idxs = _bin_indices(ep, h0, h0 + float(bucket_h))
        if len(idxs) < 2:
            continue
        order = list(range(len(idxs)))[::-1]
        events = _permute_event_list(events, idxs, order)
        ep["events"] = events
    return ep


def _logit_at_or_before(path: Dict[str, Any], t_end: float) -> Optional[float]:
    z = None
    for th, zz in zip(path["t"], path["z"]):
        if th < t_end:
            z = zz
        else:
            break
    return z


def _event_minute_clusters(
    times: Sequence[float],
    min_count: int = 5,
    merge_within_min: int = 2,
) -> List[Tuple[float, int]]:
    """Return (hour, count) for same-minute bursts (≥ min_count), merging nearby minutes."""
    from collections import Counter

    keys = sorted(Counter(int(round(float(t) * 60.0)) for t in times).items())
    if not keys:
        return []
    # Merge adjacent minute buckets within merge_within_min.
    groups: List[List[Tuple[int, int]]] = [[keys[0]]]
    for minute, n in keys[1:]:
        if minute - groups[-1][-1][0] <= merge_within_min:
            groups[-1].append((minute, n))
        else:
            groups.append([(minute, n)])
    out: List[Tuple[float, int]] = []
    for g in groups:
        total = sum(n for _, n in g)
        if total < min_count:
            continue
        # Weight center by event count.
        center = sum(m * n for m, n in g) / total
        out.append((center / 60.0, total))
    out.sort(key=lambda x: -x[1])
    return out


def _mark_aki_onset(ax, aki_hours: Optional[float], *, h0: float, h1: float, label: bool) -> None:
    """Vertical marker for real AKI onset when it falls in (or near) the plotted window."""
    if aki_hours is None or not np.isfinite(aki_hours):
        return
    # Allow a small overhang so an onset exactly at the last event is still drawn.
    if aki_hours < h0 - 0.05 or aki_hours > h1 + 0.35:
        return
    ax.axvline(aki_hours, color=AKI_MARK, ls="--", lw=1.35, alpha=0.95, zorder=4)
    if label:
        # Place the label in the empty mid-window region; arrow points to onset.
        ax.annotate(
            f"AKI onset  {aki_hours:.1f} h",
            xy=(aki_hours, 0.50),
            xytext=(max(h0 + 0.4, aki_hours - 1.55), 0.72),
            textcoords="data",
            ha="center", va="center", fontsize=6.6, color=AKI_MARK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=AKI_MARK, lw=0.9,
                            connectionstyle="arc3,rad=0.05"),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=AKI_MARK,
                      alpha=0.92, lw=0.7),
            zorder=5,
        )


def _style_time_axis(ax, h0: float, h1: float, *, label_bottom: bool) -> np.ndarray:
    xticks = np.arange(h0, h1 + 0.01, 0.5)
    ax.set_xlim(h0 - 0.12, h1 + 0.12)
    ax.set_xticks(xticks)
    ax.set_xticks(np.arange(h0, h1 + 0.01, 0.25), minor=True)
    ax.tick_params(labelsize=7.2, labelbottom=label_bottom)
    ax.grid(axis="x", which="major", alpha=0.12)
    if label_bottom:
        ax.set_xticklabels([f"{t:g}" if t < h1 else f"{t:g} h" for t in xticks])
        ax.set_xlabel("Hours since index surgery", fontsize=8.2)
    return xticks


def _draw_case_path(
    ax,
    path: Dict[str, Any],
    title: str,
    color: str,
    *,
    h0: float = 0.0,
    h1: float = 6.0,
    n_events: Optional[int] = None,
    aki_hours: Optional[float] = None,
    annotate: bool = False,
    shade_bursts: bool = False,
    show_ylabel: bool = True,
) -> None:
    """One probability path. AKI / scope note only when annotate=True (real stream)."""
    xs = np.asarray(path["t"], dtype=float)
    ys = np.asarray(path["p"], dtype=float)
    n = len(ys)

    # Burst bands first (same timestamps under reverse); labels live on the density strip.
    if shade_bursts or annotate:
        for th, _cnt in _event_minute_clusters(xs.tolist(), min_count=5):
            ax.axvspan(th - 0.06, th + 0.06, color=color, alpha=0.12, zorder=0, lw=0)

    if n:
        ax.plot(xs, ys, color=color, lw=1.7, marker="o", markersize=2.6,
                drawstyle="steps-post", zorder=3)
    ax.axhline(0.5, color="0.5", ls=(0, (3, 2)), lw=0.9, alpha=0.85)
    ax.axvline(h1, color="0.55", ls=":", lw=0.95, alpha=0.95, zorder=1)
    if annotate:
        _mark_aki_onset(ax, aki_hours, h0=h0, h1=h1, label=True)
    ax.set_title(title, loc="left", pad=2, **fig_style.subtitle_kw(ax.figure))
    _style_time_axis(ax, h0, h1, label_bottom=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    ax.set_ylim(-0.02, 1.05)
    if show_ylabel:
        ax.set_ylabel("Predicted PO-AKI\nprobability", fontsize=8.0)
    else:
        ax.tick_params(labelleft=False)

    if n:
        ax.annotate(
            f"{ys[0]:.2f}", xy=(xs[0], ys[0]),
            xytext=(3, 7 if ys[0] < 0.5 else -11), textcoords="offset points",
            fontsize=6.6, color="0.35",
        )
        # End value sits just above the path tip (AKI callout is mid-height, to the left).
        end_dy = 8 if ys[-1] < 0.85 else -12
        ax.annotate(
            f"at {h1:g} h: {ys[-1]:.2f}", xy=(xs[-1], ys[-1]),
            xytext=(-2, end_dy), textcoords="offset points",
            fontsize=6.8, color="0.2", fontweight="bold", ha="right",
        )

    if annotate:
        note = f"Only the {h0:g}–{h1:g} h window"
        if n_events is not None:
            note += f"  ·  {n_events} events"
        ax.text(
            0.02, 0.96, note, transform=ax.transAxes, ha="left", va="top",
            fontsize=6.6, color="0.35", style="italic",
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="0.85", alpha=0.9),
        )


def _draw_shared_density(
    ax,
    times: Sequence[float],
    *,
    h0: float,
    h1: float,
    aki_hours: Optional[float] = None,
) -> None:
    """Single event-arrival histogram (timestamps identical under reverse)."""
    xs = np.asarray(times, dtype=float)
    bin_w = 0.25
    edges = np.arange(h0, h1 + bin_w, bin_w)
    hist, _ = np.histogram(xs, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.bar(
        centers, hist, width=bin_w * 0.92, color="0.35", alpha=0.55, align="center",
        edgecolor="0.25", linewidth=0.35,
    )
    ax.axvline(h1, color="0.55", ls=":", lw=0.95, alpha=0.95)
    _mark_aki_onset(ax, aki_hours, h0=h0, h1=h1, label=False)
    _style_time_axis(ax, h0, h1, label_bottom=True)
    ymax = max(int(hist.max()) if len(hist) else 1, 1)
    ax.set_ylim(0, ymax * 1.45)
    ax.set_ylabel("Events /\n0.25 h", fontsize=7.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

    # Burst callouts on the density strip (shared by both paths).
    for th, cnt in _event_minute_clusters(xs.tolist(), min_count=5):
        # Height of the 0.25 h bin containing this burst.
        bi = int(np.clip(np.searchsorted(edges, th, side="right") - 1, 0, len(hist) - 1))
        y_bar = float(hist[bi]) if len(hist) else float(cnt)
        ax.annotate(
            f"{cnt} events\n≈{th:.1f} h",
            xy=(th, y_bar), xytext=(0, 6), textcoords="offset points",
            fontsize=6.4, color="0.2", ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="0.45",
                      alpha=0.9, lw=0.6),
        )


def _draw_case_stack(
    ax_orig,
    ax_rev,
    ax_dens,
    path0, path1, dz: float,
    *,
    highlight_h0: Optional[float] = None,
    highlight_h1: Optional[float] = None,
    n_events: Optional[int] = None,
    aki_hours: Optional[float] = None,
) -> None:
    """Stacked B panel: real stream (annotated) → reverse → shared density once."""
    h0 = float(highlight_h0 if highlight_h0 is not None else 0.0)
    h1 = float(highlight_h1 if highlight_h1 is not None else 6.0)
    _draw_case_path(
        ax_orig, path0, "Original arrival order", ORIG_COL,
        h0=h0, h1=h1, n_events=n_events, aki_hours=aki_hours,
        annotate=True, shade_bursts=True, show_ylabel=True,
    )
    _draw_case_path(
        ax_rev, path1,
        rf"Reversed arrival order  ·  $|\Delta\mathrm{{logit}}|$ at {h1:g} h = {dz:.2f}",
        REV_COL,
        h0=h0, h1=h1, n_events=n_events, aki_hours=None,
        annotate=False, shade_bursts=True, show_ylabel=True,
    )
    _draw_shared_density(
        ax_dens, path0["t"], h0=h0, h1=h1, aki_hours=aki_hours,
    )


def _select_illustrative_case(
    es: pd.DataFrame,
    *,
    min_k: int = 20,
    n_scan: int = 40,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Pick a high-|Δlogit| reverse example from the first postoperative 0–6 h window."""
    rev = es[es["perm"].astype(str) == "reverse"].copy()
    g = rev if not rev.empty else es
    g = g[(g["h0"] == 0.0) & (g["h1"] == 6.0)].sort_values("abs_delta_logit", ascending=False)
    if g.empty:
        g = (
            es.groupby(["person_id", "bin_j", "h0", "h1", "k", "n_distinct"], as_index=False)["abs_delta_logit"]
            .mean()
            .sort_values("abs_delta_logit", ascending=False)
        )

    cand_pids = []
    seen = set()
    for _, row in g.iterrows():
        pid = str(int(row.person_id)) if float(row.person_id).is_integer() else str(row.person_id)
        if pid in seen:
            continue
        if int(row.k) < min_k:
            continue
        seen.add(pid)
        cand_pids.append(pid)
        if len(cand_pids) >= n_scan:
            break

    episodes = _load_episodes(cand_pids)
    # Prefer windows where most postoperative events in the plotted span fall inside 0–6 h.
    scored: List[Tuple[Tuple[float, float], pd.Series, Dict[str, Any]]] = []
    for _, row in g.iterrows():
        pid = str(int(row.person_id)) if float(row.person_id).is_integer() else str(row.person_id)
        if pid not in episodes or int(row.k) < min_k:
            continue
        hrs = _postop_hours(episodes[pid])
        if not hrs:
            continue
        early = [h for h in hrs if 0.0 <= h < 6.0]
        if len(early) < min_k:
            continue
        frac_early = len(early) / max(len([h for h in hrs if h < 48.0]), 1)
        scored.append(((float(row.abs_delta_logit), frac_early), row, episodes[pid]))

    if scored:
        scored.sort(key=lambda x: x[0], reverse=True)
        _score, row, ep = scored[0]
        return row, ep

    top = g.iloc[0]
    pid = str(int(top.person_id)) if float(top.person_id).is_integer() else str(top.person_id)
    return top, episodes[pid]


def fig_c_case_path(es: pd.DataFrame, out: Path, device_str: str = "auto") -> Dict[str, Any]:
    """Panel B: 0–6 h window only; x = hours since surgery; reverse within this window."""
    import sys
    sys.path.insert(0, str(ROOT / "code/model2"))
    from common.med_exposure import infer_num_med_exposure
    from common.utils import resolve_torch_device
    from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
    from event_stream.event_attribution import logit_to_prob

    top, episode = _select_illustrative_case(es)
    pid = str(int(top.person_id)) if float(top.person_id).is_integer() else str(top.person_id)
    h0, h1 = float(top.h0), float(top.h1)
    assert h0 == 0.0 and h1 == 6.0, (h0, h1)
    idxs = _bin_indices(episode, h0, h1)
    assert len(idxs) == int(top.k), (len(idxs), top.k, pid, h0, h1)

    device = resolve_torch_device(device_str)
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(str(ES_CKPT))
    if num_med < 0:
        num_med = infer_num_med_exposure([episode])
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Restrict the illustration to this single 0–6 h window.
    path0 = _path_in_bin(model, episode, code_vocab, lab_stats, num_med, device, h0, h1, age_stats)
    ep_rev = copy.deepcopy(episode)
    order = list(range(len(idxs)))[::-1]
    ep_rev["events"] = _permute_event_list(episode["events"], idxs, order)
    path1 = _path_in_bin(model, ep_rev, code_vocab, lab_stats, num_med, device, h0, h1, age_stats)

    z0 = path0["z"][-1] if path0["z"] else None
    z1 = path1["z"][-1] if path1["z"] else None
    if z0 is not None and z1 is not None:
        dz = float(abs(z1 - z0))
    else:
        dz = float(top.abs_delta_logit)

    aki_h = _aki_hours_postop(episode)
    CASE_PATH_DATA.write_text(json.dumps({
        "path_orig": path0,
        "path_rev": path1,
        "abs_delta_logit_reverse": dz,
        "highlight_h0": h0,
        "highlight_h1": h1,
        "person_id": pid,
        "n_events": int(top.k),
        "aki_hours": aki_h,
        "scope": "0-6h_window_only",
    }))

    fig = plt.figure(figsize=(6.4, 6.6))
    gs = fig.add_gridspec(
        3, 1, height_ratios=[1.15, 1.15, 0.55], hspace=0.22,
        left=0.14, right=0.98, top=0.94, bottom=0.08,
    )
    ax_o = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax_o, sharey=ax_o)
    ax_d = fig.add_subplot(gs[2], sharex=ax_o)
    _draw_case_stack(
        ax_o, ax_r, ax_d, path0, path1, dz,
        highlight_h0=h0, highlight_h1=h1, n_events=int(top.k), aki_hours=aki_h,
    )
    fig.savefig(out, dpi=170, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    meta = {
        "person_id": pid,
        "bin_j": int(top.bin_j),
        "h0": h0,
        "h1": h1,
        "k": int(top.k),
        "n_distinct": int(top.n_distinct),
        "abs_delta_logit_reverse": dz,
        "scope": "0-6h_window_only",
        "aki_hours": aki_h,
        "p_end_orig": float(path0["p"][-1]) if path0["p"] else None,
        "p_end_rev": float(path1["p"][-1]) if path1["p"] else None,
        "p_at_highlight_orig": float(logit_to_prob(z0)) if z0 is not None else None,
        "p_at_highlight_rev": float(logit_to_prob(z1)) if z1 is not None else None,
    }
    return meta


def fig_main_A_joint_B_case(
    out: Path | None = None,
    meta_c: Dict[str, Any] | None = None,
) -> Path:
    """Paper main figure, drawn natively: A = joint k × time; B = reversed-order case."""
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    out = out or (OUT / "order_sensitivity_main_A_joint_B_case.png")
    if not CASE_PATH_DATA.exists():
        raise FileNotFoundError(f"{CASE_PATH_DATA} (run without --skip_c once)")
    case = json.loads(CASE_PATH_DATA.read_text())
    es = pd.read_csv(OUT / "order_sensitivity_es_rows.csv")
    rnd = _add_joint_bins(
        _window_abs_dz(es[es["perm"].astype(str).str.startswith("random")]), T_LABS, K_LABS,
    )
    med, cnt = _joint_med_cnt(rnd, T_LABS, K_LABS)
    finite = med.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    vmin, vmax = (float(finite.min()), float(finite.max())) if finite.size else (0.0, 1.0)

    # Landscape: A (left) | B (right). B stacks original → reverse → density once.
    fig = plt.figure(figsize=(14.2, 7.4))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.12, 1.0], wspace=0.28,
        left=0.055, right=0.985, top=0.88, bottom=0.08,
    )
    left = gs[0].subgridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.38)
    ax_h = fig.add_subplot(left[0])
    ax_l = fig.add_subplot(left[1])
    im = _draw_joint_heatmap(
        ax_h, med, cnt, T_LABS, K_LABS,
        r"Median $|\Delta\mathrm{logit}|$ by postoperative time and $k$", vmin, vmax,
    )
    cax = make_axes_locatable(ax_h).append_axes("right", size="3.8%", pad=0.06)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.set_label(r"Median $|\Delta\mathrm{logit}|$", fontsize=8.0)
    _draw_joint_lines(
        ax_l, rnd, T_LABS, K_LABS,
        r"Median $|\Delta\mathrm{logit}|$ vs $k$ (95% bootstrap CI)", seed_offset=200,
    )

    right = gs[1].subgridspec(
        3, 1, height_ratios=[1.15, 1.15, 0.55], hspace=0.20,
    )
    ax_o = fig.add_subplot(right[0])
    ax_r = fig.add_subplot(right[1], sharex=ax_o, sharey=ax_o)
    ax_d = fig.add_subplot(right[2], sharex=ax_o)
    aki_h = case.get("aki_hours")
    if aki_h is None and case.get("person_id") is not None:
        try:
            aki_h = _aki_hours_postop(_load_episode(str(case["person_id"])))
        except Exception:
            aki_h = None
    _draw_case_stack(
        ax_o, ax_r, ax_d,
        case["path_orig"], case["path_rev"], float(case["abs_delta_logit_reverse"]),
        highlight_h0=case.get("highlight_h0", 0.0),
        highlight_h1=case.get("highlight_h1", 6.0),
        n_events=case.get("n_events"),
        aki_hours=float(aki_h) if aki_h is not None else None,
    )

    fig.canvas.draw()
    pos_a = ax_h.get_position()
    pos_b = ax_o.get_position()
    fig.text(
        pos_a.x0 - 0.01, pos_a.y1 + 0.035,
        fig_style.panel_label("A", "Order sensitivity by postoperative time and event density"),
        va="bottom", **fig_style.panel_kw(fig),
    )
    fig.text(
        pos_b.x0 - 0.01, pos_b.y1 + 0.035,
        fig_style.panel_label(
            "B",
            "Illustrative reverse within the first 0–6 h window",
        ),
        va="bottom", **fig_style.panel_kw(fig),
    )
    fig_style.add_suptitle(fig, "Within-window event-order sensitivity", y=0.975)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def update_report(
    meta_c: Dict[str, Any],
    time_stats: Dict[str, Any] | None = None,
    joint_stats: Dict[str, Any] | None = None,
) -> None:
    report = OUT / "REPORT.md"
    summary = json.loads((OUT / "summary.json").read_text())
    es_sum = summary["event_stream"]
    ts_sum = summary["timeseries"]
    mode = "full" if int(summary.get("n_episodes", 0)) > 500 else "smoke"

    lines = [
        "# Within-bin order sensitivity report",
        "",
        f"**Mode:** `{mode}`  ",
        f"**Episodes scanned:** {summary['n_episodes']}  ",
        f"**Eligible windows:** {summary['n_windows']}  ",
        "",
        "## Method (brief)",
        "",
        "Keep event set / values / 6h bin fixed; reassign event identities onto the "
        "original chronological timestamp slots "
        r"$\mathbf{x}^{(\pi)}=\{(e_{\pi(i)},t_i)\}$ "
        "(stream stays time-sorted). Compare end-of-bin online logit "
        r"$\Delta z = z(x)-z(x^{(\pi)})$. "
        "Fixed-window time-series bag features are order-invariant, so "
        r"$\Delta z_{\mathrm{TS}}\equiv 0$.",
        "",
        "## Main figures",
        "",
        "### Main (paper) — A joint $k$×time + B illustrative reverse case",
        "",
        "![Main](order_sensitivity_main_A_joint_B_case.png)",
        "",
        "**A (primary):** random within-bin order sensitivity by postoperative time × "
        "event density $k$. **B (secondary, de-emphasized):** extreme reverse-order "
        "path for one dense early window.",
        "",
        "### A — Event-stream vs time-series",
        "",
        "![A](order_sensitivity_A_es_vs_ts.png)",
        "",
        "ECDF of $|\\Delta z|$: event-stream mass shifts away from 0; time-series stays at 0.",
        "",
        f"- ES median $|\\Delta z|$ = **{es_sum['median']:.3f}**, P90 = **{es_sum['p90']:.3f}**, "
        f"frac $>0.05$ = **{es_sum['frac_gt_0_05']:.1%}**",
        f"- TS max $|\\Delta z|$ = **{ts_sum.get('max_abs_dz', 0):.1e}**, "
        f"max feature L2 = **{ts_sum.get('max_feat_l2', 0):.1e}**",
        "",
        "### B — Sensitivity grows with bin complexity",
        "",
        "![B](order_sensitivity_B_vs_k.png)",
        "",
        "Windows with more events (larger $k$) show larger mean $|\\Delta z_{\\mathrm{ES}}|$ "
        "across permutations.",
        "",
        "### C — Case path: original vs reversed order",
        "",
        "![C](order_sensitivity_C_case_path.png)",
        "",
        f"Person `{meta_c['person_id']}`, postoperative 6 h bin "
        f"**{meta_c['h0']:.0f}–{meta_c['h1']:.0f} h** "
        f"(`[{meta_c['h0']:.0f},{meta_c['h1']:.0f})`, bin_j={meta_c.get('bin_j', '?')}), "
        f"$k={meta_c['k']}$ events ({meta_c['n_distinct']} distinct). "
        f"End-of-bin risk {meta_c['p_end_orig']:.2f} → {meta_c['p_end_rev']:.2f} under reverse "
        f"($|\\Delta z|={meta_c['abs_delta_logit_reverse']:.2f}$).",
        "",
        "### D — Sensitivity by time since surgery",
        "",
        "![D](order_sensitivity_D_vs_time.png)",
        "",
        "Order sensitivity appears across 0–7 d; earliest hours are mildly higher but also denser.",
        "",
    ]
    if time_stats:
        lines.append("| Stratum | N windows | Median $|\\Delta z|$ | Median $k$ |")
        lines.append("|--|--:|--:|--:|")
        for lab, st in time_stats.items():
            lines.append(
                f"| {lab} | {st['n']} | {st['median_abs_dz']:.3f} | {st['median_k']:.0f} |"
            )
        lines.append("")

    lines.extend(
        [
            "### E — Joint: time since surgery × bin complexity (random)",
            "",
            "![E](order_sensitivity_E_joint_k_time.png)",
            "",
            "Primary (time × $k$) cross-tab using "
            r"$\operatorname{Mean}_{\pi}|\Delta z|$ "
            "over random chronology-preserving permutations per window "
            f"($R={joint_stats.get('n_random') if joint_stats else '?'}$). "
            "Right panel: cell medians with **95% bootstrap CI**. "
            "$n$ is the number of eligible windows in each cell.",
            "",
        ]
    )
    if joint_stats:
        rnd_c = (joint_stats.get("random") or {}).get("within_time_corr_k") or joint_stats.get(
            "within_time_corr_k"
        )
        if rnd_c:
            lines.append("Within-stratum correlation $\\mathrm{corr}(k, |\\Delta z|)$ (random):")
            lines.append("")
            lines.append("| Time stratum | corr($k$, $|\\Delta z|$) |")
            lines.append("|--|--:|")
            for lab, c in rnd_c.items():
                lines.append(f"| {lab} | {c:.3f} |")
            lines.append("")

    lines.extend(
        [
            "## Interpretation",
            "",
            "Event-stream modeling is **sensitive to within-window temporal order**; "
            "matched fixed-window 6h bag representations are **structurally invariant** to the "
            "same perturbation. Jointly, **bin complexity ($k$)** is the stronger organizing "
            "factor; **time since surgery** modulates where dense bins occur (especially 0–6 h) "
            "but order sensitivity is not confined to a single postoperative window.",
            "",
            "## Files",
            "",
            "- `order_sensitivity_main_A_joint_B_case.png` (**paper main**: A=E, B=C)",
            "- `order_sensitivity_A_es_vs_ts.png`",
            "- `order_sensitivity_B_vs_k.png`",
            "- `order_sensitivity_C_case_path.png`",
            "- `order_sensitivity_D_vs_time.png`",
            "- `order_sensitivity_E_joint_k_time.png`",
            "- `order_sensitivity_es_hist.png` (SI-style histogram)",
            "- `order_sensitivity_es_rows.csv`, `order_sensitivity_ts_rows.csv`",
            "",
        ]
    )
    report.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip_c", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    es = pd.read_csv(OUT / "order_sensitivity_es_rows.csv")
    ts = pd.read_csv(OUT / "order_sensitivity_ts_rows.csv")

    fig_a_es_vs_ts(es, ts, OUT / "order_sensitivity_A_es_vs_ts.png")
    print("wrote A")
    fig_b_vs_k(es, OUT / "order_sensitivity_B_vs_k.png")
    print("wrote B")

    time_stats = fig_d_vs_postop_time(es, OUT / "order_sensitivity_D_vs_time.png")
    print("wrote D", time_stats)
    (OUT / "time_stratum_stats.json").write_text(json.dumps(time_stats, indent=2))

    joint_stats = fig_e_joint_k_time(es, OUT / "order_sensitivity_E_joint_k_time.png")
    joint_out = {
        "n_random": joint_stats.get("n_random"),
        "within_time_corr_k": joint_stats["within_time_corr_k"],
        "random": {"within_time_corr_k": joint_stats["random"]["within_time_corr_k"]},
    }
    (OUT / "joint_k_time_stats.json").write_text(json.dumps(joint_out, indent=2))
    print("wrote E", joint_out)

    meta_path = OUT / "case_path_meta.json"
    if args.skip_c:
        if meta_path.exists():
            meta_c = json.loads(meta_path.read_text())
            print("skip_c: reusing existing C meta")
        else:
            meta_c = {
                "person_id": "?", "h0": 0, "h1": 6, "k": 0, "n_distinct": 0,
                "abs_delta_logit_reverse": 0, "p_end_orig": 0, "p_end_rev": 0,
            }
    else:
        meta_c = fig_c_case_path(es, OUT / "order_sensitivity_C_case_path.png", device_str=args.device)
        print("wrote C", meta_c)
        meta_path.write_text(json.dumps(meta_c, indent=2))

    main_out = fig_main_A_joint_B_case(meta_c=meta_c)
    print("wrote main A+B", main_out)

    update_report(meta_c, time_stats=time_stats, joint_stats=joint_out)
    print("updated REPORT.md")


if __name__ == "__main__":
    main()
