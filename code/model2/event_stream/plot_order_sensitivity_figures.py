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
    import sys
    sys.path.insert(0, str(ROOT / "code/model2"))
    from event_stream.dataset import load_episode_jsonl

    for ep in load_episode_jsonl(TEST_JSONL):
        if str(ep.get("person_id")) == str(pid):
            return ep
    raise KeyError(pid)


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
    from event_stream.event_attribution import episode_to_batch, forward_logits, logit_to_prob

    batch, seq, _ = episode_to_batch(
        episode, code_vocab, lab_stats, num_med, device, age_stats=age_stats
    )
    z_base, base_idx, postop_idx, z_postop = forward_logits(model, batch)
    ts, zs, ps, codes = [], [], [], []
    for pos, z in zip(postop_idx, z_postop):
        th = float(seq[pos].get("time_hour", -1))
        if h0 <= th < h1:
            ts.append(th)
            zs.append(float(z))
            ps.append(logit_to_prob(z))
            codes.append(str(seq[pos].get("raw_code", "")))
    return {
        "t": ts, "z": zs, "p": ps, "codes": codes,
        "z_base": float(z_base), "p_base": logit_to_prob(z_base),
    }


def _draw_case_path(ax, path: Dict[str, Any], title: str, color: str) -> None:
    n = len(path["p"])
    xs = np.arange(n)
    ax.plot(xs, path["p"], color=color, lw=1.6, marker="o", markersize=2.6, zorder=3)
    ax.axhline(0.5, color="0.5", ls=(0, (3, 2)), lw=0.9, alpha=0.85)
    ax.set_xlabel("Position within 6-h window (arrival order)", fontsize=8.6)
    ax.set_title(title, loc="left", pad=4, **fig_style.subtitle_kw(ax.figure))
    ax.tick_params(labelsize=7.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.22)
    if n:
        ax.annotate(
            f"{path['p'][0]:.2f}", xy=(0, path["p"][0]),
            xytext=(5, 0), textcoords="offset points", fontsize=7.0, color="0.35",
        )
        ax.annotate(
            f"{path['p'][-1]:.2f}", xy=(n - 1, path["p"][-1]),
            xytext=(5, 0), textcoords="offset points", fontsize=7.0, color="0.35",
            fontweight="bold",
        )


def _draw_case_pair(ax0, ax1, path0, path1, dz: float) -> None:
    _draw_case_path(ax0, path0, "Original order", BLUE)
    _draw_case_path(
        ax1, path1, rf"Reversed order  ·  $|\Delta\mathrm{{logit}}|$ at window end = {dz:.2f}", RED,
    )
    ax0.set_ylabel("Predicted PO-AKI probability", fontsize=8.6)
    ax1.tick_params(labelleft=False)


def fig_c_case_path(es: pd.DataFrame, out: Path, device_str: str = "auto") -> Dict[str, Any]:
    """Panel C: original vs reverse within-bin risk path for a high-|Δz| window."""
    import sys
    sys.path.insert(0, str(ROOT / "code/model2"))
    from common.med_exposure import infer_num_med_exposure
    from common.utils import resolve_torch_device
    from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle

    g = (
        es.groupby(["person_id", "bin_j", "h0", "h1", "k", "n_distinct"], as_index=False)["abs_delta_logit"]
        .mean()
        .sort_values("abs_delta_logit", ascending=False)
    )
    top = g.iloc[0]
    pid = str(int(top.person_id)) if float(top.person_id).is_integer() else str(top.person_id)
    h0, h1 = float(top.h0), float(top.h1)
    # Prefer reverse for interpretability
    rev_row = es[(es.person_id == top.person_id) & (es.bin_j == top.bin_j) & (es.perm == "reverse")]
    if rev_row.empty:
        rev_row = es[(es.person_id == top.person_id) & (es.bin_j == top.bin_j)].iloc[[0]]

    episode = _load_episode(pid)
    idxs = _bin_indices(episode, h0, h1)
    assert len(idxs) == int(top.k), (len(idxs), top.k)

    device = resolve_torch_device(device_str)
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(str(ES_CKPT))
    if num_med < 0:
        num_med = infer_num_med_exposure([episode])
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    path0 = _path_in_bin(model, episode, code_vocab, lab_stats, num_med, device, h0, h1, age_stats)
    ep_rev = copy.deepcopy(episode)
    order = list(range(len(idxs)))[::-1]
    ep_rev["events"] = _permute_event_list(episode["events"], idxs, order)
    path1 = _path_in_bin(model, ep_rev, code_vocab, lab_stats, num_med, device, h0, h1, age_stats)

    dz = float(rev_row.iloc[0]["abs_delta_logit"])
    CASE_PATH_DATA.write_text(json.dumps({"path_orig": path0, "path_rev": path1, "abs_delta_logit_reverse": dz}))

    fig, axes = plt.subplots(1, 2, figsize=(11.4, 2.65), sharey=True)
    _draw_case_pair(axes[0], axes[1], path0, path1, dz)

    fig.tight_layout(pad=0.35)
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
        "p_end_orig": float(path0["p"][-1]) if path0["p"] else None,
        "p_end_rev": float(path1["p"][-1]) if path1["p"] else None,
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

    fig = plt.figure(figsize=(12.0, 8.6))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 0.62], hspace=0.62,
        left=0.075, right=0.985, top=0.855, bottom=0.075,
    )
    top = gs[0].subgridspec(1, 2, width_ratios=[1.18, 1.05], wspace=0.32)
    ax_h = fig.add_subplot(top[0])
    ax_l = fig.add_subplot(top[1])
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

    bot = gs[1].subgridspec(1, 2, wspace=0.06)
    ax0 = fig.add_subplot(bot[0])
    ax1 = fig.add_subplot(bot[1], sharey=ax0)
    _draw_case_pair(
        ax0, ax1, case["path_orig"], case["path_rev"], float(case["abs_delta_logit_reverse"]),
    )

    fig.canvas.draw()
    x0 = 0.012
    ya = ax_l.get_position().y1 + 0.052
    yb = ax0.get_position().y1 + 0.058
    fig.text(x0, ya, fig_style.panel_label("A", "Order sensitivity by postoperative time and event density"),
             va="bottom", **fig_style.panel_kw(fig))
    fig.text(x0, yb, fig_style.panel_label("B", "Illustrative reversed-order perturbation"),
             va="bottom", **fig_style.panel_kw(fig))
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
