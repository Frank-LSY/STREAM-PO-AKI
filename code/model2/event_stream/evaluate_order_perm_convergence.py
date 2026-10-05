#!/usr/bin/env python3
"""Convergence analysis for #random within-bin order permutations per window.

Samples eligible windows and evaluates nested random sets R ∈ {3,5,10,20,50}
under chronology-preserving identity permutation:

    x^{(π)} = {(e_{π(i)}, t_i)}

Compares overall median |Δz|, time×k cell medians, and Spearman ρ vs R=50.
Supplementary Methods: choose formal R once estimates stabilize.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from event_stream.evaluate_within_bin_order_sensitivity import (
    ES_CKPT,
    HORIZON_H,
    ROOT,
    TEST_JSONL,
    _permute_event_list,
    end_of_bin_logit,
    find_candidate_bins,
)

OUT_DIR = ROOT / "result/figures/order_sensitivity/supp_perm_convergence"
R_GRID = (3, 5, 10, 20, 50)


def _sample_windows(
    episodes: Sequence[Dict[str, Any]],
    *,
    n_windows: int,
    max_per_ep: int,
    min_k: int,
    min_distinct: int,
    rng: np.random.Generator,
) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    pool: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for ep in episodes:
        cands = find_candidate_bins(ep, min_k=min_k, min_distinct=min_distinct)
        if not cands:
            continue
        if len(cands) > max_per_ep:
            pick = rng.choice(len(cands), size=max_per_ep, replace=False)
            cands = [cands[i] for i in pick]
        for c in cands:
            pool.append((ep, c))
    if len(pool) > n_windows:
        idx = rng.choice(len(pool), size=n_windows, replace=False)
        pool = [pool[i] for i in idx]
    return pool


def _nested_random_orders(k: int, r_max: int, rng: np.random.Generator) -> List[List[int]]:
    """Distinct non-identity random permutations; reverse excluded (random-only study)."""
    seen = {tuple(range(k))}
    out: List[List[int]] = []
    tries = 0
    while len(out) < r_max and tries < r_max * 40:
        tries += 1
        perm = list(rng.permutation(k))
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)
        out.append(perm)
    return out


def _window_mean_abs(vals: Sequence[float], r: int) -> float:
    use = list(vals)[:r]
    return float(np.mean(np.abs(use))) if use else float("nan")


def _add_bins(df: pd.DataFrame) -> pd.DataFrame:
    t_edges = [0, 6, 24, 72, 168]
    t_labs = ["0–6 h", "6–24 h", "24–72 h", "72 h–7 d"]
    k_edges = [3, 5, 9, 17, 33, 10_000]
    k_labs = ["3–4", "5–8", "9–16", "17–32", "≥33"]
    out = df.copy()
    out["tbin"] = pd.cut(out["h0"], bins=t_edges, right=False, labels=t_labs)
    out["kbin"] = pd.cut(out["k"], bins=k_edges, right=False, labels=k_labs)
    return out


def _cell_medians(df: pd.DataFrame, col: str) -> pd.Series:
    return (
        df.pivot_table(index="tbin", columns="kbin", values=col, aggfunc="median", observed=True)
        .stack(dropna=False)
    )


def plot_convergence(summary: Dict[str, Any], out: Path) -> None:
    rs = summary["R_grid"]
    med = [summary["by_R"][str(r)]["median_window_mean_abs_dz"] for r in rs]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))

    ax = axes[0]
    ax.plot(rs, med, color="#1f6cb0", marker="o", lw=2.0)
    ax.set_xlabel("Random permutations per window ($R$)")
    ax.set_ylabel(r"Overall median of window-mean $|\Delta z|$")
    ax.set_title("Convergence of overall sensitivity", fontsize=10.5, fontweight="bold", loc="left")
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    ax = axes[1]
    rhos = [summary["by_R"][str(r)].get("spearman_vs_R50") for r in rs]
    ax.plot(rs, rhos, color="#c0392b", marker="o", lw=2.0)
    ax.axhline(0.98, color="0.45", ls=(0, (3, 2)), lw=1.0)
    ax.set_xlabel("Random permutations per window ($R$)")
    ax.set_ylabel(r"Spearman $\rho$ vs $R=50$ (window means)")
    ax.set_title("Agreement with $R=50$ reference", fontsize=10.5, fontweight="bold", loc="left")
    ax.set_ylim(0.9, 1.005)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: Dict[str, Any], out: Path) -> None:
    lines = [
        "# Random-permutation convergence (supplementary)",
        "",
        f"**Windows sampled:** {summary['n_windows']}  ",
        f"**Max $R$:** {summary['R_max']}  ",
        "Permutation: chronology-preserving "
        r"$\mathbf{x}^{(\pi)}=\{(e_{\pi(i)},t_i)\}$ (random only; reverse excluded).",
        "",
        "## Results",
        "",
        "| $R$ | Median window-mean $|\\Delta z|$ | Mean | Spearman vs $R=50$ | Cell-median max rel. change vs $R=50$ |",
        "|--|--:|--:|--:|--:|",
    ]
    for r in summary["R_grid"]:
        st = summary["by_R"][str(r)]
        rho = st.get("spearman_vs_R50")
        rel = st.get("cell_median_max_rel_change_vs_R50")
        rho_s = f"{rho:.4f}" if rho is not None else "—"
        rel_s = f"{rel:.3%}" if rel is not None else "—"
        lines.append(
            f"| {r} | {st['median_window_mean_abs_dz']:.4f} | "
            f"{st['mean_window_mean_abs_dz']:.4f} | {rho_s} | {rel_s} |"
        )
    lines.extend(
        [
            "",
            "![convergence](perm_convergence.png)",
            "",
            "## Recommendation",
            "",
            summary.get("recommendation", ""),
            "",
        ]
    )
    out.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n_windows", type=int, default=1500)
    p.add_argument("--max_windows_per_episode", type=int, default=2)
    p.add_argument("--min_k", type=int, default=3)
    p.add_argument("--min_distinct", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    import sys

    sys.path.insert(0, str(ROOT / "code/model2"))
    from common.med_exposure import infer_num_med_exposure
    from common.utils import resolve_torch_device
    from event_stream.dataset import load_episode_jsonl
    from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
    from event_stream.event_attribution import episode_to_batch
    from scipy.stats import spearmanr

    device = resolve_torch_device(args.device)
    print(f"device={device}", flush=True)
    episodes = load_episode_jsonl(TEST_JSONL)
    print(f"episodes={len(episodes)}", flush=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(str(ES_CKPT))
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    windows = _sample_windows(
        episodes,
        n_windows=args.n_windows,
        max_per_ep=args.max_windows_per_episode,
        min_k=args.min_k,
        min_distinct=args.min_distinct,
        rng=rng,
    )
    print(f"sampled_windows={len(windows)}", flush=True)

    r_max = max(R_GRID)
    rows: List[Dict[str, Any]] = []
    for wi, (ep, cand) in enumerate(windows):
        h0, h1 = float(cand["h0"]), float(cand["h1"])
        idxs = list(cand["indices"])
        k = len(idxs)
        batch0, seq0, _ = episode_to_batch(ep, code_vocab, lab_stats, num_med, device)
        z0 = end_of_bin_logit(model, batch0, seq0, h0, h1)
        if z0 is None:
            continue
        orders = _nested_random_orders(k, r_max, rng)
        if len(orders) < r_max:
            # rare for large k; pad by resampling with replacement of distinct set
            while len(orders) < r_max:
                orders.append(list(rng.permutation(k)))
        abs_dz: List[float] = []
        for order in orders:
            ep_p = copy.deepcopy(ep)
            ep_p["events"] = _permute_event_list(ep["events"], idxs, order)
            batch_p, seq_p, _ = episode_to_batch(ep_p, code_vocab, lab_stats, num_med, device)
            z_p = end_of_bin_logit(model, batch_p, seq_p, h0, h1)
            if z_p is None:
                abs_dz.append(float("nan"))
            else:
                abs_dz.append(abs(float(z0 - z_p)))
        rec: Dict[str, Any] = {
            "person_id": str(ep.get("person_id")),
            "bin_j": int(cand["bin_j"]),
            "h0": h0,
            "h1": h1,
            "k": k,
            "n_distinct": int(cand["n_distinct"]),
            "n_perms_ok": int(np.sum(np.isfinite(abs_dz))),
        }
        for r in R_GRID:
            rec[f"mean_abs_dz_R{r}"] = _window_mean_abs(abs_dz, r)
        rows.append(rec)
        if (wi + 1) % 50 == 0:
            print(f"  {wi+1}/{len(windows)}", flush=True)

    df = pd.DataFrame(rows)
    csv_path = out_dir / "perm_convergence_windows.csv"
    df.to_csv(csv_path, index=False)

    dfb = _add_bins(df)
    ref = dfb[f"mean_abs_dz_R{r_max}"].to_numpy(dtype=float)
    cell_ref = _cell_medians(dfb, f"mean_abs_dz_R{r_max}")

    by_r: Dict[str, Any] = {}
    for r in R_GRID:
        col = f"mean_abs_dz_R{r}"
        x = dfb[col].to_numpy(dtype=float)
        st: Dict[str, Any] = {
            "median_window_mean_abs_dz": float(np.nanmedian(x)),
            "mean_window_mean_abs_dz": float(np.nanmean(x)),
        }
        if r == r_max:
            st["spearman_vs_R50"] = 1.0
            st["cell_median_max_rel_change_vs_R50"] = 0.0
        else:
            m = np.isfinite(x) & np.isfinite(ref)
            st["spearman_vs_R50"] = float(spearmanr(x[m], ref[m]).correlation) if m.sum() > 5 else None
            cell = _cell_medians(dfb, col)
            rels = []
            for key, v_ref in cell_ref.items():
                v = cell.get(key, np.nan)
                if np.isfinite(v) and np.isfinite(v_ref) and abs(v_ref) > 1e-6:
                    rels.append(abs(v - v_ref) / abs(v_ref))
            st["cell_median_max_rel_change_vs_R50"] = float(max(rels)) if rels else None
        by_r[str(r)] = st

    # Pick smallest R with rho>=0.98 and cell max rel change < 2% if available
    chosen = r_max
    for r in R_GRID:
        st = by_r[str(r)]
        rho = st.get("spearman_vs_R50")
        rel = st.get("cell_median_max_rel_change_vs_R50")
        if rho is not None and rho >= 0.98 and rel is not None and rel < 0.02:
            chosen = r
            break
    recommendation = (
        f"On {len(df)} windows, estimates stabilize by **R={chosen}** "
        f"(Spearman vs R=50 ≥ 0.98 and time×k cell-median relative change < 2% when met; "
        f"otherwise default to the smallest R meeting thresholds or R=20). "
        f"Recommended formal metric: window-mean over **R={chosen}** random chronology-preserving "
        f"permutations (plus reverse only as an illustrative / supplementary probe)."
    )

    summary = {
        "n_windows": int(len(df)),
        "R_grid": list(R_GRID),
        "R_max": r_max,
        "by_R": by_r,
        "recommended_R": chosen,
        "recommendation": recommendation,
        "paths": {"windows_csv": str(csv_path)},
    }
    (out_dir / "perm_convergence_summary.json").write_text(json.dumps(summary, indent=2))
    plot_convergence(summary, out_dir / "perm_convergence.png")
    write_report(summary, out_dir / "REPORT.md")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
