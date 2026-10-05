#!/usr/bin/env python3
"""Analyze isolated time_mlp geometry: e(t) = time_mlp([day/30, hour/24]).

Runs on a dense postoperative time grid and/or test-set event times. Produces:
  - distance vs |Δt| (L2 and 1 - cosine)
  - same-Δt difference-vector consistency
  - PCA / optional UMAP colored by continuous postoperative hours

Continuous postoperative time uses cumulative hours since surgery (``process_data``):
``time_hour`` = hours from anchor; ``time_day`` = days from anchor (= time_hour / 24).
Do **not** use ``time_day * 24 + time_hour`` (that doubles when both fields are cumulative).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from event_stream.dataset import load_episode_jsonl
from event_stream.model import PerioperativeBlockMaskModel
from common.utils import resolve_torch_device, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Isolate time_mlp embedding geometry on test/grid times.")
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to best_model.pt (uses time_mlp weights only).",
    )
    p.add_argument(
        "--episode_path",
        type=str,
        default=None,
        help="Optional test JSONL; postoperative event (time_day, time_hour) are merged into the time set.",
    )
    p.add_argument("--output_dir", type=str, required=True, help="Directory for figures and metrics JSON.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--grid_postop_hour_max",
        type=float,
        default=720.0,
        help="Dense grid: postoperative continuous hours in [0, grid_postop_hour_max] (step below).",
    )
    p.add_argument(
        "--grid_step_hours",
        type=float,
        default=6.0,
        help="Step for dense postoperative hour grid.",
    )
    p.add_argument(
        "--pair_sample_size",
        type=int,
        default=50000,
        help="Max random pairs for distance-vs-dt scatter (from grid points).",
    )
    p.add_argument(
        "--delta_hours_list",
        type=str,
        default="6,12,24,48,72",
        help="Comma-separated Δt (hours) for same-delta difference analysis.",
    )
    p.add_argument(
        "--max_time_points",
        type=int,
        default=100000,
        help="Cap unique time points after merging grid + episode times (grid always kept; episode times subsampled).",
    )
    p.add_argument(
        "--same_delta_max_starts",
        type=int,
        default=10000,
        help="Max (t, t+Δ) starts per Δ for same-delta analysis.",
    )
    p.add_argument(
        "--same_delta_pair_sample",
        type=int,
        default=20000,
        help="Max random vector pairs per within-/between-Δ cosine estimate (no N×N matrix).",
    )
    p.add_argument(
        "--manifold_max_points",
        type=int,
        default=5000,
        help="Subsample time points for PCA/UMAP figures when N is larger.",
    )
    p.add_argument(
        "--pca_max_diff_vectors",
        type=int,
        default=10000,
        help="Subsample e(t+Δ)−e(t) vectors for same-delta PCA figure.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--no_compare_untrained",
        action="store_true",
        help="Skip random-init time_mlp baseline and comparison figures.",
    )
    return p.parse_args()


def load_model(checkpoint: str, device: torch.device) -> PerioperativeBlockMaskModel:
    ckpt = torch.load(checkpoint, map_location="cpu")
    train_args = ckpt.get("args", {})
    code_vocab = ckpt.get("code_vocab") or {}
    model = PerioperativeBlockMaskModel(
        vocab_size=len(code_vocab),
        d_model=int(train_args.get("d_model", 128)),
        nhead=int(train_args.get("nhead", 4)),
        num_layers=int(train_args.get("num_layers", 2)),
        dim_feedforward=int(train_args.get("ffn_dim", 256)),
        dropout=float(train_args.get("dropout", 0.1)),
        num_med_exposure=int(train_args.get("num_med_exposure", 0)),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model


def load_model_untrained_time_mlp(
    checkpoint: str, device: torch.device, seed: int
) -> PerioperativeBlockMaskModel:
    """Same architecture as checkpoint; ``time_mlp`` re-initialized (not task-trained)."""
    torch.manual_seed(int(seed))
    model = load_model(checkpoint, device)
    for module in model.time_mlp.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    model.eval()
    return model


def postop_hours_from_event(ev: Dict[str, Any]) -> Optional[float]:
    """Cumulative postoperative hours for one event (matches ``process_data`` JSONL)."""
    day = float(ev.get("time_day", 0.0))
    if day <= 0.0:
        return None
    if ev.get("time_hour") is not None:
        return float(ev["time_hour"])
    return day * 24.0


def postop_hours_to_model_fields(t_hours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map cumulative postop hours → (time_day, time_hour) as in training JSONL / ``time_mlp``."""
    t = np.asarray(t_hours, dtype=np.float64)
    return t / 24.0, t.copy()


def time_mlp_embed(model: PerioperativeBlockMaskModel, time_days: np.ndarray, time_hours: np.ndarray, device: torch.device) -> np.ndarray:
    """Return [N, d_model] numpy embeddings from time_mlp only (hour/24)."""
    del time_days
    hour_t = torch.tensor(time_hours, dtype=torch.float32, device=device)
    feat = (hour_t / 24.0).unsqueeze(-1)
    with torch.no_grad():
        emb = model.time_mlp(feat)
    return emb.cpu().numpy()


def collect_postop_times_from_episodes(
    episodes: List[Dict[str, Any]],
    max_hours: Optional[float] = None,
) -> np.ndarray:
    """Postoperative events: cumulative ``time_hour`` (time_day > 0, non-procedure)."""
    hours_list: List[float] = []
    for ep in episodes:
        for ev in ep.get("events", []):
            mod = str(ev.get("modality", "")).lower()
            if mod == "procedure":
                continue
            th = postop_hours_from_event(ev)
            if th is None:
                continue
            if max_hours is not None and th > max_hours + 1e-9:
                continue
            hours_list.append(th)
    if not hours_list:
        return np.array([], dtype=np.float64)
    return np.unique(np.asarray(hours_list, dtype=np.float64))


def build_time_grid(max_hours: float, step_hours: float) -> np.ndarray:
    if step_hours <= 0 or max_hours < 0:
        raise ValueError("grid_step_hours must be > 0 and grid_postop_hour_max >= 0.")
    return np.arange(0.0, max_hours + 1e-9, step_hours, dtype=np.float64)


def merge_and_cap_time_points(
    grid_hours: np.ndarray,
    ep_hours: np.ndarray,
    max_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Merge episode times into grid; if over cap, keep all grid points and subsample extras."""
    grid_hours = np.unique(np.asarray(grid_hours, dtype=np.float64))
    info: Dict[str, Any] = {"n_grid": int(len(grid_hours)), "n_episode_unique": int(len(ep_hours))}
    if ep_hours.size == 0:
        info["n_merged"] = info["n_grid"]
        info["capped"] = False
        return grid_hours, info

    merged = np.unique(np.concatenate([grid_hours, np.asarray(ep_hours, dtype=np.float64)]))
    info["n_merged_before_cap"] = int(len(merged))
    if max_points <= 0 or len(merged) <= max_points:
        info["n_merged"] = int(len(merged))
        info["capped"] = False
        return merged, info

    # Keep every grid point; fill remaining budget with random episode-only times.
    extra = merged[~np.isin(merged, grid_hours)]
    n_budget = max(len(grid_hours), max_points)
    n_extra = max(0, n_budget - len(grid_hours))
    if n_extra == 0:
        out = grid_hours
    else:
        if len(extra) > n_extra:
            pick = rng.choice(len(extra), size=n_extra, replace=False)
            extra = extra[pick]
        out = np.unique(np.concatenate([grid_hours, extra]))
    out.sort()
    info["n_merged"] = int(len(out))
    info["capped"] = True
    info["max_time_points"] = max_points
    return out, info


def _cosine_sim_random_pairs(
    ci: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[float]:
    n = len(ci)
    if n < 2 or max_pairs <= 0:
        return []
    n_pairs = min(max_pairs, n * (n - 1) // 2)
    ii = rng.integers(0, n, size=n_pairs)
    jj = rng.integers(0, n, size=n_pairs)
    mask = ii != jj
    ii, jj = ii[mask], jj[mask]
    if ii.size == 0:
        return []
    return (ci[ii] * ci[jj]).sum(axis=1).tolist()


def _cosine_sim_random_cross(
    ci: np.ndarray,
    cj: np.ndarray,
    max_pairs: int,
    rng: np.random.Generator,
) -> List[float]:
    if len(ci) == 0 or len(cj) == 0 or max_pairs <= 0:
        return []
    n_pairs = min(max_pairs, len(ci) * len(cj))
    ii = rng.integers(0, len(ci), size=n_pairs)
    jj = rng.integers(0, len(cj), size=n_pairs)
    return (ci[ii] * cj[jj]).sum(axis=1).tolist()


def _spearman(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    try:
        from scipy.stats import spearmanr

        r, _ = spearmanr(x, y)
        return float(r) if np.isfinite(r) else None
    except Exception:
        return None


def distance_vs_delta_metrics(
    t_hours: np.ndarray,
    emb: np.ndarray,
    pair_sample_size: int,
    rng: np.random.Generator,
    n_bins: int = 40,
) -> Dict[str, Any]:
    n = len(t_hours)
    if n < 2:
        return {"error": "need at least 2 time points"}

    # All pairs is O(n^2); subsample for scatter, bin all pairs if n small else subsample pairs
    max_pairs = n * (n - 1) // 2
    n_pairs = min(pair_sample_size, max_pairs)
    idx_i = rng.integers(0, n, size=n_pairs)
    idx_j = rng.integers(0, n, size=n_pairs)
    mask = idx_i != idx_j
    idx_i, idx_j = idx_i[mask], idx_j[mask]
    if idx_i.size == 0:
        return {"error": "no valid pairs"}

    dt = np.abs(t_hours[idx_i] - t_hours[idx_j])
    diff = emb[idx_i] - emb[idx_j]
    l2 = np.linalg.norm(diff, axis=1)
    # cosine distance
    norm_i = np.linalg.norm(emb[idx_i], axis=1) + 1e-12
    norm_j = np.linalg.norm(emb[idx_j], axis=1) + 1e-12
    cos_sim = np.sum(emb[idx_i] * emb[idx_j], axis=1) / (norm_i * norm_j)
    cos_dist = 1.0 - np.clip(cos_sim, -1.0, 1.0)

    spearman_l2 = _spearman(dt, l2)
    spearman_cos = _spearman(dt, cos_dist)

    # Binned means for monotonicity check
    edges = np.linspace(dt.min(), dt.max(), n_bins + 1)
    bin_idx = np.digitize(dt, edges[1:-1], right=False)
    bin_centers, mean_l2, mean_cos, counts = [], [], [], []
    for b in range(n_bins):
        m = bin_idx == b
        if m.sum() < 5:
            continue
        bin_centers.append(float(0.5 * (edges[b] + edges[b + 1])))
        mean_l2.append(float(l2[m].mean()))
        mean_cos.append(float(cos_dist[m].mean()))
        counts.append(int(m.sum()))

    return {
        "n_points": int(n),
        "n_pairs_sampled": int(len(dt)),
        "spearman_dt_l2": spearman_l2,
        "spearman_dt_cosine_dist": spearman_cos,
        "binned": {
            "bin_center_hours": bin_centers,
            "mean_l2": mean_l2,
            "mean_cosine_dist": mean_cos,
            "count": counts,
        },
        "scatter_dt": dt.tolist() if len(dt) <= 20000 else None,
        "scatter_l2": l2.tolist() if len(dt) <= 20000 else None,
        "scatter_cos": cos_dist.tolist() if len(dt) <= 20000 else None,
    }


def same_delta_consistency(
    model: PerioperativeBlockMaskModel,
    device: torch.device,
    t_hours: np.ndarray,
    delta_list: List[float],
    t_min: float,
    t_max: float,
    max_starts_per_delta: int,
    pair_sample_size: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """For each Δ, collect e(t+Δ)-e(t) over valid t; summarize variance explained by Δ."""
    all_delta: List[float] = []
    all_vecs: List[np.ndarray] = []
    per_delta: Dict[str, Any] = {}

    for d in delta_list:
        starts = t_hours[(t_hours >= t_min) & (t_hours + d <= t_max)]
        if starts.size == 0:
            per_delta[str(d)] = {"n": 0}
            continue
        if max_starts_per_delta > 0 and len(starts) > max_starts_per_delta:
            pick = rng.choice(len(starts), size=max_starts_per_delta, replace=False)
            starts = starts[pick]
        t0 = starts
        t1 = starts + d
        d0, h0 = postop_hours_to_model_fields(t0)
        d1, h1 = postop_hours_to_model_fields(t1)
        e0 = time_mlp_embed(model, d0, h0, device)
        e1 = time_mlp_embed(model, d1, h1, device)
        diff = e1 - e0
        per_delta[str(d)] = {
            "n": int(len(starts)),
            "n_starts_capped": bool(max_starts_per_delta > 0 and len(starts) >= max_starts_per_delta),
            "diff_norm_mean": float(np.linalg.norm(diff, axis=1).mean()),
            "diff_norm_std": float(np.linalg.norm(diff, axis=1).std()),
        }
        all_delta.extend([d] * len(starts))
        all_vecs.append(diff)

    if not all_vecs:
        return {"per_delta": per_delta, "error": "no valid (t, t+Δ) pairs"}

    vecs = np.concatenate(all_vecs, axis=0)
    labels = np.asarray(all_delta, dtype=np.float64)

    # Between-group vs within-group cosine similarity of difference vectors
    uniq = sorted(set(delta_list))
    within_sims, between_sims = [], []
    for i, di in enumerate(uniq):
        vi = vecs[labels == di]
        if len(vi) < 2:
            continue
        ni = np.linalg.norm(vi, axis=1, keepdims=True) + 1e-12
        ci = vi / ni
        within_sims.extend(_cosine_sim_random_pairs(ci, pair_sample_size, rng))
        for dj in uniq[i + 1 :]:
            vj = vecs[labels == dj]
            if len(vj) == 0:
                continue
            nj = np.linalg.norm(vj, axis=1, keepdims=True) + 1e-12
            cj = vj / nj
            between_sims.extend(_cosine_sim_random_cross(ci, cj, pair_sample_size, rng))

    return {
        "per_delta": per_delta,
        "mean_within_delta_cosine_sim": float(np.mean(within_sims)) if within_sims else None,
        "mean_between_delta_cosine_sim": float(np.mean(between_sims)) if between_sims else None,
        "difference_vectors": vecs,
        "delta_labels": labels,
    }


def plot_distance_vs_delta(
    metrics: Dict[str, Any],
    pair_sample: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    out_path: Path,
    title_suffix: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    binned = metrics.get("binned", {})
    if binned.get("bin_center_hours"):
        x = binned["bin_center_hours"]
        axes[0].plot(x, binned["mean_l2"], "o-", label="mean L2")
        axes[1].plot(x, binned["mean_cosine_dist"], "o-", color="C1", label="mean 1-cos")
    if pair_sample is not None:
        dt, l2, cos_d = pair_sample
        n = min(8000, len(dt))
        idx = np.linspace(0, len(dt) - 1, n, dtype=int)
        axes[0].scatter(dt[idx], l2[idx], alpha=0.15, s=4, c="C0", label="pairs (subsample)")
        axes[1].scatter(dt[idx], cos_d[idx], alpha=0.15, s=4, c="C1")
    for ax, ylab, sp in zip(
        axes,
        ["L2 distance", "Cosine distance"],
        [metrics.get("spearman_dt_l2"), metrics.get("spearman_dt_cosine_dist")],
    ):
        ax.set_xlabel("|Δt| (postoperative hours)")
        ax.set_ylabel(ylab)
        if sp is not None:
            ax.set_title(f"Spearman ρ = {sp:.3f}")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    title = "time_mlp: embedding distance vs time separation"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_distance_vs_delta_comparison(
    trained: Dict[str, Any],
    untrained: Dict[str, Any],
    out_path: Path,
    dpi: int = 150,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    specs = [
        ("mean_l2", "L2 distance", "spearman_dt_l2"),
        ("mean_cosine_dist", "Cosine distance (1 − cos)", "spearman_dt_cosine_dist"),
    ]
    for ax, (key, ylab, sp_key) in zip(axes, specs):
        for metrics, label, color in (
            (trained, "trained", "C0"),
            (untrained, "untrained (random init)", "C3"),
        ):
            binned = metrics.get("binned", {})
            if not binned.get("bin_center_hours"):
                continue
            ax.plot(
                binned["bin_center_hours"],
                binned[key],
                "o-",
                color=color,
                label=label,
                markersize=4,
            )
            sp = metrics.get(sp_key)
            if sp is not None:
                ax.plot([], [], " ", label=f"{label} Spearman ρ={sp:.3f}")
        ax.set_xlabel("|Δt| (postoperative hours)")
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("time_mlp: trained vs untrained — distance vs |Δt|", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def pca_pc1_variance_ratio(emb: np.ndarray, max_points: int, rng: np.random.Generator) -> float:
    from sklearn.decomposition import PCA

    emb = np.asarray(emb)
    if max_points > 0 and len(emb) > max_points:
        idx = rng.choice(len(emb), size=max_points, replace=False)
        emb = emb[idx]
    if len(emb) < 2:
        return float("nan")
    pca = PCA(n_components=2, random_state=0)
    pca.fit(emb)
    return float(pca.explained_variance_ratio_[0])


def plot_same_delta_pca(
    vecs: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    title_suffix: str = "",
) -> None:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(vecs)
    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = np.unique(labels)
    for d in uniq:
        m = labels == d
        ax.scatter(xy[m, 0], xy[m, 1], s=12, alpha=0.5, label=f"Δ={d:g}h (n={m.sum()})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    title = "PCA of e(t+Δ) − e(t) by Δt"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_time_manifold_comparison(
    t_hours: np.ndarray,
    emb_trained: np.ndarray,
    emb_untrained: np.ndarray,
    out_path: Path,
    dpi: int,
    max_points: int,
    rng: np.random.Generator,
) -> None:
    from sklearn.decomposition import PCA

    t_hours = np.asarray(t_hours)
    emb_trained = np.asarray(emb_trained)
    emb_untrained = np.asarray(emb_untrained)
    if max_points > 0 and len(t_hours) > max_points:
        idx = rng.choice(len(t_hours), size=max_points, replace=False)
        idx.sort()
        t_hours = t_hours[idx]
        emb_trained = emb_trained[idx]
        emb_untrained = emb_untrained[idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, emb, label in zip(
        axes,
        (emb_trained, emb_untrained),
        ("trained", "untrained (random init)"),
    ):
        pca = PCA(n_components=2, random_state=0)
        xy = pca.fit_transform(emb)
        order = np.argsort(t_hours)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=t_hours, cmap="viridis", s=16, alpha=0.85)
        ax.plot(xy[order, 0], xy[order, 1], "k-", alpha=0.25, linewidth=0.8)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
    fig.subplots_adjust(right=0.88)
    fig.colorbar(sc, ax=axes.tolist(), label="postoperative hours")
    fig.suptitle("PCA of e(t): trained vs untrained time_mlp", fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.86, 0.96])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def plot_time_manifold(
    t_hours: np.ndarray,
    emb: np.ndarray,
    out_dir: Path,
    dpi: int,
    max_points: int = 5000,
    rng: Optional[np.random.Generator] = None,
    title_suffix: str = "",
) -> float:
    from sklearn.decomposition import PCA

    t_hours = np.asarray(t_hours)
    emb = np.asarray(emb)
    if max_points > 0 and len(t_hours) > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(t_hours), size=max_points, replace=False)
        idx.sort()
        t_hours = t_hours[idx]
        emb = emb[idx]

    pca = PCA(n_components=2, random_state=0)
    xy = pca.fit_transform(emb)
    order = np.argsort(t_hours)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=t_hours, cmap="viridis", s=20, alpha=0.85)
    plt.colorbar(sc, ax=ax, label="postoperative hours")
    ax.plot(xy[order, 0], xy[order, 1], "k-", alpha=0.25, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    title = "PCA of e(t) on time grid (line = increasing t)"
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "time_mlp_pca_grid.png", dpi=dpi)
    plt.close(fig)

    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(t_hours) - 1))
        xy_u = reducer.fit_transform(emb)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc = ax.scatter(xy_u[:, 0], xy_u[:, 1], c=t_hours, cmap="viridis", s=20, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="postoperative hours")
        ax.plot(xy_u[order, 0], xy_u[order, 1], "k-", alpha=0.25, linewidth=0.8)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        um_title = "UMAP of e(t) on time grid"
        if title_suffix:
            um_title = f"{um_title} ({title_suffix})"
        ax.set_title(um_title)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "time_mlp_umap_grid.png", dpi=dpi)
        plt.close(fig)
    except ImportError:
        pass
    return float(pca.explained_variance_ratio_[0])


def _pair_sample_for_plot(
    grid_hours: np.ndarray,
    emb: np.ndarray,
    dist_metrics: Dict[str, Any],
    pair_sample_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dist_metrics.get("scatter_dt") is not None:
        return (
            np.asarray(dist_metrics["scatter_dt"]),
            np.asarray(dist_metrics["scatter_l2"]),
            np.asarray(dist_metrics["scatter_cos"]),
        )
    n = len(grid_hours)
    ni = rng.integers(0, n, size=min(8000, pair_sample_size))
    nj = rng.integers(0, n, size=min(8000, pair_sample_size))
    m = ni != nj
    dt = np.abs(grid_hours[ni[m]] - grid_hours[nj[m]])
    diff = emb[ni[m]] - emb[nj[m]]
    l2 = np.linalg.norm(diff, axis=1)
    cos_d = 1.0 - np.clip(
        np.sum(emb[ni[m]] * emb[nj[m]], axis=1)
        / (np.linalg.norm(emb[ni[m]], axis=1) * np.linalg.norm(emb[nj[m]], axis=1) + 1e-12),
        -1,
        1,
    )
    return dt, l2, cos_d


def run_geometry_variant(
    model: PerioperativeBlockMaskModel,
    variant: str,
    grid_hours: np.ndarray,
    day: np.ndarray,
    hour: np.ndarray,
    out_dir: Path,
    args: argparse.Namespace,
    device: torch.device,
    delta_list: List[float],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Run full geometry pipeline for one time_mlp (trained or untrained)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = time_mlp_embed(model, day, hour, device)

    dist_metrics = distance_vs_delta_metrics(grid_hours, emb, args.pair_sample_size, rng)
    pair_for_plot = _pair_sample_for_plot(grid_hours, emb, dist_metrics, args.pair_sample_size, rng)
    plot_distance_vs_delta(
        dist_metrics,
        pair_for_plot,
        out_dir / "time_mlp_distance_vs_delta.png",
        title_suffix=variant,
    )

    t_min, t_max = float(grid_hours.min()), float(grid_hours.max())
    delta_metrics = same_delta_consistency(
        model,
        device,
        grid_hours,
        delta_list,
        t_min,
        t_max,
        args.same_delta_max_starts,
        args.same_delta_pair_sample,
        rng,
    )
    if "difference_vectors" in delta_metrics:
        vecs_pca = delta_metrics["difference_vectors"]
        labels_pca = delta_metrics["delta_labels"]
        if args.pca_max_diff_vectors > 0 and len(vecs_pca) > args.pca_max_diff_vectors:
            pick = rng.choice(len(vecs_pca), size=args.pca_max_diff_vectors, replace=False)
            vecs_pca = vecs_pca[pick]
            labels_pca = labels_pca[pick]
        plot_same_delta_pca(
            vecs_pca, labels_pca, out_dir / "time_mlp_same_delta_pca.png", title_suffix=variant
        )
        delta_metrics_save = {
            k: v for k, v in delta_metrics.items() if k not in ("difference_vectors", "delta_labels")
        }
    else:
        delta_metrics_save = delta_metrics

    pc1_ratio = plot_time_manifold(
        grid_hours,
        emb,
        out_dir,
        args.dpi,
        args.manifold_max_points,
        rng,
        title_suffix=variant,
    )

    np.savez(
        out_dir / "time_mlp_grid.npz",
        t_hours=grid_hours,
        time_days=day,
        time_hours=hour,
        embeddings=emb,
        variant=variant,
    )

    summary: Dict[str, Any] = {
        "variant": variant,
        "n_grid_points": int(len(grid_hours)),
        "t_hours_min": t_min,
        "t_hours_max": t_max,
        "pca_pc1_variance_ratio": pc1_ratio,
        "distance_vs_delta": {k: v for k, v in dist_metrics.items() if not str(k).startswith("scatter_")},
        "same_delta_consistency": delta_metrics_save,
    }
    save_json(summary, out_dir / "time_mlp_geometry_metrics.json")
    return summary


def build_comparison_summary(
    trained: Dict[str, Any],
    untrained: Dict[str, Any],
    checkpoint: str,
    episode_path: Optional[str],
    merge_info: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    def _get(d: Dict[str, Any], *keys: str) -> Optional[float]:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        return float(cur) if cur is not None else None

    t_l2 = _get(trained, "distance_vs_delta", "spearman_dt_l2")
    u_l2 = _get(untrained, "distance_vs_delta", "spearman_dt_l2")
    t_cos = _get(trained, "distance_vs_delta", "spearman_dt_cosine_dist")
    u_cos = _get(untrained, "distance_vs_delta", "spearman_dt_cosine_dist")
    t_pc1 = trained.get("pca_pc1_variance_ratio")
    u_pc1 = untrained.get("pca_pc1_variance_ratio")

    per_delta_cmp: Dict[str, Any] = {}
    t_pd = (trained.get("same_delta_consistency") or {}).get("per_delta") or {}
    u_pd = (untrained.get("same_delta_consistency") or {}).get("per_delta") or {}
    for key in sorted(set(t_pd.keys()) | set(u_pd.keys()), key=lambda x: float(x)):
        t_norm = (t_pd.get(key) or {}).get("diff_norm_mean")
        u_norm = (u_pd.get(key) or {}).get("diff_norm_mean")
        per_delta_cmp[key] = {
            "trained_diff_norm_mean": t_norm,
            "untrained_diff_norm_mean": u_norm,
            "delta_trained_minus_untrained": (
                float(t_norm) - float(u_norm) if t_norm is not None and u_norm is not None else None
            ),
        }

    return {
        "checkpoint": str(Path(checkpoint).resolve()),
        "episode_path": episode_path,
        "time_convention": (
            "t_hours = cumulative time_hour since surgery (0..720 for 30d); "
            "time_day = t_hours/24; model input [time_day/30, time_hour/24]"
        ),
        "grid_postop_hour_max": args.grid_postop_hour_max,
        "untrained_seed": args.seed,
        "time_point_merge": merge_info,
        "trained": trained,
        "untrained": untrained,
        "comparison": {
            "spearman_dt_l2": {"trained": t_l2, "untrained": u_l2, "delta": (t_l2 - u_l2) if t_l2 is not None and u_l2 is not None else None},
            "spearman_dt_cosine_dist": {
                "trained": t_cos,
                "untrained": u_cos,
                "delta": (t_cos - u_cos) if t_cos is not None and u_cos is not None else None,
            },
            "pca_pc1_variance_ratio": {
                "trained": t_pc1,
                "untrained": u_pc1,
                "delta": (t_pc1 - u_pc1) if t_pc1 is not None and u_pc1 is not None else None,
            },
            "mean_within_delta_cosine_sim": {
                "trained": _get(trained, "same_delta_consistency", "mean_within_delta_cosine_sim"),
                "untrained": _get(untrained, "same_delta_consistency", "mean_within_delta_cosine_sim"),
            },
            "per_delta_diff_norm_mean": per_delta_cmp,
        },
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    device = resolve_torch_device(args.device)

    delta_list = [float(x.strip()) for x in args.delta_hours_list.split(",") if x.strip()]

    grid_only = build_time_grid(args.grid_postop_hour_max, args.grid_step_hours)
    ep_hours = np.array([], dtype=np.float64)
    if args.episode_path:
        ep_hours = collect_postop_times_from_episodes(
            load_episode_jsonl(args.episode_path),
            max_hours=args.grid_postop_hour_max,
        )
    grid_hours, merge_info = merge_and_cap_time_points(
        grid_only, ep_hours, args.max_time_points, rng
    )
    if merge_info.get("capped"):
        print(
            f"Capped time points: {merge_info['n_merged_before_cap']} -> {merge_info['n_merged']} "
            f"(grid={merge_info['n_grid']}, episode unique={merge_info['n_episode_unique']})"
        )

    day, hour = postop_hours_to_model_fields(grid_hours)

    model_trained = load_model(args.checkpoint, device)
    trained_summary = run_geometry_variant(
        model_trained,
        "trained",
        grid_hours,
        day,
        hour,
        out_dir,
        args,
        device,
        delta_list,
        rng,
    )
    print(f"[trained] Wrote {out_dir / 'time_mlp_geometry_metrics.json'}")

    if not args.no_compare_untrained:
        untrained_dir = out_dir / "untrained"
        model_untrained = load_model_untrained_time_mlp(args.checkpoint, device, args.seed)
        untrained_summary = run_geometry_variant(
            model_untrained,
            "untrained",
            grid_hours,
            day,
            hour,
            untrained_dir,
            args,
            device,
            delta_list,
            rng,
        )
        print(f"[untrained] Wrote {untrained_dir / 'time_mlp_geometry_metrics.json'}")

        emb_trained = time_mlp_embed(model_trained, day, hour, device)
        emb_untrained = time_mlp_embed(model_untrained, day, hour, device)
        plot_distance_vs_delta_comparison(
            trained_summary["distance_vs_delta"],
            untrained_summary["distance_vs_delta"],
            out_dir / "time_mlp_distance_vs_delta_comparison.png",
            dpi=args.dpi,
        )
        plot_time_manifold_comparison(
            grid_hours,
            emb_trained,
            emb_untrained,
            out_dir / "time_mlp_pca_grid_comparison.png",
            args.dpi,
            args.manifold_max_points,
            rng,
        )

        comparison = build_comparison_summary(
            trained_summary,
            untrained_summary,
            args.checkpoint,
            args.episode_path,
            merge_info,
            args,
        )
        comparison["grid_step_hours"] = args.grid_step_hours
        save_json(comparison, out_dir / "time_mlp_geometry_comparison.json")
        print(f"[compare] Wrote {out_dir / 'time_mlp_geometry_comparison.json'}")
    else:
        # Back-compat single-file summary at root (trained only, enriched).
        single = {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "episode_path": args.episode_path,
            "grid_step_hours": args.grid_step_hours,
            "time_convention": (
                "t_hours = cumulative time_hour since surgery (0..720 for 30d); "
                "time_day = t_hours/24; model input [time_day/30, time_hour/24]"
            ),
            "grid_postop_hour_max": args.grid_postop_hour_max,
            "time_point_merge": merge_info,
            **trained_summary,
        }
        save_json(single, out_dir / "time_mlp_geometry_metrics.json")


if __name__ == "__main__":
    main()
