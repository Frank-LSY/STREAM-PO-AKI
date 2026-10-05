#!/usr/bin/env python3
"""Landmark dynamic calibration for postoperative PO-AKI risk.

For landmarks L ∈ {0, 6, 12, 24, 48, 168} h:

  Risk set:  R_L = {i : T_i^{AKI} > L}
  Outcome:   Y_{i,L} = 1(L < T_i^{AKI} ≤ 30 d)
  Prediction p̂_i(L) = most recent available risk by L
                      (BASE at L=0; last postop event ≤ L thereafter;
                       else BASE carry-forward)

Metrics: Brier, Brier Skill Score vs prevalence null, logistic calibration
intercept / slope, equal-frequency reliability bins (optional bootstrap CI).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.clinical_metrics import EpisodeTrajectory, prob_at_or_before

DEFAULT_LANDMARKS_H = (0.0, 6.0, 12.0, 24.0, 48.0, 168.0)
DEFAULT_HORIZON_H = 30.0 * 24.0  # postoperative day 30


@dataclass
class LandmarkRow:
    person_id: str
    landmark_h: float
    y: int
    p: float
    t_aki_hours: Optional[float]
    used_base: int
    fold: int = 0


def resolve_t_aki_hours(traj: EpisodeTrajectory) -> Optional[float]:
    """Return onset hours after surgery, +inf for never-AKI, None if positive but untimed."""
    if int(traj.label_aki) == 0:
        return float("inf")
    if traj.t_aki_hours is None or not np.isfinite(float(traj.t_aki_hours)):
        return None
    return float(traj.t_aki_hours)


def pred_at_landmark(traj: EpisodeTrajectory, landmark_h: float) -> Tuple[float, bool]:
    """Carry-forward risk available by landmark ``landmark_h``.

    Returns (probability, used_base).
    """
    L = float(landmark_h)
    if L <= 0.0 + 1e-12:
        return float(traj.baseline_prob), True
    p = prob_at_or_before(traj.postop_hours, traj.postop_probs, L)
    if p is None:
        return float(traj.baseline_prob), True
    return float(p), False


def landmark_rows_for_trajectory(
    traj: EpisodeTrajectory,
    landmarks_h: Sequence[float] = DEFAULT_LANDMARKS_H,
    horizon_h: float = DEFAULT_HORIZON_H,
    fold: int = 0,
) -> Tuple[List[LandmarkRow], Optional[str]]:
    """Build landmark rows for one episode. Returns (rows, skip_reason)."""
    t_aki = resolve_t_aki_hours(traj)
    if t_aki is None:
        return [], "aki_positive_missing_onset_time"
    rows: List[LandmarkRow] = []
    for L in landmarks_h:
        Lf = float(L)
        if not (t_aki > Lf):
            continue  # not in risk set
        y = int(Lf < t_aki <= float(horizon_h))
        p, used_base = pred_at_landmark(traj, Lf)
        if not np.isfinite(p):
            continue
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        rows.append(
            LandmarkRow(
                person_id=str(traj.person_id),
                landmark_h=Lf,
                y=y,
                p=p,
                t_aki_hours=None if not np.isfinite(t_aki) else float(t_aki),
                used_base=int(used_base),
                fold=int(fold),
            )
        )
    return rows, None


def build_landmark_table(
    trajectories: Sequence[EpisodeTrajectory],
    landmarks_h: Sequence[float] = DEFAULT_LANDMARKS_H,
    horizon_h: float = DEFAULT_HORIZON_H,
    fold: int = 0,
) -> Tuple[List[LandmarkRow], Dict[str, Any]]:
    rows: List[LandmarkRow] = []
    n_skip = 0
    skip_reasons: Dict[str, int] = {}
    for tr in trajectories:
        part, reason = landmark_rows_for_trajectory(
            tr, landmarks_h=landmarks_h, horizon_h=horizon_h, fold=fold
        )
        if reason is not None:
            n_skip += 1
            skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            continue
        rows.extend(part)
    meta = {
        "n_trajectories": int(len(trajectories)),
        "n_skipped_episodes": int(n_skip),
        "skip_reasons": skip_reasons,
        "n_rows": int(len(rows)),
        "landmarks_h": [float(x) for x in landmarks_h],
        "horizon_h": float(horizon_h),
    }
    return rows, meta


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p) - np.log(1.0 - p)


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Logistic calibration: logit Pr(Y=1) = a + b logit(p). Ideal a=0, b=1."""
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size < 8 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    x = _logit(p).reshape(-1, 1)
    try:
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        lr.fit(x, y)
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])
    except Exception:
        coef = np.polyfit(x.ravel(), y.astype(float), 1)
        return float(coef[1]), float(coef[0])


def brier_and_skill(y: np.ndarray, p: np.ndarray) -> Tuple[float, float, float]:
    """Return (Brier, BS_null, BSS)."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if y.size == 0:
        return float("nan"), float("nan"), float("nan")
    bs = float(np.mean((p - y) ** 2))
    prev = float(np.mean(y))
    bs_null = float(prev * (1.0 - prev))
    if bs_null <= 1e-12:
        bss = float("nan")
    else:
        bss = float(1.0 - bs / bs_null)
    return bs, bs_null, bss


def equal_frequency_bins(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Equal-frequency (quantile) reliability bins: (mean p̂, mean y) per bin."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = int(y.size)
    if n == 0:
        return {
            "n_bins": 0,
            "bin_pred": [],
            "bin_obs": [],
            "bin_n": [],
            "bin_lo": [],
            "bin_hi": [],
        }
    n_bins = int(max(1, min(n_bins, n)))
    order = np.argsort(p, kind="mergesort")
    p_s, y_s = p[order], y[order]
    edges = np.linspace(0, n, n_bins + 1, dtype=int)
    bin_pred, bin_obs, bin_n, bin_lo, bin_hi = [], [], [], [], []
    for b in range(n_bins):
        lo, hi = int(edges[b]), int(edges[b + 1])
        if hi <= lo:
            continue
        pb = p_s[lo:hi]
        yb = y_s[lo:hi]
        bin_pred.append(float(np.mean(pb)))
        bin_obs.append(float(np.mean(yb)))
        bin_n.append(int(len(pb)))
        bin_lo.append(float(pb.min()))
        bin_hi.append(float(pb.max()))
    return {
        "n_bins": int(len(bin_pred)),
        "bin_pred": bin_pred,
        "bin_obs": bin_obs,
        "bin_n": bin_n,
        "bin_lo": bin_lo,
        "bin_hi": bin_hi,
    }


def bootstrap_equal_frequency_bins(
    y: np.ndarray,
    p: np.ndarray,
    n_bins: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Patient-level bootstrap CIs for equal-frequency calibration points."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    point = equal_frequency_bins(y, p, n_bins=n_bins)
    k = int(point["n_bins"])
    if k == 0 or n_boot <= 0 or y.size < 20:
        point["bin_obs_lo"] = [float("nan")] * k
        point["bin_obs_hi"] = [float("nan")] * k
        point["n_boot"] = 0
        return point

    rng = np.random.default_rng(seed)
    preds = np.full((n_boot, k), np.nan, dtype=float)
    obs = np.full((n_boot, k), np.nan, dtype=float)
    n = y.size
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        cur = equal_frequency_bins(y[idx], p[idx], n_bins=n_bins)
        kk = min(k, int(cur["n_bins"]))
        for j in range(kk):
            preds[b, j] = cur["bin_pred"][j]
            obs[b, j] = cur["bin_obs"][j]
    lo_q, hi_q = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    point["bin_pred_lo"] = [float(np.nanpercentile(preds[:, j], lo_q)) for j in range(k)]
    point["bin_pred_hi"] = [float(np.nanpercentile(preds[:, j], hi_q)) for j in range(k)]
    point["bin_obs_lo"] = [float(np.nanpercentile(obs[:, j], lo_q)) for j in range(k)]
    point["bin_obs_hi"] = [float(np.nanpercentile(obs[:, j], hi_q)) for j in range(k)]
    point["n_boot"] = int(n_boot)
    return point


def summarize_landmark(
    rows: Sequence[LandmarkRow],
    landmark_h: float,
    n_bins: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
    hist_bins: int = 20,
) -> Dict[str, Any]:
    sub = [r for r in rows if abs(float(r.landmark_h) - float(landmark_h)) < 1e-9]
    y = np.asarray([r.y for r in sub], dtype=float)
    p = np.asarray([r.p for r in sub], dtype=float)
    n = int(len(sub))
    n_pos = int(y.sum()) if n else 0
    prev = float(n_pos / n) if n else float("nan")
    bs, bs_null, bss = brier_and_skill(y, p)
    intercept, slope = calibration_intercept_slope(y, p)
    curve = bootstrap_equal_frequency_bins(
        y, p, n_bins=n_bins, n_boot=n_boot, seed=seed + int(landmark_h * 10)
    )
    n_base = int(sum(r.used_base for r in sub))
    edges = np.linspace(0.0, 1.0, int(hist_bins) + 1)
    if n:
        counts, _ = np.histogram(p, bins=edges)
        counts = counts.astype(int).tolist()
    else:
        counts = [0] * int(hist_bins)
    return {
        "landmark_h": float(landmark_h),
        "n_at_risk": n,
        "n_future_aki": n_pos,
        "future_aki_rate": prev,
        "frac_used_base": float(n_base / n) if n else float("nan"),
        "brier": bs,
        "brier_null": bs_null,
        "brier_skill": bss,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "curve": curve,
        "pred_hist": {
            "edges": edges.astype(float).tolist(),
            "counts": counts,
        },
    }


def summarize_all_landmarks(
    rows: Sequence[LandmarkRow],
    landmarks_h: Sequence[float] = DEFAULT_LANDMARKS_H,
    n_bins: int = 10,
    n_boot: int = 1000,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    return [
        summarize_landmark(rows, L, n_bins=n_bins, n_boot=n_boot, seed=seed)
        for L in landmarks_h
    ]


def aggregate_fold_summaries(
    fold_summaries: Sequence[Sequence[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Mean (min, max) across folds for scalar metrics and calibration curves.

    Calibration curve for plotting: solid = fold-mean reliability points;
    shade = fold min–max of observed proportions at each equal-frequency bin
    (with n=5 this is the natural fold-wise band; scalar metrics use the same
    mean(min,max) convention).
    """
    if not fold_summaries:
        return []
    n_folds = len(fold_summaries)
    n_L = len(fold_summaries[0])
    out: List[Dict[str, Any]] = []
    keys = (
        "brier",
        "brier_null",
        "brier_skill",
        "calibration_intercept",
        "calibration_slope",
        "future_aki_rate",
        "frac_used_base",
        "n_at_risk",
        "n_future_aki",
    )
    for j in range(n_L):
        block: Dict[str, Any] = {
            "landmark_h": float(fold_summaries[0][j]["landmark_h"]),
            "n_folds": int(n_folds),
            "curves_by_fold": [fold_summaries[f][j]["curve"] for f in range(n_folds)],
        }
        for k in keys:
            vals = np.asarray(
                [fold_summaries[f][j][k] for f in range(n_folds)], dtype=float
            )
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                block[k] = float("nan")
                block[f"{k}_min"] = float("nan")
                block[f"{k}_max"] = float("nan")
            else:
                block[k] = float(np.mean(finite))
                block[f"{k}_min"] = float(np.min(finite))
                block[f"{k}_max"] = float(np.max(finite))
        curves = block["curves_by_fold"]
        n_bins = min(int(c.get("n_bins", 0)) for c in curves) if curves else 0
        mean_pred, mean_obs, obs_lo, obs_hi = [], [], [], []
        for b in range(n_bins):
            ps = [
                float(c["bin_pred"][b])
                for c in curves
                if b < len(c.get("bin_pred", [])) and np.isfinite(c["bin_pred"][b])
            ]
            ys = [
                float(c["bin_obs"][b])
                for c in curves
                if b < len(c.get("bin_obs", [])) and np.isfinite(c["bin_obs"][b])
            ]
            mean_pred.append(float(np.mean(ps)) if ps else float("nan"))
            mean_obs.append(float(np.mean(ys)) if ys else float("nan"))
            obs_lo.append(float(np.min(ys)) if ys else float("nan"))
            obs_hi.append(float(np.max(ys)) if ys else float("nan"))
        block["curve_mean"] = {
            "n_bins": n_bins,
            "bin_pred": mean_pred,
            "bin_obs": mean_obs,
            "bin_obs_lo": obs_lo,
            "bin_obs_hi": obs_hi,
            "ci": "fold_minmax",
            "n_folds": int(n_folds),
        }
        # Prefer curve_mean as the primary curve for downstream plot/table code.
        block["curve"] = block["curve_mean"]

        # Mean predicted-probability histogram counts across folds (same edges).
        hists = [
            fold_summaries[f][j].get("pred_hist") or {}
            for f in range(n_folds)
        ]
        edges = None
        for h in hists:
            if h.get("edges"):
                edges = list(h["edges"])
                break
        if edges is not None and len(edges) >= 2:
            n_h = len(edges) - 1
            mat = []
            for h in hists:
                c = h.get("counts") or []
                if len(c) == n_h:
                    mat.append([float(x) for x in c])
            if mat:
                arr = np.asarray(mat, dtype=float)
                block["pred_hist"] = {
                    "edges": edges,
                    "counts": np.mean(arr, axis=0).tolist(),
                    "counts_lo": np.min(arr, axis=0).tolist(),
                    "counts_hi": np.max(arr, axis=0).tolist(),
                    "n_folds": int(n_folds),
                }
        out.append(block)
    return out


def resolve_calibration_curve(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the curve dict used for plotting (aggregated mean preferred)."""
    return summary.get("curve") or summary.get("curve_mean") or {}

def rows_to_records(rows: Sequence[LandmarkRow]) -> List[Dict[str, Any]]:
    return [
        {
            "person_id": r.person_id,
            "landmark_h": r.landmark_h,
            "y": r.y,
            "p": r.p,
            "t_aki_hours": r.t_aki_hours,
            "used_base": r.used_base,
            "fold": r.fold,
        }
        for r in rows
    ]


def summary_table_records(summaries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flat table rows for CSV / markdown."""
    out = []
    for s in summaries:
        out.append(
            {
                "landmark_h": s["landmark_h"],
                "n_at_risk": s.get("n_at_risk"),
                "n_future_aki": s.get("n_future_aki"),
                "future_aki_pct": (
                    100.0 * float(s["future_aki_rate"])
                    if s.get("future_aki_rate") is not None
                    and np.isfinite(s["future_aki_rate"])
                    else float("nan")
                ),
                "brier": s.get("brier"),
                "brier_min": s.get("brier_min", s.get("brier")),
                "brier_max": s.get("brier_max", s.get("brier")),
                "brier_skill": s.get("brier_skill"),
                "brier_skill_min": s.get("brier_skill_min", s.get("brier_skill")),
                "brier_skill_max": s.get("brier_skill_max", s.get("brier_skill")),
                "calibration_intercept": s.get("calibration_intercept"),
                "calibration_intercept_min": s.get(
                    "calibration_intercept_min", s.get("calibration_intercept")
                ),
                "calibration_intercept_max": s.get(
                    "calibration_intercept_max", s.get("calibration_intercept")
                ),
                "calibration_slope": s.get("calibration_slope"),
                "calibration_slope_min": s.get(
                    "calibration_slope_min", s.get("calibration_slope")
                ),
                "calibration_slope_max": s.get(
                    "calibration_slope_max", s.get("calibration_slope")
                ),
                "frac_used_base": s.get("frac_used_base"),
            }
        )
    return out
