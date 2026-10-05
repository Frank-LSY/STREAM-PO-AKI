"""Clinical evaluation metrics for postoperative online AKI risk (model2)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from common.utils import safe_auprc, safe_auroc


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    s = str(s).strip().replace("Z", "")
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def hours_after_anchor(t0: Optional[datetime], t: Optional[datetime]) -> Optional[float]:
    if t0 is None or t is None:
        return None
    return (t - t0).total_seconds() / 3600.0


@dataclass
class EpisodeTrajectory:
    person_id: str
    label_aki: int
    t_aki_hours: Optional[float]
    postop_hours: np.ndarray
    postop_probs: np.ndarray
    baseline_prob: float
    postop_patient_days: float

    @property
    def n_postop_steps(self) -> int:
        return int(len(self.postop_hours))


def trajectory_from_episode_and_forward(
    episode: Dict[str, Any],
    token_probs: np.ndarray,
    baseline_prob: float,
    sequence_meta: List[Dict[str, Any]],
) -> EpisodeTrajectory:
    """Build trajectory from one forward pass (sequence aligned with token_probs)."""
    t0 = _parse_dt(episode.get("t0"))
    aki_dt = _parse_dt(episode.get("aki_time"))
    t_aki = hours_after_anchor(t0, aki_dt) if int(episode.get("label_aki", 0)) == 1 else None

    hours_list: List[float] = []
    probs_list: List[float] = []
    for p, meta in zip(token_probs, sequence_meta):
        if int(meta.get("is_postop", 0)) != 1:
            continue
        th = float(meta.get("time_hour", 0.0))
        if th < 0:
            continue
        hours_list.append(th)
        probs_list.append(float(p))

    if hours_list:
        order = np.argsort(hours_list)
        postop_hours = np.asarray(hours_list, dtype=np.float64)[order]
        postop_probs = np.asarray(probs_list, dtype=np.float64)[order]
        max_h = float(postop_hours.max())
        postop_days = max(max_h / 24.0, 1.0 / 24.0)
    else:
        postop_hours = np.array([], dtype=np.float64)
        postop_probs = np.array([], dtype=np.float64)
        postop_days = 0.0

    return EpisodeTrajectory(
        person_id=str(episode.get("person_id", "")),
        label_aki=int(float(episode.get("label_aki", 0)) >= 0.5),
        t_aki_hours=t_aki,
        postop_hours=postop_hours,
        postop_probs=postop_probs,
        baseline_prob=float(baseline_prob),
        postop_patient_days=postop_days,
    )


def first_alert_time(hours: np.ndarray, probs: np.ndarray, tau: float) -> Optional[float]:
    if len(hours) == 0:
        return None
    mask = probs >= tau
    if not mask.any():
        return None
    return float(hours[mask][0])


def prob_at_or_before(hours: np.ndarray, probs: np.ndarray, t_target: float) -> Optional[float]:
    if len(hours) == 0 or not np.isfinite(t_target):
        return None
    m = hours <= t_target + 1e-9
    if not m.any():
        return None
    return float(probs[m][-1])


def early_warning_metrics(
    trajectories: Sequence[EpisodeTrajectory],
    tau: float,
    lead_thresholds_hours: Sequence[float] = (6.0, 12.0, 24.0),
) -> Dict[str, Any]:
    aki_eps = [t for t in trajectories if t.label_aki == 1]
    n_aki = len(aki_eps)
    alerted_before: List[float] = []
    lead_times: List[float] = []

    skipped_no_onset = 0
    skipped_no_postop = 0
    for tr in aki_eps:
        if tr.t_aki_hours is None or not np.isfinite(tr.t_aki_hours):
            skipped_no_onset += 1
            continue
        if tr.n_postop_steps == 0:
            skipped_no_postop += 1
            continue
        t_alert = first_alert_time(tr.postop_hours, tr.postop_probs, tau)
        if t_alert is not None and t_alert < tr.t_aki_hours:
            alerted_before.append(1.0)
            lead_times.append(tr.t_aki_hours - t_alert)
        else:
            alerted_before.append(0.0)

    n_eval = len(alerted_before)
    sensitivity = float(np.mean(alerted_before)) if n_eval > 0 else None
    lead_arr = np.asarray(lead_times, dtype=np.float64)
    out: Dict[str, Any] = {
        "alert_threshold_tau": float(tau),
        "n_aki_episodes": n_aki,
        "n_aki_evaluated": n_eval,
        "n_aki_skipped_no_onset_time": skipped_no_onset,
        "n_aki_skipped_no_postop_steps": skipped_no_postop,
        "alert_sensitivity_before_aki": sensitivity,
        "n_alerted_before_aki": int(lead_arr.size),
        "onset_time_definition": "aki_time from episode JSONL (last AKI condition time vs t0)",
        "alert_rule": "T_alert = min{time_hour : p_online >= tau} on postoperative event steps",
    }
    if lead_arr.size > 0:
        out["median_lead_time_hours"] = float(np.median(lead_arr))
        out["mean_lead_time_hours"] = float(np.mean(lead_arr))
        q25, q75 = np.percentile(lead_arr, [25, 75])
        out["lead_time_iqr_hours"] = [float(q25), float(q75)]
        for h in lead_thresholds_hours:
            key = f"proportion_lead_ge_{int(h)}h"
            out[key] = float(np.mean(lead_arr >= h))
    else:
        out["median_lead_time_hours"] = None
        out["mean_lead_time_hours"] = None
        out["lead_time_iqr_hours"] = None
        for h in lead_thresholds_hours:
            out[f"proportion_lead_ge_{int(h)}h"] = None
    return out


def false_alert_burden_metrics(
    trajectories: Sequence[EpisodeTrajectory],
    tau: float,
) -> Dict[str, Any]:
    neg = [t for t in trajectories if t.label_aki == 0]
    n_neg = len(neg)
    false_alert_episodes = 0
    false_alert_timesteps = 0
    total_patient_days = 0.0
    alerts_per_episode: List[int] = []

    for tr in trajectories:
        total_patient_days += tr.postop_patient_days
        n_alerts = int(np.sum(tr.postop_probs >= tau)) if tr.n_postop_steps > 0 else 0
        alerts_per_episode.append(n_alerts)
        if tr.label_aki == 0 and n_alerts > 0:
            false_alert_episodes += 1
            false_alert_timesteps += n_alerts

    all_alerts = int(sum(alerts_per_episode))
    return {
        "alert_threshold_tau": float(tau),
        "n_non_aki_episodes": n_neg,
        "false_alert_episode_rate": float(false_alert_episodes / n_neg) if n_neg > 0 else None,
        "n_non_aki_with_any_alert": false_alert_episodes,
        "false_alert_timesteps": false_alert_timesteps,
        "total_postop_patient_days": float(total_patient_days),
        "false_alerts_per_100_patient_days": (
            float(100.0 * false_alert_timesteps / total_patient_days) if total_patient_days > 0 else None
        ),
        "mean_alerts_per_episode_all": float(np.mean(alerts_per_episode)) if alerts_per_episode else None,
        "mean_alerts_per_episode_non_aki": (
            float(np.mean([a for tr, a in zip(trajectories, alerts_per_episode) if tr.label_aki == 0]))
            if n_neg > 0
            else None
        ),
        "total_alert_timesteps_all_episodes": all_alerts,
    }


def _episode_max_risk(tr: EpisodeTrajectory) -> float:
    if tr.n_postop_steps == 0:
        return float(tr.baseline_prob)
    return float(np.max(tr.postop_probs))


def alert_burden_metrics(
    trajectories: Sequence[EpisodeTrajectory],
    top_fractions: Sequence[float] = (0.05, 0.10),
    target_alerts_per_100_patient_days: Sequence[float] = (5.0, 10.0, 20.0),
) -> Dict[str, Any]:
    """Episode-level ranking by max postoperative online risk."""
    y = np.array([t.label_aki for t in trajectories], dtype=np.int64)
    scores = np.array([_episode_max_risk(t) for t in trajectories], dtype=np.float64)
    total_days = float(sum(t.postop_patient_days for t in trajectories))
    n = len(trajectories)
    order = np.argsort(-scores)

    by_top_frac: Dict[str, Any] = {}
    for frac in top_fractions:
        k = max(1, int(np.ceil(frac * n)))
        idx = order[:k]
        y_sel = y[idx]
        tp = int((y_sel == 1).sum())
        ppv = float(tp / k) if k > 0 else None
        sens = float(tp / max(int(y.sum()), 1)) if y.sum() > 0 else None
        by_top_frac[f"top_{int(frac * 100)}pct"] = {
            "k_episodes_flagged": k,
            "ppv": ppv,
            "sensitivity": sens,
            "n_true_aki_flagged": tp,
        }

    by_alert_budget: Dict[str, Any] = {}
    if total_days > 0 and n > 0:
        flat_probs = []
        flat_y_token = []
        for tr in trajectories:
            if tr.n_postop_steps == 0:
                continue
            flat_probs.extend(tr.postop_probs.tolist())
            flat_y_token.extend([tr.label_aki] * tr.n_postop_steps)
        flat_probs = np.asarray(flat_probs, dtype=np.float64)
        flat_y = np.asarray(flat_y_token, dtype=np.int64)

        for target in target_alerts_per_100_patient_days:
            target_count = target * total_days / 100.0
            if len(flat_probs) == 0:
                by_alert_budget[f"alerts_per_100pd_{target:g}"] = {"tau": None, "ppv": None, "sensitivity": None}
                continue
            sorted_probs = np.sort(flat_probs)[::-1]
            ki = min(len(sorted_probs) - 1, max(0, int(np.ceil(target_count)) - 1))
            tau_budget = float(sorted_probs[ki])
            flagged = flat_probs >= tau_budget
            tp = int(((flat_y == 1) & flagged).sum())
            fp = int(((flat_y == 0) & flagged).sum())
            ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else None
            aki_eps = [t for t in trajectories if t.label_aki == 1]
            sens_ep = _episode_sensitivity_at_tau(aki_eps, tau_budget)
            by_alert_budget[f"alerts_per_100pd_{target:g}"] = {
                "tau": tau_budget,
                "approx_flagged_timesteps": int(flagged.sum()),
                "ppv_token_level": ppv,
                "sensitivity_episode_level": sens_ep,
            }

    return {
        "ranking_score": "max postoperative online probability per episode",
        "by_top_episode_fraction": by_top_frac,
        "by_alert_budget_per_100_patient_days": by_alert_budget,
        "total_postop_patient_days": total_days,
    }


def _episode_sensitivity_at_tau(aki_trajectories: Sequence[EpisodeTrajectory], tau: float) -> Optional[float]:
    if not aki_trajectories:
        return None
    hits = 0
    for tr in aki_trajectories:
        if tr.n_postop_steps == 0:
            continue
        if tr.t_aki_hours is None:
            continue
        t_alert = first_alert_time(tr.postop_hours, tr.postop_probs, tau)
        if t_alert is not None and t_alert < tr.t_aki_hours:
            hits += 1
    return float(hits / len(aki_trajectories))


def baseline_vs_online_metrics(
    trajectories: Sequence[EpisodeTrajectory],
    horizon_hours_before_aki: float = 24.0,
) -> Dict[str, Any]:
    y = np.array([t.label_aki for t in trajectories], dtype=np.int64)
    p_base = np.array([t.baseline_prob for t in trajectories], dtype=np.float64)
    p_online_max = np.array([_episode_max_risk(t) for t in trajectories], dtype=np.float64)

  # NRI (categorical at 0.5) baseline vs online max
    nri = _net_reclassification_improvement(y, p_base, p_online_max, threshold=0.5)

    deltas_aki: List[float] = []
    deltas_non: List[float] = []
    for tr in trajectories:
        if tr.label_aki == 1 and tr.t_aki_hours is not None:
            t_ref = max(0.0, tr.t_aki_hours - horizon_hours_before_aki)
            p_pre = prob_at_or_before(tr.postop_hours, tr.postop_probs, t_ref)
            if p_pre is not None:
                deltas_aki.append(p_pre - tr.baseline_prob)
        elif tr.label_aki == 0:
            if tr.n_postop_steps > 0:
                deltas_non.append(float(tr.postop_probs[-1]) - tr.baseline_prob)
            else:
                deltas_non.append(0.0)

    dca = {
        "baseline_max_online": decision_curve_net_benefit(y, p_online_max),
        "baseline_only": decision_curve_net_benefit(y, p_base),
    }

    return {
        "discrimination": {
            "baseline_auroc": safe_auroc(y, p_base),
            "baseline_auprc": safe_auprc(y, p_base),
            "online_max_auroc": safe_auroc(y, p_online_max),
            "online_max_auprc": safe_auprc(y, p_online_max),
        },
        "nri_at_0.5": nri,
        "risk_escalation": {
            "horizon_hours_before_aki": horizon_hours_before_aki,
            "definition": "delta_p = p_online(at or before T_AKI - h) - p_baseline",
            "aki_median_delta_p": _median_or_none(deltas_aki),
            "non_aki_median_delta_p": _median_or_none(deltas_non),
            "n_aki_with_delta": len(deltas_aki),
            "n_non_aki_with_delta": len(deltas_non),
        },
        "decision_curve": dca,
    }


def _median_or_none(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _net_reclassification_improvement(
    y: np.ndarray,
    p_old: np.ndarray,
    p_new: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.int64)
    old_pos = p_old >= threshold
    new_pos = p_new >= threshold
    up = int(((old_pos == 0) & (new_pos == 1) & (y == 1)).sum())
    down = int(((old_pos == 1) & (new_pos == 0) & (y == 1)).sum())
    events = int((y == 1).sum())
    n = len(y)
    event_rate = events / n if n > 0 else 0.0
    nri = (up - down) / n if n > 0 else None
    return {
        "threshold": threshold,
        "nri": nri,
        "upclassified_events": up,
        "downclassified_events": down,
        "n_events": events,
        "n": n,
    }


def decision_curve_net_benefit(
    y: np.ndarray,
    p: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    n = len(y)
    if n == 0:
        return {"thresholds": [], "net_benefit": [], "treat_all": [], "treat_none": []}
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.50, 10)
    prev = float(y.mean())
    nb_list, ta_list, tn_list = [], [], []
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            continue
        pred = p >= pt
        tp = float(((pred == 1) & (y == 1)).sum()) / n
        fp = float(((pred == 1) & (y == 0)).sum()) / n
        nb = tp - fp * (pt / (1.0 - pt))
        nb_list.append(float(nb))
        ta_list.append(float(prev - pt / (1.0 - pt) * prev))
        tn_list.append(0.0)
    return {
        "thresholds": [float(x) for x in thresholds],
        "net_benefit": nb_list,
        "treat_all": ta_list,
        "treat_none": tn_list,
        "prevalence": float(prev),
    }


def _logit_prob(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_scaling(y_true: np.ndarray, y_prob: np.ndarray, max_iter: int = 200) -> float:
    """Fit scalar temperature T>0 on logits/T (binary NLL)."""
    import torch

    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_prob, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) < 2 or len(np.unique(y)) < 2:
        return 1.0
    y_t = torch.tensor(y, dtype=torch.float64)
    logits = torch.tensor(_logit_prob(p), dtype=torch.float64)
    log_T = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.5, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scaled = logits / torch.exp(log_T)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaled, y_t)
        loss.backward()
        return loss

    opt.step(closure)
    T = float(torch.exp(log_T).detach().cpu().item())
    if not np.isfinite(T) or T <= 0:
        return 1.0
    return T


def apply_temperature_scaling(y_prob: np.ndarray, temperature: float) -> np.ndarray:
    T = float(temperature)
    if not np.isfinite(T) or abs(T - 1.0) < 1e-12:
        return np.asarray(y_prob, dtype=np.float64)
    return _sigmoid(_logit_prob(y_prob) / T)


def _pool_online_from_trajectories(
    trajectories: Sequence[EpisodeTrajectory],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (hours, probs, labels) for all postoperative prediction steps."""
    hours: List[float] = []
    probs: List[float] = []
    labels: List[int] = []
    for tr in trajectories:
        for h, p in zip(tr.postop_hours, tr.postop_probs):
            hours.append(float(h))
            probs.append(float(p))
            labels.append(int(tr.label_aki))
    return (
        np.asarray(hours, dtype=np.float64),
        np.asarray(probs, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def _reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {"mean_predicted": [], "fraction_positive": [], "n_bins_used": 0}
    try:
        from sklearn.calibration import calibration_curve

        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")
        return {
            "mean_predicted": [float(x) for x in mean_pred],
            "fraction_positive": [float(x) for x in frac_pos],
            "n_bins_used": int(len(mean_pred)),
        }
    except Exception:
        return {"mean_predicted": [], "fraction_positive": [], "n_bins_used": 0}


def _calibration_block(y: np.ndarray, p: np.ndarray, n_bins: int) -> Dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    if len(y) == 0:
        return {
            "n": 0,
            "n_pos": 0,
            "brier": None,
            "ece": None,
            "calibration_slope": None,
            "calibration_intercept": None,
            "reliability_curve": _reliability_curve(y, p, n_bins),
        }
    brier = float(np.mean((p - y) ** 2))
    ece = _expected_calibration_error(y, p, n_bins)
    slope, intercept = _calibration_slope_intercept(y, p)
    return {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "brier": brier,
        "ece": ece,
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "reliability_curve": _reliability_curve(y, p, n_bins),
    }


def _mask_online_window(hours: np.ndarray, h_min: float, h_max: Optional[float]) -> np.ndarray:
    m = hours >= float(h_min) - 1e-9
    if h_max is not None:
        m &= hours <= float(h_max) + 1e-9
    return m


def calibration_metrics(
    trajectories: Sequence[EpisodeTrajectory],
    n_bins: int = 10,
    early_postop_hour_max: float = 48.0,
    postop_hour_bin_width: float = 6.0,
    min_bucket_n: int = 200,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    base_probs = np.array([t.baseline_prob for t in trajectories], dtype=np.float64)
    base_y = np.array([t.label_aki for t in trajectories], dtype=np.int64)
    base_probs_cal = apply_temperature_scaling(base_probs, temperature)

    hours, online_p, online_y = _pool_online_from_trajectories(trajectories)
    online_p_cal = apply_temperature_scaling(online_p, temperature)

    def _online_block(mask: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
        return _calibration_block(online_y[mask], p[mask], n_bins)

    early_mask = hours <= early_postop_hour_max + 1e-9

    by_hour: Dict[str, Any] = {}
    if postop_hour_bin_width > 0 and len(hours) > 0:
        bucket_ids = np.floor(hours / float(postop_hour_bin_width)).astype(int)
        for b in sorted(np.unique(bucket_ids)):
            mask = bucket_ids == b
            if int(mask.sum()) < int(min_bucket_n):
                continue
            h0 = float(b) * postop_hour_bin_width
            h1 = h0 + postop_hour_bin_width
            key = str(int(b))
            by_hour[key] = {
                "postop_hour_range": [h0, h1],
                "uncalibrated": _calibration_block(online_y[mask], online_p[mask], n_bins),
                "temperature_scaled": _calibration_block(online_y[mask], online_p_cal[mask], n_bins),
            }

    coarse_windows = {
        "postop_0_24h": (0.0, 24.0),
        "postop_24_72h": (24.0, 72.0),
        "postop_72h_plus": (72.0, None),
    }
    by_coarse: Dict[str, Any] = {}
    for name, (h0, h1) in coarse_windows.items():
        mask = _mask_online_window(hours, h0, h1)
        if mask.sum() == 0:
            continue
        by_coarse[name] = {
            "postop_hour_range": [h0, h1 if h1 is not None else None],
            "uncalibrated": _calibration_block(online_y[mask], online_p[mask], n_bins),
            "temperature_scaled": _calibration_block(online_y[mask], online_p_cal[mask], n_bins),
        }

    return {
        "baseline_episode": {
            "uncalibrated": _calibration_block(base_y, base_probs, n_bins),
            "temperature_scaled": _calibration_block(base_y, base_probs_cal, n_bins),
        },
        "online_early_postop_0_48h": {
            "uncalibrated": _calibration_block(online_y[early_mask], online_p[early_mask], n_bins),
            "temperature_scaled": _calibration_block(
                online_y[early_mask], online_p_cal[early_mask], n_bins
            ),
        },
        "online_all_postop_steps": {
            "uncalibrated": _calibration_block(online_y, online_p, n_bins),
            "temperature_scaled": _calibration_block(online_y, online_p_cal, n_bins),
        },
        "by_coarse_postop_window": by_coarse,
        "by_postop_hour_bucket": by_hour,
        "postop_hour_bin_width": float(postop_hour_bin_width),
        "probability_bins": int(n_bins),
        "min_bucket_n": int(min_bucket_n),
        "temperature_scaling_T": float(temperature),
        "label_note": (
            "Episode-level AKI label at each postoperative step (incidence framing). "
            "Hour buckets use floor(time_hour / postop_hour_bin_width)."
        ),
    }


def _expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if mask.sum() == 0:
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def _calibration_slope_intercept(y: np.ndarray, p: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if len(y) < 2 or len(np.unique(y)) < 2:
        return None, None
    p_clip = np.clip(p, 1e-6, 1 - 1e-6)
    logit_p = np.log(p_clip / (1 - p_clip))
    try:
        coef = np.polyfit(logit_p, y, 1)
        return float(coef[0]), float(coef[1])
    except Exception:
        return None, None


def _draw_reliability_on_ax(
    ax: Any,
    block: Dict[str, Any],
    *,
    label: str,
    linestyle: str = "-",
    color: Optional[str] = None,
) -> None:
    curve = block.get("reliability_curve") or {}
    xs = curve.get("mean_predicted") or []
    ys = curve.get("fraction_positive") or []
    if not xs:
        return
    ax.plot(xs, ys, linestyle=linestyle, marker="o", ms=4, label=label, color=color)


def _plot_dual_calibration_panel(
    ax: Any,
    dual_block: Dict[str, Any],
    title: str,
    *,
    show_temperature: bool,
) -> None:
    uncal = dual_block.get("uncalibrated") or dual_block
    _draw_reliability_on_ax(ax, uncal, label="uncalibrated", linestyle="-", color="C0")
    if show_temperature:
        cal = dual_block.get("temperature_scaled")
        if cal:
            _draw_reliability_on_ax(ax, cal, label="temperature-scaled", linestyle="--", color="C1")
    ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction AKI+")
    n = uncal.get("n", 0)
    ece = uncal.get("ece")
    ece_s = f"{ece:.3f}" if ece is not None else "n/a"
    ax.set_title(f"{title}\n(n={n}, ECE={ece_s})")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.25)


def plot_calibration_figures(
    calibration: Dict[str, Any],
    out_dir: Any,
    dpi: int = 150,
    max_hour_bucket_panels: int = 12,
) -> List[str]:
    """Save calibration reliability diagrams; returns paths written."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    T = float(calibration.get("temperature_scaling_T", 1.0))
    show_temp = abs(T - 1.0) > 1e-6
    written: List[str] = []

    # --- Overview: baseline + pooled online + coarse postop windows ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    panels = [
        (axes[0, 0], calibration.get("baseline_episode"), "Baseline (episode)"),
        (axes[0, 1], calibration.get("online_all_postop_steps"), "Online all steps"),
        (axes[0, 2], calibration.get("online_early_postop_0_48h"), "Online 0–48h"),
        (axes[1, 0], (calibration.get("by_coarse_postop_window") or {}).get("postop_0_24h"), "Online 0–24h"),
        (axes[1, 1], (calibration.get("by_coarse_postop_window") or {}).get("postop_24_72h"), "Online 24–72h"),
        (axes[1, 2], (calibration.get("by_coarse_postop_window") or {}).get("postop_72h_plus"), "Online 72h+"),
    ]
    for ax, block, title in panels:
        if block:
            _plot_dual_calibration_panel(ax, block, title, show_temperature=show_temp)
        else:
            ax.axis("off")
            ax.set_title(title)
    fig.suptitle(
        f"Calibration (T={T:.3f})" if show_temp else "Calibration (uncalibrated)",
        fontsize=11,
    )
    fig.tight_layout()
    p_overview = out / "calibration_overview.png"
    fig.savefig(p_overview, dpi=dpi)
    plt.close(fig)
    written.append(str(p_overview))

    # --- Per postoperative hour bucket (6h default) ---
    by_hour = calibration.get("by_postop_hour_bucket") or {}
    if by_hour:
        keys = sorted(by_hour.keys(), key=lambda k: int(k))[: max_hour_bucket_panels]
        ncols = 4
        nrows = int(np.ceil(len(keys) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()
        bin_w = float(calibration.get("postop_hour_bin_width", 6.0))
        for ax, key in zip(axes_flat, keys):
            block = by_hour[key]
            hr = block.get("postop_hour_range", [int(key) * bin_w, (int(key) + 1) * bin_w])
            title = f"Postop {hr[0]:.0f}–{hr[1]:.0f}h"
            _plot_dual_calibration_panel(ax, block, title, show_temperature=show_temp)
        for ax in axes_flat[len(keys) :]:
            ax.axis("off")
        fig.suptitle("Calibration by postoperative hour bucket", fontsize=11)
        fig.tight_layout()
        p_hour = out / "calibration_by_postop_hour_bucket.png"
        fig.savefig(p_hour, dpi=dpi)
        plt.close(fig)
        written.append(str(p_hour))

        # ECE vs bucket mid-hour
        mids: List[float] = []
        ece_u: List[float] = []
        ece_c: List[float] = []
        for key in sorted(by_hour.keys(), key=lambda k: int(k)):
            block = by_hour[key]
            hr = block.get("postop_hour_range", [0, bin_w])
            mid = 0.5 * (float(hr[0]) + float(hr[1]))
            eu = (block.get("uncalibrated") or {}).get("ece")
            ec = (block.get("temperature_scaled") or {}).get("ece")
            if eu is None:
                continue
            mids.append(mid)
            ece_u.append(float(eu))
            ece_c.append(float(ec) if ec is not None else float("nan"))
        if mids:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(mids, ece_u, "o-", label="uncalibrated ECE", color="C0")
            if show_temp and any(np.isfinite(ece_c)):
                ax.plot(mids, ece_c, "s--", label="temperature-scaled ECE", color="C1")
            ax.set_xlabel("Postoperative hour (bucket midpoint)")
            ax.set_ylabel("ECE")
            ax.set_title("Expected calibration error by postoperative hour bucket")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            p_ece = out / "calibration_ece_by_hour_bucket.png"
            fig.savefig(p_ece, dpi=dpi)
            plt.close(fig)
            written.append(str(p_ece))

    return written


def evaluate_clinical(
    trajectories: Sequence[EpisodeTrajectory],
    alert_thresholds: Sequence[float] = (0.5, 0.7, 0.9),
    calibration_n_bins: int = 10,
    postop_hour_bin_width: float = 6.0,
    calibration_min_bucket_n: int = 200,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "n_episodes": len(trajectories),
        "n_aki_episodes": int(sum(t.label_aki for t in trajectories)),
        "definitions": {
            "T_AKI": "hours from t0 to episode aki_time (AKI+ only)",
            "T_alert": "first postoperative event time with p_online >= tau",
            "early_warning_success": "T_alert < T_AKI",
            "patient_days": "max postoperative time_hour / 24 per episode",
        },
    }
    by_tau: Dict[str, Any] = {}
    for tau in alert_thresholds:
        key = f"tau_{tau:g}"
        by_tau[key] = {
            "early_warning": early_warning_metrics(trajectories, tau),
            "false_alert_burden": false_alert_burden_metrics(trajectories, tau),
        }
    report["by_alert_threshold"] = by_tau
    report["alert_burden_ranking"] = alert_burden_metrics(trajectories)
    report["baseline_vs_online"] = baseline_vs_online_metrics(trajectories)
    report["calibration"] = calibration_metrics(
        trajectories,
        n_bins=calibration_n_bins,
        postop_hour_bin_width=postop_hour_bin_width,
        min_bucket_n=calibration_min_bucket_n,
        temperature=temperature,
    )
    return report
