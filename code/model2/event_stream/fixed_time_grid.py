"""Fixed postoperative time-grid evaluation (aligned with GRU-D / LSTM bucket protocols).

At each grid point ``b`` (width ``bucket_size`` hours), truncate the episode to events with
``time_hour < b + bucket_size``, run one forward pass, and take the **last postoperative**
token probability when available. With ``carry_forward`` (default), if a bucket has no postop
prediction yet, use the previous bucket's risk (first bucket falls back to ``[BASE]``).

Two implementations produce identical numbers:

- ``evaluate_fixed_time_grid`` re-forwards the truncated episode once per grid point.
- ``evaluate_fixed_time_grid_fast`` runs a single batched forward per episode and reads the
  grid off the resulting token probabilities. This is exact rather than an approximation: the
  block mask restricts postoperative query ``i`` to ``[BASE]`` plus postoperative keys ``<= i``,
  ``[BASE]`` sees only preoperative keys, and truncation drops only trailing postoperative
  tokens. So dropping them leaves ``base_idx``, every retained token's features (including the
  per-``time_day`` medication exposure vector) and every retained attention pattern unchanged.
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from event_stream.dataset import collate_fn
from event_stream.input_builder import episode_to_sequence_features
from common.utils import _forward_model, _subset_classification_metrics, safe_auprc, safe_auroc


def _episode_label(ep: Dict[str, Any]) -> float:
    lbl = ep.get("label_aki")
    return float(lbl) if lbl is not None else float("nan")


def _last_event_hour(events: Sequence[Dict[str, Any]]) -> float:
    hours = [
        float(ev["time_hour"])
        for ev in events
        if ev.get("time_hour") is not None and np.isfinite(ev.get("time_hour"))
    ]
    return float(max(hours)) if hours else float("nan")


def _events_before_cutoff(ep: Dict[str, Any], cutoff: float) -> List[Dict[str, Any]]:
    events = ep.get("events", []) or []
    out: List[Dict[str, Any]] = []
    for ev in events:
        th = ev.get("time_hour")
        if th is None:
            continue
        if float(th) < float(cutoff):
            out.append(ev)
    return out


def _max_bucket_for_episode(
    events: Sequence[Dict[str, Any]],
    label_aki: int,
    buckets: np.ndarray,
    bucket_size: float,
) -> int:
    """Last bucket index to evaluate (inclusive left edge), matching simple_model follow rules."""
    last_bucket_left = int(buckets[-2]) if len(buckets) >= 2 else 0
    if label_aki != 1:
        return last_bucket_left
    last_hour = _last_event_hour(events)
    if not np.isfinite(last_hour):
        return last_bucket_left
    for b_left, b_right in zip(buckets[:-1], buckets[1:]):
        if last_hour >= b_left and last_hour < b_right:
            return int(b_left)
    return last_bucket_left


@torch.no_grad()
def predict_probs_on_truncated_episode(
    ep_sub: Dict[str, Any],
    model: torch.nn.Module,
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    feature_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (last_postop_prob, baseline_prob) after one forward on truncated episode."""
    feat = episode_to_sequence_features(
        ep_sub,
        code_vocab,
        add_base=True,
        lab_stats=lab_stats,
        num_med_exposure=num_med_exposure,
        **(feature_kwargs or {}),
    )
    seq = feat["sequence"]
    if not seq:
        return None, None
    batch = collate_fn([feat])
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = _forward_model(model, batch)
    probs = torch.sigmoid(outputs["token_logits"])[0].cpu().numpy()
    last_postop: Optional[float] = None
    baseline: Optional[float] = None
    for tok, p in zip(seq, probs):
        if int(tok.get("is_base", 0)) == 1:
            baseline = float(p)
        if int(tok.get("is_postop", 0)) == 1:
            last_postop = float(p)
    return last_postop, baseline


def episode_fixed_grid_rows(
    ep: Dict[str, Any],
    model: torch.nn.Module,
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    bucket_size: float = 6.0,
    bucket_start: float = 0.0,
    bucket_end: float = 720.0,
    carry_forward: bool = True,
    feature_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """One row per (episode, postoperative bucket) with grid-aligned risk."""
    events = ep.get("events", []) or []
    y_ep = int(_episode_label(ep) >= 0.5)
    person_id = ep.get("person_id")
    buckets = np.arange(bucket_start, bucket_end + bucket_size, bucket_size, dtype=np.float64)
    if len(buckets) < 2:
        return []
    max_b = _max_bucket_for_episode(events, y_ep, buckets, bucket_size)

    rows: List[Dict[str, Any]] = []
    carry_prob: Optional[float] = None
    for b_left, b_right in zip(buckets[:-1], buckets[1:]):
        b = int(b_left)
        if b > max_b:
            break
        cutoff = float(b_right)
        sub_events = _events_before_cutoff(ep, cutoff)
        y_prob = float("nan")
        y_prob_raw_postop: Optional[float] = None
        baseline_prob = float("nan")
        carry_source = "missing"
        n_events = len(sub_events)
        if n_events > 0:
            ep_sub = dict(ep)
            ep_sub["events"] = sub_events
            try:
                y_prob_v, base_v = predict_probs_on_truncated_episode(
                    ep_sub, model, code_vocab, lab_stats, num_med_exposure, device,
                    feature_kwargs=feature_kwargs,
                )
                if y_prob_v is not None:
                    y_prob_raw_postop = float(y_prob_v)
                if base_v is not None:
                    baseline_prob = float(base_v)
            except Exception:
                pass

        if carry_forward:
            if y_prob_raw_postop is not None:
                y_prob = y_prob_raw_postop
                carry_prob = y_prob_raw_postop
                carry_source = "postop"
            elif carry_prob is not None:
                y_prob = carry_prob
                carry_source = "carry_prev"
            elif np.isfinite(baseline_prob):
                y_prob = baseline_prob
                carry_prob = baseline_prob
                carry_source = "baseline"
        else:
            if y_prob_raw_postop is not None:
                y_prob = y_prob_raw_postop
                carry_source = "postop"
            elif np.isfinite(baseline_prob):
                y_prob = baseline_prob
                carry_source = "baseline"

        rows.append(
            {
                "person_id": person_id,
                "bucket_hour": b,
                "bucket_hour_end": float(b_right),
                "hour_range": f"[{b:g},{b_right:g})",
                "y_true": y_ep,
                "y_prob": y_prob,
                "y_prob_raw_postop": y_prob_raw_postop,
                "carry_source": carry_source,
                "baseline_prob": baseline_prob,
                "n_events_truncated": n_events,
            }
        )
    return rows


def episode_grid_rows_from_token_probs(
    ep: Dict[str, Any],
    seq: Sequence[Dict[str, Any]],
    probs: Sequence[float],
    bucket_size: float = 6.0,
    bucket_start: float = 0.0,
    bucket_end: float = 720.0,
    carry_forward: bool = True,
) -> List[Dict[str, Any]]:
    """Fold one episode's token probabilities onto the grid, without re-forwarding.

    ``probs`` must align with ``seq`` from a single forward on the untruncated episode.
    """
    events = ep.get("events", []) or []
    y_ep = int(_episode_label(ep) >= 0.5)
    person_id = ep.get("person_id")
    buckets = np.arange(bucket_start, bucket_end + bucket_size, bucket_size, dtype=np.float64)
    if len(buckets) < 2:
        return []
    max_b = _max_bucket_for_episode(events, y_ep, buckets, bucket_size)

    baseline_prob_full = float("nan")
    postop_hours: List[float] = []
    postop_probs: List[float] = []
    for tok, p in zip(seq, probs):
        if int(tok.get("is_base", 0)) == 1:
            baseline_prob_full = float(p)
        if int(tok.get("is_postop", 0)) == 1:
            postop_hours.append(float(tok.get("time_hour", 0.0)))
            postop_probs.append(float(p))

    # The slow path takes the last postoperative token *in sequence order* among those retained.
    # When hours are non-decreasing that is a bisect; otherwise fall back to a scan.
    hours_sorted = all(
        postop_hours[i] <= postop_hours[i + 1] for i in range(len(postop_hours) - 1)
    )

    def last_postop_before(cutoff: float) -> Optional[float]:
        if not postop_hours:
            return None
        if hours_sorted:
            j = bisect_left(postop_hours, cutoff) - 1
            return postop_probs[j] if j >= 0 else None
        out: Optional[float] = None
        for h, p in zip(postop_hours, postop_probs):
            if h < cutoff:
                out = p
        return out

    # Counting events below a cutoff is order-independent, so a sorted copy is always exact.
    event_hours = sorted(
        float(ev["time_hour"])
        for ev in events
        if ev.get("time_hour") is not None
    )

    rows: List[Dict[str, Any]] = []
    carry_prob: Optional[float] = None
    for b_left, b_right in zip(buckets[:-1], buckets[1:]):
        b = int(b_left)
        if b > max_b:
            break
        cutoff = float(b_right)
        n_events = bisect_left(event_hours, cutoff)
        y_prob = float("nan")
        y_prob_raw_postop: Optional[float] = None
        baseline_prob = float("nan")
        carry_source = "missing"
        # With no retained events the slow path skips the forward entirely, so nothing is known.
        if n_events > 0:
            y_prob_raw_postop = last_postop_before(cutoff)
            baseline_prob = baseline_prob_full

        if carry_forward:
            if y_prob_raw_postop is not None:
                y_prob = y_prob_raw_postop
                carry_prob = y_prob_raw_postop
                carry_source = "postop"
            elif carry_prob is not None:
                y_prob = carry_prob
                carry_source = "carry_prev"
            elif np.isfinite(baseline_prob):
                y_prob = baseline_prob
                carry_prob = baseline_prob
                carry_source = "baseline"
        else:
            if y_prob_raw_postop is not None:
                y_prob = y_prob_raw_postop
                carry_source = "postop"
            elif np.isfinite(baseline_prob):
                y_prob = baseline_prob
                carry_source = "baseline"

        rows.append(
            {
                "person_id": person_id,
                "bucket_hour": b,
                "bucket_hour_end": float(b_right),
                "hour_range": f"[{b:g},{b_right:g})",
                "y_true": y_ep,
                "y_prob": y_prob,
                "y_prob_raw_postop": y_prob_raw_postop,
                "carry_source": carry_source,
                "baseline_prob": baseline_prob,
                "n_events_truncated": n_events,
            }
        )
    return rows


@torch.no_grad()
def evaluate_fixed_time_grid_fast(
    episodes: Sequence[Dict[str, Any]],
    model: torch.nn.Module,
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    bucket_size: float = 6.0,
    bucket_start: float = 0.0,
    bucket_end: float = 720.0,
    cls_threshold: float = 0.5,
    macro_min_n: int = 10,
    max_episodes: int = 0,
    progress: bool = True,
    carry_forward: bool = True,
    batch_size: int = 16,
    feature_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Same protocol as ``evaluate_fixed_time_grid``, one batched forward per episode."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    eps = list(episodes)
    if max_episodes > 0:
        eps = eps[:max_episodes]

    all_rows: List[Dict[str, Any]] = []
    batches = [eps[i : i + batch_size] for i in range(0, len(eps), batch_size)]
    iterator: Any = batches
    if progress and tqdm is not None:
        iterator = tqdm(batches, desc="fixed_grid batches")

    for chunk in iterator:
        feats = []
        kept = []
        for ep in chunk:
            feat = episode_to_sequence_features(
                ep,
                code_vocab,
                add_base=True,
                lab_stats=lab_stats,
                num_med_exposure=num_med_exposure,
                **(feature_kwargs or {}),
            )
            if feat["sequence"]:
                feats.append(feat)
                kept.append(ep)
        if not feats:
            continue
        batch = collate_fn(feats)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = _forward_model(model, batch)
        probs = torch.sigmoid(outputs["token_logits"]).cpu().numpy()
        for i, (ep, feat) in enumerate(zip(kept, feats)):
            seq = feat["sequence"]
            all_rows.extend(
                episode_grid_rows_from_token_probs(
                    ep,
                    seq,
                    probs[i, : len(seq)],
                    bucket_size=bucket_size,
                    bucket_start=bucket_start,
                    bucket_end=bucket_end,
                    carry_forward=carry_forward,
                )
            )

    return assemble_grid_report(
        all_rows,
        n_episodes=len(eps),
        bucket_size=bucket_size,
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        cls_threshold=cls_threshold,
        macro_min_n=macro_min_n,
        carry_forward=carry_forward,
    )


def _macro_mean_bucket_auprc(
    by_bucket: Dict[str, Any],
    min_n: int = 10,
) -> Optional[float]:
    vals: List[float] = []
    for _, stats in sorted(by_bucket.items(), key=lambda x: int(x[0])):
        if not isinstance(stats, dict):
            continue
        if int(stats.get("n", 0)) < min_n:
            continue
        a = stats.get("auprc")
        if a is not None:
            vals.append(float(a))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def evaluate_fixed_time_grid(
    episodes: Sequence[Dict[str, Any]],
    model: torch.nn.Module,
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    device: torch.device,
    bucket_size: float = 6.0,
    bucket_start: float = 0.0,
    bucket_end: float = 720.0,
    cls_threshold: float = 0.5,
    macro_min_n: int = 10,
    max_episodes: int = 0,
    progress: bool = True,
    carry_forward: bool = True,
    feature_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run fixed-grid evaluation; returns metrics and optional row table."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None  # type: ignore

    eps = list(episodes)
    if max_episodes > 0:
        eps = eps[:max_episodes]

    all_rows: List[Dict[str, Any]] = []
    iterator: Any = eps
    if progress and tqdm is not None:
        iterator = tqdm(eps, desc="fixed_grid episodes")

    for ep in iterator:
        all_rows.extend(
            episode_fixed_grid_rows(
                ep,
                model,
                code_vocab,
                lab_stats,
                num_med_exposure,
                device,
                bucket_size=bucket_size,
                bucket_start=bucket_start,
                bucket_end=bucket_end,
                carry_forward=carry_forward,
                feature_kwargs=feature_kwargs,
            )
        )

    return assemble_grid_report(
        all_rows,
        n_episodes=len(eps),
        bucket_size=bucket_size,
        bucket_start=bucket_start,
        bucket_end=bucket_end,
        cls_threshold=cls_threshold,
        macro_min_n=macro_min_n,
        carry_forward=carry_forward,
    )


def assemble_grid_report(
    all_rows: List[Dict[str, Any]],
    n_episodes: int,
    bucket_size: float,
    bucket_start: float,
    bucket_end: float,
    cls_threshold: float,
    macro_min_n: int,
    carry_forward: bool,
) -> Dict[str, Any]:
    """Turn grid rows into the metrics report (shared by the slow and fast evaluators)."""
    y_true = np.array([r["y_true"] for r in all_rows], dtype=np.float64)
    y_prob = np.array([r["y_prob"] for r in all_rows], dtype=np.float64)
    base_prob = np.array([r["baseline_prob"] for r in all_rows], dtype=np.float64)

    finite = np.isfinite(y_prob)
    y_f = y_true[finite]
    p_f = y_prob[finite]

    by_bucket: Dict[str, Any] = {}
    for b in sorted({int(r["bucket_hour"]) for r in all_rows}):
        mask = np.array([int(r["bucket_hour"]) == b for r in all_rows], dtype=bool) & finite
        if mask.sum() == 0:
            continue
        yt = y_true[mask]
        yp = y_prob[mask]
        bp = base_prob[mask]
        block = _subset_classification_metrics(yt, yp, cls_threshold)
        block["brier"] = float(np.mean((yp - yt) ** 2))
        block["mean_predicted_prob"] = float(np.mean(yp))
        block["mean_baseline_prob"] = float(np.mean(bp[np.isfinite(bp)])) if np.isfinite(bp).any() else None
        by_bucket[str(b)] = block

    # Episode-level: last valid grid row per person
    last_by_person: Dict[Any, Dict[str, Any]] = {}
    for r in all_rows:
        if not np.isfinite(r["y_prob"]):
            continue
        pid = r["person_id"]
        if pid not in last_by_person or r["bucket_hour"] >= last_by_person[pid]["bucket_hour"]:
            last_by_person[pid] = r

    ep_rows = list(last_by_person.values())
    if ep_rows:
        y_ep = np.array([r["y_true"] for r in ep_rows], dtype=np.float64)
        p_ep = np.array([r["y_prob"] for r in ep_rows], dtype=np.float64)
        p_base_ep = np.array([r["baseline_prob"] for r in ep_rows], dtype=np.float64)
    else:
        y_ep = np.array([], dtype=np.float64)
        p_ep = np.array([], dtype=np.float64)
        p_base_ep = np.array([], dtype=np.float64)

    macro_auprc = _macro_mean_bucket_auprc(by_bucket, min_n=macro_min_n)

    return {
        "protocol": {
            "description": (
                "Fixed postoperative hour grid; at each bucket re-forward on truncated history "
                "(time_hour < bucket_end). Use last postoperative token probability when present; "
                "otherwise carry forward previous bucket risk (first bucket uses [BASE] baseline)."
            ),
            "bucket_size_hours": float(bucket_size),
            "bucket_start_hours": float(bucket_start),
            "bucket_end_hours": float(bucket_end),
            "carry_forward": bool(carry_forward),
            "label": "episode-level AKI label at each grid point",
            "aki_positive_follow": "AKI+ episodes: buckets only through last event hour bin",
        },
        "n_episodes": int(n_episodes),
        "n_grid_rows": len(all_rows),
        "n_grid_rows_finite_prob": int(finite.sum()),
        "pooled_grid": _subset_classification_metrics(y_f, p_f, cls_threshold) if len(y_f) else {"n": 0},
        "episode_last_grid": _subset_classification_metrics(y_ep, p_ep, cls_threshold) if len(y_ep) else {"n": 0},
        "baseline_at_last_grid": _subset_classification_metrics(y_ep, p_base_ep, cls_threshold)
        if len(y_ep)
        else {"n": 0},
        "by_bucket_hour": by_bucket,
        "summary": {
            "macro_bucket_auprc": macro_auprc,
            "pooled_grid_auprc": safe_auprc(y_f, p_f) if len(y_f) else None,
            "pooled_grid_auroc": safe_auroc(y_f, p_f) if len(y_f) else None,
            "episode_last_grid_auprc": safe_auprc(y_ep, p_ep) if len(y_ep) else None,
            "n_buckets_reported": len(by_bucket),
            "n_buckets_in_macro": sum(
                1
                for s in by_bucket.values()
                if int(s.get("n", 0)) >= macro_min_n and s.get("auprc") is not None
            ),
        },
        "grid_rows_sample": all_rows[:20],
    }


def build_bucket_table(by_bucket: Dict[str, Any], bucket_size: float) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(by_bucket.keys(), key=lambda k: int(k)):
        b = int(key)
        stats = by_bucket[key]
        h0 = float(b)
        h1 = float(b) + float(bucket_size)
        rows.append(
            {
                "bucket_hour": b,
                "hour_start": h0,
                "hour_end": h1,
                "hour_range": f"[{h0:g},{h1:g})",
                "n": int(stats.get("n", 0)),
                "n_pos": stats.get("n_pos"),
                "n_neg": stats.get("n_neg"),
                "pos_rate": stats.get("pos_rate"),
                "pos_neg_ratio": stats.get("pos_neg_ratio"),
                "auprc": stats.get("auprc"),
                "auroc": stats.get("auroc"),
                "f1": stats.get("f1"),
                "precision": stats.get("precision"),
                "recall": stats.get("recall"),
                "accuracy": stats.get("accuracy"),
                "specificity": stats.get("specificity"),
                "brier": stats.get("brier"),
                "mean_predicted_prob": stats.get("mean_predicted_prob"),
                "mean_baseline_prob": stats.get("mean_baseline_prob"),
                "hour_mid": 0.5 * (h0 + h1),
            }
        )
    return rows


# Metrics to plot vs postoperative hour (bucket midpoint on x-axis).
BUCKET_LINE_METRICS: List[Tuple[str, str, Optional[Tuple[float, float]]]] = [
    ("auprc", "AUPRC", (0.0, 1.0)),
    ("auroc", "AUROC", (0.0, 1.0)),
    ("f1", "F1", (0.0, 1.0)),
    ("precision", "Precision", (0.0, 1.0)),
    ("recall", "Recall", (0.0, 1.0)),
    ("accuracy", "Accuracy", (0.0, 1.0)),
    ("specificity", "Specificity", (0.0, 1.0)),
    ("brier", "Brier score", None),
    ("pos_rate", "Positive rate (prevalence)", (0.0, 1.0)),
    ("pos_neg_ratio", "Pos/Neg ratio", None),
    ("mean_predicted_prob", "Mean predicted probability", (0.0, 1.0)),
    ("mean_baseline_prob", "Mean baseline probability", (0.0, 1.0)),
    ("n", "Sample count (n)", None),
    ("n_pos", "AKI+ count", None),
    ("n_neg", "AKI− count", None),
]


def plot_bucket_metric_curves(
    table: List[Dict[str, Any]],
    out_dir: Any,
    dpi: int = 150,
    bucket_size: float = 6.0,
) -> List[str]:
    """One line plot per metric vs postoperative hour; plus a combined summary figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    from pathlib import Path

    if not table:
        return []

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: List[str] = []

    x = np.array([float(r.get("hour_mid", r["hour_start"] + bucket_size / 2.0)) for r in table])

    def _series(key: str) -> np.ndarray:
        vals = []
        for r in table:
            v = r.get(key)
            vals.append(float(v) if v is not None else np.nan)
        return np.array(vals, dtype=np.float64)

    # Individual metric plots
    for key, title, ylim in BUCKET_LINE_METRICS:
        y = _series(key)
        if not np.any(np.isfinite(y)):
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(x, y, "o-", color="C0", markersize=4, linewidth=1.2)
        ax.set_xlabel("Postoperative hour (bucket midpoint)")
        ax.set_ylabel(title)
        ax.set_title(f"Fixed grid: {title} by {bucket_size:g}h bucket")
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out / f"fixed_grid_by_bucket_{key}.png"
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(str(path))

    # Combined discrimination panel
    disc_keys = ["auprc", "auroc", "f1", "precision", "recall", "accuracy"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, key in zip(axes.flatten(), disc_keys):
        y = _series(key)
        label = next((t for k, t, _ in BUCKET_LINE_METRICS if k == key), key)
        if np.any(np.isfinite(y)):
            ax.plot(x, y, "o-", ms=3, lw=1.0)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Postop hour", fontsize=8)
        ax.grid(True, alpha=0.25)
        if key in ("auprc", "auroc", "f1", "precision", "recall", "accuracy"):
            ax.set_ylim(0, 1)
    fig.suptitle(f"Discrimination metrics vs {bucket_size:g}h postoperative buckets", fontsize=11)
    fig.tight_layout()
    p_panel = out / "fixed_grid_by_bucket_discrimination_panel.png"
    fig.savefig(p_panel, dpi=dpi)
    plt.close(fig)
    written.append(str(p_panel))

    # Counts + calibration-related panel
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    panels = [
        (axes[0, 0], "n", "n"),
        (axes[0, 1], "n_pos", "n_pos"),
        (axes[1, 0], "mean_predicted_prob", "mean_predicted_prob"),
        (axes[1, 1], "brier", "brier"),
    ]
    for ax, key, _ in panels:
        y = _series(key)
        if np.any(np.isfinite(y)):
            ax.plot(x, y, "o-", ms=3, lw=1.0)
        ax.set_title(next((t for k, t, _ in BUCKET_LINE_METRICS if k == key), key), fontsize=9)
        ax.set_xlabel("Postop hour", fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Counts & calibration vs {bucket_size:g}h buckets", fontsize=11)
    fig.tight_layout()
    p_aux = out / "fixed_grid_by_bucket_counts_calibration_panel.png"
    fig.savefig(p_aux, dpi=dpi)
    plt.close(fig)
    written.append(str(p_aux))

    return written
