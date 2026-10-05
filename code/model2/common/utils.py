from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, average_precision_score


def resolve_torch_device(device_str: str) -> torch.device:
    s = device_str.strip().lower()
    if s == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if s == "cpu":
        return torch.device("cpu")
    if s in ("cuda", "gpu"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        return torch.device("cuda:0")
    if s.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        idx = int(s.split(":", 1)[1])
        if idx < 0 or idx >= torch.cuda.device_count():
            raise ValueError(f"Invalid GPU index {idx}; count={torch.cuda.device_count()}")
        return torch.device(f"cuda:{idx}")
    raise ValueError("Use device auto, cpu, cuda, or cuda:N")


def save_json(obj: Dict[str, Any], output_path: str | Path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _mask_finite_pairs(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_prob)
    return y_true[m], y_prob[m]


def safe_auroc(y_true, y_prob):
    y_true, y_prob = _mask_finite_pairs(y_true, y_prob)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_prob))


def safe_auprc(y_true, y_prob):
    y_true, y_prob = _mask_finite_pairs(y_true, y_prob)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_prob))


def threshold_classification_metrics(y_true, y_prob, threshold: float = 0.5) -> Dict[str, Any]:
    y_true, y_prob = _mask_finite_pairs(y_true, y_prob)
    empty_cm = {"tn": None, "fp": None, "fn": None, "tp": None}
    empty = {
        "accuracy": None,
        "specificity": None,
        "sensitivity": None,
        "recall": None,
        "precision": None,
        "f1": None,
        "confusion_matrix": empty_cm,
        "confusion_matrix_2x2": None,
    }
    if len(y_true) == 0:
        return empty
    y_int = y_true.astype(np.int64)
    y_hat = (y_prob >= threshold).astype(np.int64)
    cm = confusion_matrix(y_int, y_hat, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    acc = float(accuracy_score(y_int, y_hat))
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else None
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = float(2 * precision * recall / (precision + recall))
    else:
        f1 = None
    return {
        "accuracy": acc,
        "specificity": specificity,
        "sensitivity": recall,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "confusion_matrix_2x2": [[tn, fp], [fn, tp]],
    }


def label_balance_metrics(y_true: np.ndarray) -> Dict[str, Any]:
    """Count positive/negative labels and pos/neg ratio (n_pos / n_neg)."""
    y = np.asarray(y_true, dtype=np.float64)
    y = y[np.isfinite(y)]
    n_pos = int(np.sum(y >= 0.5))
    n_neg = int(np.sum(y < 0.5))
    n = n_pos + n_neg
    pos_rate = float(n_pos / n) if n > 0 else None
    pos_neg_ratio = float(n_pos / n_neg) if n_neg > 0 else None
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "pos_rate": pos_rate,
        "pos_neg_ratio": pos_neg_ratio,
    }


def _subset_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cls_threshold: float,
) -> Dict[str, Any]:
    return {
        "n": int(len(np.asarray(y_true))),
        **label_balance_metrics(y_true),
        "auroc": safe_auroc(y_true, y_prob),
        "auprc": safe_auprc(y_true, y_prob),
        **threshold_classification_metrics(y_true, y_prob, cls_threshold),
    }


def build_online_cls_targets(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]
    postop_mask = batch["postop_mask"]
    targets = labels.unsqueeze(1).repeat(1, attention_mask.shape[1])
    loss_mask = (attention_mask > 0) & (postop_mask > 0)
    return targets.float(), loss_mask.float()


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    return (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)


def masked_bce_with_logits_episode_normalized(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
    pos_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Online-style BCE: mean over batch of per-episode masked means (equal episode weight).

    For each episode b: L_b = sum_t(loss_bt * m_bt) / (sum_t m_bt + eps), then return mean_b L_b
    over episodes with at least one masked position. Avoids long, event-dense episodes dominating
    the loss vs. short episodes.
    """
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
    weighted = loss * loss_mask
    per_ep_sum = weighted.sum(dim=1)
    per_ep_cnt = loss_mask.sum(dim=1)
    valid = per_ep_cnt > 0
    if not valid.any():
        return logits.sum() * 0.0
    per_ep_mean = per_ep_sum / (per_ep_cnt + eps)
    return per_ep_mean[valid].mean()


def baseline_bce_loss(
    baseline_logits_all: torch.Tensor,
    labels: torch.Tensor,
    base_mask: torch.Tensor,
    pos_weight: float = 1.0,
) -> torch.Tensor:
    base_bool = base_mask > 0
    if base_bool.sum().item() == 0:
        return baseline_logits_all.sum() * 0.0
    logits = baseline_logits_all[base_bool]
    targets = labels.unsqueeze(1).repeat(1, base_mask.shape[1])[base_bool]
    pw = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets.float(), pos_weight=pw)


def make_mem_masked_inputs(
    batch: Dict[str, torch.Tensor],
    mask_token_id: int,
    mask_ratio: float = 0.15,
    zero_unknown_med_exposure: bool = True,
) -> Dict[str, torch.Tensor]:
    """Create masked preoperative view for masked event-code reconstruction.

    Only preoperative condition/medication/lab candidates are sampled. The code ID is
    replaced by [MASK]. For masked lab tokens, value/value_mask are also hidden. For
    masked medication tokens, the corresponding daily-exposure dimension is zeroed
    across the whole sequence; if the medication dimension is unknown, the full
    exposure vector is zeroed for that episode when `zero_unknown_med_exposure=True`.
    """
    out: Dict[str, torch.Tensor] = {
        "code_ids": batch["code_ids"].clone(),
        "modality_ids": batch["modality_ids"],
        "time_days": batch["time_days"],
        "time_hours": batch["time_hours"],
        "values": batch["values"].clone(),
        "value_masks": batch["value_masks"].clone(),
        "attention_mask": batch["attention_mask"],
        "base_mask": batch["base_mask"],
    }
    if "event_type_ids" in batch:
        out["event_type_ids"] = batch["event_type_ids"]
    if "age_values" in batch:
        out["age_values"] = batch["age_values"]
    if "age_masks" in batch:
        out["age_masks"] = batch["age_masks"]
    if "preop_mask" in batch:
        out["preop_mask"] = batch["preop_mask"]
    if "postop_mask" in batch:
        out["postop_mask"] = batch["postop_mask"]
    if "med_day_feats" in batch:
        out["med_day_feats"] = batch["med_day_feats"].clone()

    candidate = (batch["mem_candidate_mask"] > 0) & (batch["attention_mask"] > 0)
    rand = torch.rand_like(batch["values"])
    mem_mask = candidate & (rand < float(mask_ratio))
    mem_targets = batch["code_ids"].clone()

    out["code_ids"][mem_mask] = int(mask_token_id)

    # Hide lab values when lab test code is masked.
    lab_mask = mem_mask & (batch["modality_ids"] == 3)
    out["values"][lab_mask] = 0.0
    out["value_masks"][lab_mask] = 0.0

    # Prevent medication code reconstruction from trivially reading r_i.
    if "med_day_feats" in out:
        med_mask = mem_mask & (batch["modality_ids"] == 4)
        if med_mask.any().item():
            B, L = med_mask.shape
            med_dim = out["med_day_feats"].shape[-1]
            for b in range(B):
                masked_positions = torch.nonzero(med_mask[b], as_tuple=False).flatten()
                if masked_positions.numel() == 0:
                    continue
                unknown_seen = False
                for pos in masked_positions.tolist():
                    idx = int(batch["med_exposure_indices"][b, pos].item())
                    if 0 <= idx < med_dim:
                        out["med_day_feats"][b, :, idx] = 0.0
                    else:
                        unknown_seen = True
                if unknown_seen and zero_unknown_med_exposure:
                    out["med_day_feats"][b, :, :] = 0.0

    out["mem_mask"] = mem_mask.float()
    out["mem_targets"] = mem_targets
    return out


def masked_event_ce_loss(mem_logits: torch.Tensor, mem_targets: torch.Tensor, mem_mask: torch.Tensor) -> torch.Tensor:
    # mem_logits [B,L,V], mem_targets [B,L], mem_mask [B,L]
    if mem_mask.sum().item() == 0:
        return mem_logits.sum() * 0.0
    B, L, V = mem_logits.shape
    loss = F.cross_entropy(mem_logits.reshape(B * L, V), mem_targets.reshape(B * L), reduction="none")
    loss = loss.reshape(B, L)
    return (loss * mem_mask).sum() / (mem_mask.sum() + 1e-8)


def _forward_model(model, batch: Dict[str, torch.Tensor]):
    kw = {
        "code_ids": batch["code_ids"],
        "modality_ids": batch["modality_ids"],
        "time_days": batch["time_days"],
        "time_hours": batch["time_hours"],
        "values": batch["values"],
        "value_masks": batch["value_masks"],
        "attention_mask": batch["attention_mask"],
        "base_mask": batch["base_mask"],
    }
    if "med_day_feats" in batch:
        kw["med_day_feats"] = batch["med_day_feats"]
    if "event_type_ids" in batch:
        kw["event_type_ids"] = batch["event_type_ids"]
    if "age_values" in batch:
        kw["age_values"] = batch["age_values"]
    if "age_masks" in batch:
        kw["age_masks"] = batch["age_masks"]
    if "preop_mask" in batch:
        kw["preop_mask"] = batch["preop_mask"]
    if "postop_mask" in batch:
        kw["postop_mask"] = batch["postop_mask"]
    return model(**kw)


def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_targets, all_time_days, all_time_hours, all_step_indices = [], [], [], [], []
    final_probs, final_targets = [], []
    base_probs, base_targets = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = _forward_model(model, batch)
            token_probs = torch.sigmoid(outputs["token_logits"])
            online_targets, online_mask = build_online_cls_targets(batch)
            base_bool = batch["base_mask"] > 0
            if base_bool.any().item():
                bp = torch.sigmoid(outputs["baseline_logits_all"])[base_bool]
                bt = batch["labels"].unsqueeze(1).repeat(1, batch["base_mask"].shape[1])[base_bool]
                base_probs.extend(bp.detach().cpu().numpy().tolist())
                base_targets.extend(bt.detach().cpu().numpy().tolist())
            B, _ = token_probs.shape
            for i in range(B):
                valid = online_mask[i] > 0
                if valid.sum().item() == 0:
                    continue
                probs_i = token_probs[i][valid].detach().cpu().numpy()
                targets_i = online_targets[i][valid].detach().cpu().numpy()
                time_d = batch["time_days"][i][valid].detach().cpu().numpy()
                time_h = batch["time_hours"][i][valid].detach().cpu().numpy()
                step_i = np.arange(len(probs_i), dtype=np.int64)
                all_probs.append(probs_i)
                all_targets.append(targets_i)
                all_time_days.append(time_d)
                all_time_hours.append(time_h)
                all_step_indices.append(step_i)
                final_probs.append(float(probs_i[-1]))
                final_targets.append(float(targets_i[-1]))
    return {
        "probs": np.concatenate(all_probs, axis=0) if all_probs else np.array([]),
        "targets": np.concatenate(all_targets, axis=0) if all_targets else np.array([]),
        "time_days": np.concatenate(all_time_days, axis=0) if all_time_days else np.array([]),
        "time_hours": np.concatenate(all_time_hours, axis=0) if all_time_hours else np.array([]),
        "step_indices": np.concatenate(all_step_indices, axis=0) if all_step_indices else np.array([]),
        "final_probs": np.asarray(final_probs),
        "final_targets": np.asarray(final_targets),
        "base_probs": np.asarray(base_probs),
        "base_targets": np.asarray(base_targets),
    }


def postop_continuous_hours(time_days: np.ndarray, time_hours: np.ndarray) -> np.ndarray:
    """Cumulative postoperative hours since surgery (``process_data`` JSONL convention).

    ``time_hour`` is hours from the surgery anchor; ``time_day`` is days from anchor
    (typically ``time_hour / 24``). Do **not** use ``time_day * 24 + time_hour`` — that
    double-counts when both fields are cumulative.
    """
    th = np.asarray(time_hours, dtype=np.float64)
    td = np.asarray(time_days, dtype=np.float64)
    if th.shape != td.shape:
        raise ValueError("time_days and time_hours must have the same shape")
    out = th.copy()
    # Rare fallback when hour is missing/zero but day is set (legacy rows).
    fallback = (td > 0.0) & ((~np.isfinite(th)) | (np.abs(th) < 1e-12))
    out[fallback] = td[fallback] * 24.0
    return out


def evaluate_online(
    model,
    loader,
    device,
    day_bin_width: float = 1.0,
    cls_threshold: float = 0.5,
    postop_val_hour_bin_width: float = 6.0,
):
    arrs = collect_predictions(model, loader, device)
    probs, targets = arrs["probs"], arrs["targets"]
    time_days = arrs["time_days"]
    time_hours = arrs.get("time_hours", np.array([]))
    step_indices = arrs["step_indices"]
    final_probs, final_targets = arrs["final_probs"], arrs["final_targets"]
    base_probs, base_targets = arrs["base_probs"], arrs["base_targets"]
    metrics: Dict[str, Any] = {
        "cls_threshold": cls_threshold,
        "postop_val_hour_bin_width": float(postop_val_hour_bin_width),
        "baseline_node": _subset_classification_metrics(base_targets, base_probs, cls_threshold),
        "online_all_valid": _subset_classification_metrics(targets, probs, cls_threshold),
        "final_valid_node": _subset_classification_metrics(final_targets, final_probs, cls_threshold),
        "by_postop_day_bucket": {},
        "by_postop_hour_bucket": {},
        "by_valid_step_index": {},
    }
    if len(time_days) > 0:
        day_buckets = np.floor(time_days / day_bin_width).astype(int)
        for b in sorted(np.unique(day_buckets)):
            mask = day_buckets == b
            metrics["by_postop_day_bucket"][str(int(b))] = _subset_classification_metrics(
                targets[mask], probs[mask], cls_threshold
            )
    if (
        len(time_days) > 0
        and len(time_hours) == len(time_days)
        and float(postop_val_hour_bin_width) > 0
    ):
        postop_hours = postop_continuous_hours(time_days, time_hours)
        w = float(postop_val_hour_bin_width)
        hour_buckets = np.floor(postop_hours / w).astype(int)
        for b in sorted(np.unique(hour_buckets)):
            mask = hour_buckets == b
            metrics["by_postop_hour_bucket"][str(int(b))] = _subset_classification_metrics(
                targets[mask], probs[mask], cls_threshold
            )
    if len(step_indices) > 0:
        for k in sorted(np.unique(step_indices)):
            mask = step_indices == k
            metrics["by_valid_step_index"][str(int(k))] = _subset_classification_metrics(
                targets[mask], probs[mask], cls_threshold
            )
    return metrics


def macro_mean_postop_hour_bucket_auprc(val_metrics: Dict[str, Any], min_n: int = 1) -> Optional[float]:
    """Unweighted mean of AUPRC over ``by_postop_hour_bucket`` bins (each bin weighted equally).

    Skips bins with ``n < min_n`` or undefined ``auprc`` (e.g. single-class).
    """
    buckets = val_metrics.get("by_postop_hour_bucket") or {}
    vals: list[float] = []
    for _, b in buckets.items():
        if not isinstance(b, dict):
            continue
        if int(b.get("n", 0)) < int(min_n):
            continue
        a = b.get("auprc")
        if a is None:
            continue
        vals.append(float(a))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def macro_mean_postop_day_bucket_auprc(val_metrics: Dict[str, Any], min_n: int = 1) -> Optional[float]:
    """Unweighted mean of AUPRC over ``by_postop_day_bucket`` bins (each day-bin weighted equally).

    Skips bins with ``n < min_n`` or undefined ``auprc`` (e.g. single-class).
    """
    buckets = val_metrics.get("by_postop_day_bucket") or {}
    vals: list[float] = []
    for _, b in buckets.items():
        if not isinstance(b, dict):
            continue
        if int(b.get("n", 0)) < int(min_n):
            continue
        a = b.get("auprc")
        if a is None:
            continue
        vals.append(float(a))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def validation_combined_selection_score(
    val_metrics: Dict[str, Any],
    lam: float,
    macro_min_n: int = 1,
) -> Tuple[float, Optional[float], Optional[float]]:
    """Best-checkpoint score: ``lam * macro_online + (1-lam) * baseline_auprc``.

    ``macro_online`` is macro-mean AUPRC over **postoperative hour** buckets
    (``by_postop_hour_bucket``; bins use ``floor(time_hour / postop_val_hour_bin_width)`` with cumulative ``time_hour``).
    Returns ``(score, macro_online_or_none, baseline_auprc_or_none)``. ``score`` is -1.0 if undefined.
    """
    if lam < 0.0 or lam > 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")
    macro = macro_mean_postop_hour_bucket_auprc(val_metrics, min_n=macro_min_n)
    base = val_metrics["baseline_node"].get("auprc")
    base_f = float(base) if base is not None else None
    macro_f = float(macro) if macro is not None else None
    if lam >= 1.0 - 1e-12:
        return (-1.0 if macro_f is None else macro_f, macro_f, base_f)
    if lam <= 1e-12:
        return (-1.0 if base_f is None else base_f, macro_f, base_f)
    if macro_f is None or base_f is None:
        return (-1.0, macro_f, base_f)
    combined = float(lam) * macro_f + float(1.0 - lam) * base_f
    return (combined, macro_f, base_f)
