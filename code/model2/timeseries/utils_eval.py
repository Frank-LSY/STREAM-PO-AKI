"""Evaluation helpers for 6h bucket sequence models (Transformer / LSTM)."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch

from timeseries.dataset import batch_to_device
from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, BUCKET_SIZE_H
from common.utils import _subset_classification_metrics


def model_uses_obs_mask(model) -> bool:
    return bool(getattr(model, "uses_obs_mask", False))


def forward_bucket_model(model, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    extra = {"obs_mask": batch["obs_mask"]} if model_uses_obs_mask(model) else {}
    return model(
        bucket_features=batch["bucket_features"],
        attention_mask=batch["attention_mask"],
        base_mask=batch["base_mask"],
        **extra,
    )


_forward_bucket = forward_bucket_model


def collect_bucket_predictions(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    all_probs, all_targets, all_bucket_hours = [], [], []
    final_probs, final_targets = [], []
    base_probs, base_targets = [], []

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, device, non_blocking=device.type == "cuda")
            out = _forward_bucket(model, batch)
            token_probs = torch.sigmoid(out["token_logits"])
            online_mask = batch["online_mask"] > 0
            base_bool = batch["base_mask"] > 0

            if base_bool.any().item():
                bp = torch.sigmoid(out["baseline_logits_all"])[base_bool]
                bt = batch["labels"].unsqueeze(1).expand_as(batch["base_mask"])[base_bool]
                base_probs.extend(bp.cpu().numpy().tolist())
                base_targets.extend(bt.cpu().numpy().tolist())

            B, L = token_probs.shape
            for i in range(B):
                valid = online_mask[i] > 0
                if valid.sum().item() == 0:
                    continue
                probs_i = token_probs[i][valid].cpu().numpy()
                n_valid = int(valid.sum().item())
                targets_i = np.full(n_valid, float(batch["labels"][i].item()), dtype=np.float64)
                idxs = torch.nonzero(valid, as_tuple=False).flatten().cpu().numpy()
                hours = []
                for idx in idxs:
                    j = int(idx) - BASE_BUCKET_IDX - 1
                    hours.append(float(j * BUCKET_SIZE_H))
                all_probs.append(probs_i)
                all_targets.append(targets_i)
                all_bucket_hours.append(np.asarray(hours, dtype=np.float64))
                final_probs.append(float(probs_i[-1]))
                final_targets.append(float(batch["labels"][i].item()))

    return {
        "probs": np.concatenate(all_probs) if all_probs else np.array([]),
        "targets": np.concatenate(all_targets) if all_targets else np.array([]),
        "bucket_hours": np.concatenate(all_bucket_hours) if all_bucket_hours else np.array([]),
        "final_probs": np.asarray(final_probs),
        "final_targets": np.asarray(final_targets),
        "base_probs": np.asarray(base_probs),
        "base_targets": np.asarray(base_targets),
    }


def evaluate_bucket_online(
    model,
    loader,
    device,
    cls_threshold: float = 0.5,
    postop_val_hour_bin_width: float = 6.0,
) -> Dict[str, Any]:
    arrs = collect_bucket_predictions(model, loader, device)
    probs, targets = arrs["probs"], arrs["targets"]
    bucket_hours = arrs["bucket_hours"]
    metrics: Dict[str, Any] = {
        "cls_threshold": cls_threshold,
        "postop_val_hour_bin_width": float(postop_val_hour_bin_width),
        "baseline_node": _subset_classification_metrics(arrs["base_targets"], arrs["base_probs"], cls_threshold),
        "online_all_valid": _subset_classification_metrics(targets, probs, cls_threshold),
        "final_valid_node": _subset_classification_metrics(
            arrs["final_targets"], arrs["final_probs"], cls_threshold
        ),
        "by_postop_hour_bucket": {},
    }
    if len(bucket_hours) > 0 and float(postop_val_hour_bin_width) > 0:
        w = float(postop_val_hour_bin_width)
        hour_buckets = np.floor(bucket_hours / w).astype(int)
        for b in sorted(np.unique(hour_buckets)):
            mask = hour_buckets == b
            metrics["by_postop_hour_bucket"][str(int(b))] = _subset_classification_metrics(
                targets[mask], probs[mask], cls_threshold
            )
    return metrics
