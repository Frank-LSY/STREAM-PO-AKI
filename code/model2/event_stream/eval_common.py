"""Shared checkpoint loading and online evaluation for model2 scripts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from event_stream.checkpoint import build_blockmask_model, infer_arch
from event_stream.dataset import PeriopEpisodeDataset, collate_fn, load_episode_jsonl
from common.med_exposure import infer_num_med_exposure
from common.utils import evaluate_online, resolve_torch_device


def load_checkpoint_bundle(checkpoint: str) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[Any], Optional[Any], int]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    code_vocab = ckpt.get("code_vocab")
    if code_vocab is None:
        raise KeyError("Checkpoint does not contain code_vocab.")
    train_args = ckpt.get("args", {}) or {}
    lab_stats = ckpt.get("lab_stats")
    age_stats = ckpt.get("age_stats")
    return ckpt, code_vocab, lab_stats, age_stats, int(train_args.get("num_med_exposure", 0))


def build_model_from_checkpoint(
    code_vocab: Dict[str, int],
    train_args: Dict[str, Any],
    num_med_exposure: int,
) -> nn.Module:
    arch = infer_arch(train_args)
    return build_blockmask_model(arch, len(code_vocab), train_args, num_med_exposure)


def build_eval_dataloader(
    episode_path: str,
    code_vocab: Dict[str, int],
    lab_stats: Optional[Any],
    num_med_exposure: int,
    batch_size: int,
    *,
    train_args: Optional[Dict[str, Any]] = None,
    age_stats: Optional[Any] = None,
) -> DataLoader:
    episodes = load_episode_jsonl(episode_path)
    if num_med_exposure < 0:
        num_med_exposure = infer_num_med_exposure(episodes)
    ta = train_args or {}
    # Honor training-time input ablations so test eval matches the trained model.
    dataset = PeriopEpisodeDataset(
        episodes,
        code_vocab,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med_exposure,
        baseline_token=str(ta.get("baseline_token", "dedicated")),
        drop_conditions=bool(ta.get("drop_conditions", False)),
        drop_lab_values=bool(ta.get("drop_lab_values", False)),
        drop_labs=bool(ta.get("drop_labs", False)),
        drop_meds=bool(ta.get("drop_meds", False)),
        generic_surgery=bool(ta.get("generic_surgery", False)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def run_online_evaluation(
    checkpoint: str,
    episode_path: str,
    device_str: str = "auto",
    batch_size: int = 8,
    day_bin_width: float = 1.0,
    postop_hour_bin: float = 6.0,
    cls_threshold: float = 0.5,
) -> Tuple[Dict[str, Any], Dict[str, Any], nn.Module]:
    """Load checkpoint, run ``evaluate_online``, return (metrics, train_args, model)."""
    if postop_hour_bin <= 0:
        raise ValueError("postop_hour_bin must be > 0.")
    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(checkpoint)
    train_args = ckpt.get("args", {}) or {}
    # Prefer the exact training setting (incl. 0 for I1); only infer when unset (-1).
    if num_med < 0:
        episodes = load_episode_jsonl(episode_path)
        num_med = infer_num_med_exposure(episodes)
    loader = build_eval_dataloader(
        episode_path,
        code_vocab,
        lab_stats,
        num_med,
        batch_size,
        train_args=train_args,
        age_stats=age_stats,
    )
    model = build_model_from_checkpoint(code_vocab, train_args, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(device_str)
    model.to(device)
    metrics = evaluate_online(
        model,
        loader,
        device,
        day_bin_width=day_bin_width,
        cls_threshold=cls_threshold,
        postop_val_hour_bin_width=float(postop_hour_bin),
    )
    return metrics, train_args, model
