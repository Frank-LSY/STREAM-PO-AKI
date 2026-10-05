"""Shared training loop for 6h bucket sequence models."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from timeseries.checkpoint import build_bucket_model
from timeseries.dataset import Ts6hEpisodeDataset, batch_to_device, load_feature_layout
from common.utils import (
    baseline_bce_loss,
    masked_bce_with_logits_episode_normalized,
    resolve_torch_device,
    save_json,
    validation_combined_selection_score,
)
from timeseries.utils_eval import evaluate_bucket_online, forward_bucket_model


def build_online_targets(batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels"]
    targets = labels.unsqueeze(1).expand_as(batch["online_mask"])
    return targets.float(), batch["online_mask"].float()


def _flatten_metrics(prefix: str, obj: Any, out: Dict[str, float]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}/{k}" if prefix else str(k)
            _flatten_metrics(key, v, out)
    elif isinstance(obj, (bool, str)) or obj is None:
        return
    else:
        try:
            out[prefix] = float(obj)
        except (TypeError, ValueError):
            return


def run_training(args: Namespace, arch: str) -> None:
    data_root = Path(args.data_root).resolve()
    save_dir = Path(args.save_dir).resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    arg_dict = vars(args).copy()
    arg_dict["arch"] = arch
    save_json(arg_dict, save_dir / "args.json")

    wandb_run: Optional[Any] = None
    wandb_mod = None
    if getattr(args, "wandb_project", None):
        from common.wandb_utils import finish_wandb, init_wandb_run, log_wandb_metrics, update_wandb_summary

        tags = None
        if getattr(args, "wandb_tags", None):
            tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
        wandb_run, wandb_mod = init_wandb_run(
            project=args.wandb_project,
            entity=getattr(args, "wandb_entity", None),
            name=args.wandb_run_name or f"ts6h_{arch}_fold{args.val_fold}",
            tags=tags,
            config={k: v for k, v in arg_dict.items() if k != "wandb_tags"},
            dir=save_dir,
        )

    layout = load_feature_layout(data_root)
    n_features = int(layout["n_features"])
    split_dir = data_root / "splits" / f"val_fold_{args.val_fold}"

    device = resolve_torch_device(args.device)
    pin_memory = bool(getattr(args, "pin_memory", True)) and device.type == "cuda"

    load_obs = arch == "grud" and str(arg_dict.get("grud_obs_mode", "legacy")) == "raw"
    train_ds = Ts6hEpisodeDataset(split_dir / "train.npz", load_obs=load_obs)
    val_ds = Ts6hEpisodeDataset(split_dir / "val.npz", load_obs=load_obs)
    num_workers = int(getattr(args, "num_workers", 0))
    loader_kw: Dict[str, Any] = {
        "batch_size": args.batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4))
    train_loader = DataLoader(train_ds, shuffle=True, collate_fn=train_ds.collate_fn, **loader_kw)
    val_loader = DataLoader(val_ds, shuffle=False, collate_fn=val_ds.collate_fn, **loader_kw)

    fold_split = {
        "mode": "5fold_ts6h",
        "arch": arch,
        "val_fold": int(args.val_fold),
        "data_root": str(data_root),
        "train_npz": str(split_dir / "train.npz"),
        "val_npz": str(split_dir / "val.npz"),
        "test_npz": str(data_root / "test.npz"),
        "test_jsonl": args.test_jsonl,
    }
    save_json(fold_split, save_dir / "fold_split.json")
    save_json(layout, save_dir / "feature_layout.json")

    model = build_bucket_model(arch, n_features, arg_dict, layout=layout).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
    )

    best_score = -1.0
    epochs_no_improve = 0
    history_path = save_dir / "metrics_history.jsonl"

    try:
        for epoch in range(args.epochs):
            model.train()
            total_loss = total_online = total_base = 0.0
            n_batches = 0
            for batch in train_loader:
                batch = batch_to_device(batch, device, non_blocking=pin_memory)
                out = forward_bucket_model(model, batch)
                online_targets, online_mask = build_online_targets(batch)
                online_loss = masked_bce_with_logits_episode_normalized(
                    out["token_logits"],
                    online_targets,
                    online_mask,
                    pos_weight=args.cls_pos_weight,
                )
                base_loss = baseline_bce_loss(
                    out["baseline_logits_all"],
                    batch["labels"],
                    batch["base_mask"],
                    pos_weight=args.base_pos_weight,
                )
                loss = args.lambda_online * online_loss + args.lambda_base * base_loss
                if not torch.isfinite(loss):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                total_loss += float(loss.item())
                total_online += float(online_loss.item())
                total_base += float(base_loss.item())
                n_batches += 1

            val_metrics = evaluate_bucket_online(
                model,
                val_loader,
                device,
                cls_threshold=args.cls_threshold,
                postop_val_hour_bin_width=args.postop_val_hour_bin,
            )
            score, sel_macro, sel_base = validation_combined_selection_score(
                val_metrics,
                lam=args.best_selection_lambda,
                macro_min_n=args.best_macro_min_n,
            )
            scheduler.step()
            denom = max(n_batches, 1)
            current_lr = float(optimizer.param_groups[0]["lr"])
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "lr": current_lr,
                "train": {
                    "loss": total_loss / denom,
                    "online_loss": total_online / denom,
                    "baseline_loss": total_base / denom,
                },
                "val": val_metrics,
                "selection": {
                    "combined_score": score,
                    "macro_online_hour_bucket_auprc": sel_macro,
                    "baseline_auprc": sel_base,
                },
            }
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            save_json(val_metrics, save_dir / f"val_metrics_epoch_{epoch+1}.json")

            if wandb_run is not None:
                log_dict: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train/lr": current_lr,
                    "train/loss": total_loss / denom,
                    "train/online_loss": total_online / denom,
                    "train/baseline_loss": total_base / denom,
                }
                flat_val: Dict[str, float] = {}
                _flatten_metrics("val", val_metrics, flat_val)
                log_dict.update(flat_val)
                log_dict["val/selection_combined_score"] = float(score)
                if sel_macro is not None:
                    log_dict["val/selection_macro_online_hour_bucket_auprc"] = sel_macro
                if sel_base is not None:
                    log_dict["val/selection_baseline_auprc"] = sel_base
                log_dict["val/best_selection_lambda"] = float(args.best_selection_lambda)
                log_wandb_metrics(wandb_mod, wandb_run, log_dict, step=epoch + 1)

            improved = score > best_score + args.early_stop_min_delta
            if improved:
                best_score = score
                epochs_no_improve = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "args": arg_dict,
                        "feature_layout": layout,
                        "n_features": n_features,
                        "epoch": epoch + 1,
                        "selection_score": score,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"epoch {epoch+1}: saved best_model.pt score={score:.4f}")
                if wandb_run is not None:
                    update_wandb_summary(
                        wandb_mod,
                        wandb_run,
                        {
                            "best_selection_combined_score": float(score),
                            "best_epoch": epoch + 1,
                        },
                    )
            else:
                epochs_no_improve += 1
                print(f"epoch {epoch+1}: score={score:.4f} (best={best_score:.4f})")

            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                print("Early stopping.")
                break

        print(f"Done. best_score={best_score:.4f}")
    finally:
        if getattr(args, "wandb_project", None):
            from common.wandb_utils import finish_wandb

            finish_wandb(wandb_mod, wandb_run)
