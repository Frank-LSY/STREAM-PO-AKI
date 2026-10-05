"""Shared training loop for event-stream Transformer and LSTM."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, random_split

from event_stream.checkpoint import build_blockmask_model
from event_stream.dataset import PeriopEpisodeDataset, collate_fn, load_episode_jsonl, load_episode_jsonl_many
from event_stream.input_builder import MASK_TOKEN, build_code_vocab, load_code_vocab_json
from common.lab_zscore import fit_lab_zscore_from_episodes
from common.age_zscore import fit_age_zscore_from_episodes
from common.med_exposure import infer_num_med_exposure
from common.utils import (
    baseline_bce_loss,
    build_online_cls_targets,
    evaluate_online,
    make_mem_masked_inputs,
    masked_bce_with_logits_episode_normalized,
    masked_event_ce_loss,
    resolve_torch_device,
    save_json,
    validation_combined_selection_score,
)


def _resolve_fold_paths(folds_dir: str, template: str) -> Dict[int, str]:
    base = Path(folds_dir).resolve()
    paths: Dict[int, str] = {}
    for k in range(1, 6):
        fp = base / template.format(k)
        if not fp.is_file():
            raise FileNotFoundError(f"Missing fold file: {fp}")
        paths[k] = str(fp)
    return paths


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


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _load_splits(args) -> tuple:
    if args.folds_dir:
        if args.val_fold is None:
            raise ValueError("--val_fold is required with --folds_dir")
        paths = _resolve_fold_paths(args.folds_dir, args.fold_filename_template)
        train_paths = [paths[k] for k in range(1, 6) if k != args.val_fold]
        train_episodes = load_episode_jsonl_many(train_paths)
        val_episodes = load_episode_jsonl(paths[args.val_fold])
        return (
            train_episodes,
            val_episodes,
            {
                "mode": "5fold",
                "val_fold": args.val_fold,
                "train_paths": train_paths,
                "val_path": paths[args.val_fold],
            },
        )
    if not args.episode_path:
        raise ValueError("Provide --episode_path or --folds_dir")
    episodes = load_episode_jsonl(args.episode_path)
    n_total = len(episodes)
    n_train = int(0.8 * n_total)
    gen = torch.Generator().manual_seed(args.split_seed)
    indices = list(range(n_total))
    train_subset, val_subset = random_split(indices, [n_train, n_total - n_train], generator=gen)
    train_episodes = [episodes[i] for i in list(train_subset)]
    val_episodes = [episodes[i] for i in list(val_subset)]
    return (
        train_episodes,
        val_episodes,
        {
            "mode": "random80_20",
            "split_seed": args.split_seed,
            "n_train": len(train_episodes),
            "n_val": len(val_episodes),
        },
    )


def _batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _model_forward_kwargs(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    fwd = {
        "code_ids": batch["code_ids"],
        "modality_ids": batch["modality_ids"],
        "time_days": batch["time_days"],
        "time_hours": batch["time_hours"],
        "values": batch["values"],
        "value_masks": batch["value_masks"],
        "attention_mask": batch["attention_mask"],
        "base_mask": batch["base_mask"],
    }
    if "preop_mask" in batch:
        fwd["preop_mask"] = batch["preop_mask"]
    if "postop_mask" in batch:
        fwd["postop_mask"] = batch["postop_mask"]
    if "med_day_feats" in batch:
        fwd["med_day_feats"] = batch["med_day_feats"]
    if "event_type_ids" in batch:
        fwd["event_type_ids"] = batch["event_type_ids"]
    if "age_values" in batch:
        fwd["age_values"] = batch["age_values"]
    if "age_masks" in batch:
        fwd["age_masks"] = batch["age_masks"]
    return fwd


def run_training(args, arch: str = "transformer") -> None:
    arch = str(arch).lower()
    if args.lr_step_size < 1:
        raise ValueError("--lr_step_size must be >= 1 for StepLR.")
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if args.best_selection_lambda < 0.0 or args.best_selection_lambda > 1.0:
        raise ValueError("--best_selection_lambda must be in [0, 1].")
    if args.best_macro_min_n < 1:
        raise ValueError("--best_macro_min_n must be >= 1.")
    if args.postop_val_hour_bin <= 0:
        raise ValueError("--postop_val_hour_bin must be > 0.")

    os.makedirs(args.save_dir, exist_ok=True)
    args_dict = dict(vars(args))
    args_dict["arch"] = arch
    # Normalize ablation flags for checkpoint reload.
    args_dict["use_time_encoding"] = bool(int(getattr(args, "use_time_encoding", 1)))
    args_dict["attention_mode"] = str(getattr(args, "attention_mode", "block"))
    args_dict["baseline_token"] = str(getattr(args, "baseline_token", "dedicated"))
    args_dict["drop_lab_values"] = bool(getattr(args, "drop_lab_values", False))
    args_dict["drop_labs"] = bool(getattr(args, "drop_labs", False))
    args_dict["drop_meds"] = bool(getattr(args, "drop_meds", False))
    args_dict["drop_conditions"] = bool(getattr(args, "drop_conditions", False))
    args_dict["generic_surgery"] = bool(getattr(args, "generic_surgery", False))
    save_json(args_dict, os.path.join(args.save_dir, "args.json"))
    history_path = os.path.join(args.save_dir, "metrics_history.jsonl")

    wandb_run: Optional[Any] = None
    wandb_mod = None
    if getattr(args, "wandb_project", None):
        from common.wandb_utils import finish_wandb, init_wandb_run, log_wandb_metrics, update_wandb_summary

        tags = None
        if args.wandb_tags:
            tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        wandb_run, wandb_mod = init_wandb_run(
            project=args.wandb_project,
            entity=getattr(args, "wandb_entity", None),
            name=args.wandb_run_name or f"blockmask_{arch}_fold{args.val_fold}",
            tags=tags,
            config={k: v for k, v in args_dict.items() if k != "wandb_tags"},
            dir=Path(args.save_dir),
        )

    train_episodes, val_episodes, split_info = _load_splits(args)
    split_info["test_jsonl"] = args.test_jsonl
    split_info["arch"] = arch
    save_json(split_info, os.path.join(args.save_dir, "split_info.json"))
    if args.folds_dir:
        fold_split = {
            "mode": "5fold",
            "val_fold": int(args.val_fold),
            "train_folds": [k for k in range(1, 6) if k != args.val_fold],
            "train_paths": split_info["train_paths"],
            "val_path": split_info["val_path"],
            "test_jsonl": args.test_jsonl,
            "arch": arch,
        }
        save_json(fold_split, os.path.join(args.save_dir, "fold_split.json"))

    if args.code_vocab_path and not args.from_episodes_vocab:
        code_vocab = load_code_vocab_json(args.code_vocab_path)
    else:
        code_vocab = build_code_vocab(train_episodes, min_freq=1)
    save_json(code_vocab, os.path.join(args.save_dir, "code_vocab.json"))

    num_med = args.num_med_exposure
    if num_med < 0:
        num_med = infer_num_med_exposure(train_episodes + val_episodes)
    args.num_med_exposure = int(num_med)

    if args.no_lab_zscore:
        lab_stats = None
    else:
        lab_stats = fit_lab_zscore_from_episodes(train_episodes, range(len(train_episodes)))
        save_json(lab_stats, os.path.join(args.save_dir, "lab_zscore_stats.json"))

    if bool(getattr(args, "no_age_zscore", False)):
        age_stats = None
    else:
        age_stats = fit_age_zscore_from_episodes(train_episodes, range(len(train_episodes)))
        save_json(age_stats, os.path.join(args.save_dir, "age_zscore_stats.json"))

    train_dataset = PeriopEpisodeDataset(
        train_episodes,
        code_vocab,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med,
        baseline_token=getattr(args, "baseline_token", "dedicated"),
        drop_conditions=bool(getattr(args, "drop_conditions", False)),
        drop_lab_values=bool(getattr(args, "drop_lab_values", False)),
        drop_labs=bool(getattr(args, "drop_labs", False)),
        drop_meds=bool(getattr(args, "drop_meds", False)),
        generic_surgery=bool(getattr(args, "generic_surgery", False)),
    )
    val_dataset = PeriopEpisodeDataset(
        val_episodes,
        code_vocab,
        lab_stats=lab_stats,
        age_stats=age_stats,
        num_med_exposure=num_med,
        baseline_token=getattr(args, "baseline_token", "dedicated"),
        drop_conditions=bool(getattr(args, "drop_conditions", False)),
        drop_lab_values=bool(getattr(args, "drop_lab_values", False)),
        drop_labs=bool(getattr(args, "drop_labs", False)),
        drop_meds=bool(getattr(args, "drop_meds", False)),
        generic_surgery=bool(getattr(args, "generic_surgery", False)),
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    train_eval_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = resolve_torch_device(args.device)
    print(f"Using device: {device}; arch={arch}; num_med_exposure={num_med}; vocab_size={len(code_vocab)}")
    model = build_blockmask_model(arch, len(code_vocab), args_dict, num_med).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    mask_token_id = code_vocab[MASK_TOKEN]
    best_score = -1.0
    epochs_no_improve = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = total_online = total_base = total_mem = 0.0
            n_batches = 0
            accum = 0
            for batch in train_loader:
                batch = _batch_to_device(batch, device)
                outputs = model(**_model_forward_kwargs(batch))
                online_targets, online_mask = build_online_cls_targets(batch)
                online_loss = masked_bce_with_logits_episode_normalized(
                    outputs["token_logits"], online_targets, online_mask, pos_weight=args.cls_pos_weight
                )
                base_loss = baseline_bce_loss(
                    outputs["baseline_logits_all"],
                    batch["labels"],
                    batch["base_mask"],
                    pos_weight=args.base_pos_weight,
                )

                if float(args.lambda_mem) > 0.0:
                    mem_batch = make_mem_masked_inputs(
                        batch, mask_token_id=mask_token_id, mask_ratio=args.mem_mask_ratio
                    )
                    mem_outputs = model(**_model_forward_kwargs(mem_batch))
                    mem_loss = masked_event_ce_loss(
                        mem_outputs["mem_logits"], mem_batch["mem_targets"], mem_batch["mem_mask"]
                    )
                else:
                    mem_loss = online_loss.new_zeros(())

                loss = args.lambda_online * online_loss + args.lambda_base * base_loss + args.lambda_mem * mem_loss
                if not torch.isfinite(loss):
                    continue
                (loss / float(args.grad_accum_steps)).backward()
                accum += 1
                if accum >= args.grad_accum_steps:
                    if args.grad_clip_norm and args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum = 0
                total_loss += float(loss.item())
                total_online += float(online_loss.item())
                total_base += float(base_loss.item())
                total_mem += float(mem_loss.item())
                n_batches += 1

            if accum > 0:
                if args.grad_clip_norm and args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            denom = max(n_batches, 1)
            val_metrics = evaluate_online(
                model,
                val_loader,
                device,
                day_bin_width=args.day_bin_width,
                cls_threshold=args.cls_threshold,
                postop_val_hour_bin_width=args.postop_val_hour_bin,
            )
            score, sel_macro, sel_base = validation_combined_selection_score(
                val_metrics,
                lam=args.best_selection_lambda,
                macro_min_n=args.best_macro_min_n,
            )
            current_lr = optimizer.param_groups[0]["lr"]
            row: Dict[str, Any] = {
                "epoch": epoch + 1,
                "lr": float(current_lr),
                "train": {
                    "loss": total_loss / denom,
                    "online_loss": total_online / denom,
                    "baseline_loss": total_base / denom,
                    "mem_loss": total_mem / denom,
                },
                "val": val_metrics,
                "selection": {
                    "lambda": float(args.best_selection_lambda),
                    "macro_online_hour_bucket_auprc": sel_macro,
                    "postop_val_hour_bin": float(args.postop_val_hour_bin),
                    "baseline_auprc": sel_base,
                    "combined_score": float(score),
                },
            }
            if args.eval_train_metrics:
                row["train_eval"] = evaluate_online(
                    model,
                    train_eval_loader,
                    device,
                    day_bin_width=args.day_bin_width,
                    cls_threshold=args.cls_threshold,
                    postop_val_hour_bin_width=args.postop_val_hour_bin,
                )
            _append_jsonl(history_path, row)
            save_json(val_metrics, os.path.join(args.save_dir, f"val_metrics_epoch_{epoch+1}.json"))

            if wandb_run is not None:
                log_dict: Dict[str, Any] = {
                    "epoch": epoch + 1,
                    "train/lr": float(current_lr),
                    "train/loss": total_loss / denom,
                    "train/online_loss": total_online / denom,
                    "train/baseline_loss": total_base / denom,
                    "train/mem_loss": total_mem / denom,
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
                if args.eval_train_metrics and row.get("train_eval") is not None:
                    flat_te: Dict[str, float] = {}
                    _flatten_metrics("train_eval", row["train_eval"], flat_te)
                    log_dict.update(flat_te)
                log_wandb_metrics(wandb_mod, wandb_run, log_dict, step=epoch + 1)

            fv = val_metrics["final_valid_node"]
            bv = val_metrics["baseline_node"]
            print(
                f"Epoch {epoch+1}: lr={current_lr:.2e} loss={row['train']['loss']:.4f} "
                f"online={row['train']['online_loss']:.4f} base={row['train']['baseline_loss']:.4f} "
                f"mem={row['train']['mem_loss']:.4f} val_final_auprc={fv['auprc']} val_base_auprc={bv['auprc']} "
                f"sel_score={score:.4f} (λ={args.best_selection_lambda} macro={sel_macro} base={sel_base})"
            )

            improved = score > best_score + args.early_stop_min_delta
            if improved:
                best_score = score
                epochs_no_improve = 0
                best_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "code_vocab": code_vocab,
                        "lab_stats": lab_stats,
                        "age_stats": age_stats,
                        "args": args_dict,
                    },
                    best_path,
                )
                save_json(val_metrics, os.path.join(args.save_dir, "best_metrics.json"))
                if wandb_run is not None:
                    update_wandb_summary(
                        wandb_mod,
                        wandb_run,
                        {
                            "best_selection_combined_score": float(score),
                            "best_epoch": epoch + 1,
                        },
                    )
            elif args.early_stop_patience > 0:
                epochs_no_improve += 1

            scheduler.step()

            if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                save_json(
                    {
                        "stopped_at_epoch": epoch + 1,
                        "best_score": best_score,
                        "arch": arch,
                    },
                    os.path.join(args.save_dir, "early_stop.json"),
                )
                print("Early stopping.")
                break
    finally:
        finish_wandb(wandb_mod, wandb_run)
