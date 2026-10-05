#!/usr/bin/env python3
"""Train event-stream segmented LSTM (MEM + baseline + online)."""
from __future__ import annotations

import argparse

from event_stream.train_loop import run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train perioperative event-stream LSTM.")
    p.add_argument("--episode_path", type=str, default=None)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--folds_dir", type=str, default=None)
    p.add_argument("--val_fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    p.add_argument("--fold_filename_template", type=str, default="7d_window_clm.mixed.fold{}.jsonl")
    p.add_argument("--test_jsonl", type=str, default=None)
    p.add_argument("--code_vocab_path", type=str, default=None)
    p.add_argument("--from_episodes_vocab", action="store_true")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_step_size", type=int, default=5)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_med_exposure", type=int, default=0)
    p.add_argument("--no_lab_zscore", action="store_true")
    p.add_argument("--mem_mask_ratio", type=float, default=0.15)
    p.add_argument("--lambda_mem", type=float, default=0.1)
    p.add_argument("--lambda_base", type=float, default=0.5)
    p.add_argument("--lambda_online", type=float, default=1.0)
    p.add_argument("--cls_pos_weight", type=float, default=1.0)
    p.add_argument("--base_pos_weight", type=float, default=1.0)
    p.add_argument("--day_bin_width", type=float, default=1.0)
    p.add_argument("--postop_val_hour_bin", type=float, default=6.0)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--best_selection_lambda", type=float, default=0.5)
    p.add_argument("--best_macro_min_n", type=int, default=10)
    p.add_argument("--eval_train_metrics", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    return p.parse_args()


def main() -> None:
    run_training(parse_args(), arch="lstm")


if __name__ == "__main__":
    main()
