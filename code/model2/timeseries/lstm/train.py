#!/usr/bin/env python3
"""Train 6h bucket segmented LSTM."""
from __future__ import annotations

import argparse

from timeseries.train_args import add_dataloader_args, add_wandb_args
from timeseries.train_loop import run_training
from common.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train 6h bucket LSTM.")
    p.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data/five_folds_ts6h4"),
    )
    p.add_argument("--val_fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--test_jsonl", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_step_size", type=int, default=5)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lambda_base", type=float, default=0.5)
    p.add_argument("--lambda_online", type=float, default=1.0)
    p.add_argument("--cls_pos_weight", type=float, default=1.0)
    p.add_argument("--base_pos_weight", type=float, default=1.0)
    p.add_argument("--postop_val_hour_bin", type=float, default=6.0)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--early_stop_patience", type=int, default=0)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)
    p.add_argument("--best_selection_lambda", type=float, default=0.5)
    p.add_argument("--best_macro_min_n", type=int, default=10)
    add_dataloader_args(p)
    add_wandb_args(p)
    return p.parse_args()


def main() -> None:
    run_training(parse_args(), arch="lstm")


if __name__ == "__main__":
    main()
