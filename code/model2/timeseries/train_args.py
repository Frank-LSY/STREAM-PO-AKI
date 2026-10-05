"""Shared training CLI helpers for bucket_ts6h."""
from __future__ import annotations

import argparse


def add_dataloader_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="DataLoader worker processes (0 = main process only).",
    )
    p.add_argument(
        "--pin_memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CPU tensors before async H2D copy (CUDA only).",
    )
    p.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="Batches prefetched per worker when num_workers > 0.",
    )


def add_wandb_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
