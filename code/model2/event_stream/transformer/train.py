from __future__ import annotations

import argparse

from event_stream.train_loop import run_training


def parse_args():
    p = argparse.ArgumentParser(
        description="Train perioperative block-mask Transformer with MEM + baseline + online losses."
    )
    p.add_argument("--episode_path", type=str, default=None)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--folds_dir", type=str, default=None)
    p.add_argument("--val_fold", type=int, default=None, choices=[1, 2, 3, 4, 5])
    p.add_argument("--fold_filename_template", type=str, default="7d_window_clm.mixed.fold{}.jsonl")
    p.add_argument(
        "--test_jsonl",
        type=str,
        default=None,
        help="Held-out test JSONL path (recorded in metadata only; not loaded during training).",
    )
    p.add_argument("--code_vocab_path", type=str, default=None)
    p.add_argument("--from_episodes_vocab", action="store_true", help="Build vocab from training episodes.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--lr_step_size",
        type=int,
        default=5,
        help="StepLR: decay LR every this many epochs (must be >= 1).",
    )
    p.add_argument(
        "--lr_gamma",
        type=float,
        default=0.1,
        help="StepLR: new_lr = lr * gamma at each step (use 1.0 for no decay).",
    )
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--ffn_dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num_med_exposure", type=int, default=0, help="Daily med vector dim; 0=off (default), -1=auto.")
    p.add_argument("--no_lab_zscore", action="store_true")
    p.add_argument("--no_age_zscore", action="store_true", help="Pass raw age into age_mlp without train-set z-score.")
    p.add_argument(
        "--use_time_encoding",
        type=int,
        default=1,
        choices=[0, 1],
        help="1=add continuous time MLP on hour/24 (default); 0=ablate time encoding.",
    )
    p.add_argument(
        "--attention_mode",
        type=str,
        default="block",
        choices=["block", "causal"],
        help="block=preop bidirectional + postop causal; causal=full causal-only.",
    )
    p.add_argument(
        "--baseline_token",
        type=str,
        default="dedicated",
        choices=["dedicated", "procedure"],
        help="dedicated=[BASE] token; procedure=use index-surgery token as baseline summary.",
    )
    p.add_argument(
        "--drop_lab_values",
        action="store_true",
        help="Zero laboratory numeric values; keep lab identity codes (legacy partial ablation).",
    )
    p.add_argument(
        "--drop_labs",
        action="store_true",
        help="Remove all laboratory-test event tokens from the sequence.",
    )
    p.add_argument(
        "--drop_meds",
        action="store_true",
        help="Remove all medication event tokens (start/end) from the sequence.",
    )
    p.add_argument(
        "--drop_conditions",
        action="store_true",
        help="Remove condition-event tokens from the sequence.",
    )
    p.add_argument(
        "--generic_surgery",
        action="store_true",
        help="Replace procedure SNOMED with [SURGERY]; keep t0 anchor.",
    )
    p.add_argument("--mem_mask_ratio", type=float, default=0.15)
    p.add_argument("--lambda_mem", type=float, default=0.1)
    p.add_argument("--lambda_base", type=float, default=0.5)
    p.add_argument("--lambda_online", type=float, default=1.0)
    p.add_argument("--cls_pos_weight", type=float, default=1.0)
    p.add_argument("--base_pos_weight", type=float, default=1.0)
    p.add_argument("--day_bin_width", type=float, default=1.0)
    p.add_argument(
        "--postop_val_hour_bin",
        type=float,
        default=6.0,
        help="Postoperative validation hour bucket width for by_postop_hour_bucket and selection macro.",
    )
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


def main():
    run_training(parse_args(), arch="transformer")


if __name__ == "__main__":
    main()
