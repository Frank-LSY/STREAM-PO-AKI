from __future__ import annotations
import argparse

from event_stream.eval_common import run_online_evaluation
from common.utils import save_json


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate block-mask perioperative model.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_json", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--day_bin_width", type=float, default=1.0)
    p.add_argument(
        "--postop_val_hour_bin",
        type=float,
        default=6.0,
        help="Hour bucket width for by_postop_hour_bucket (same as training validation). Must be > 0.",
    )
    p.add_argument("--cls_threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    metrics, _, _ = run_online_evaluation(
        checkpoint=args.checkpoint,
        episode_path=args.episode_path,
        device_str=args.device,
        batch_size=args.batch_size,
        day_bin_width=args.day_bin_width,
        postop_hour_bin=args.postop_val_hour_bin,
        cls_threshold=args.cls_threshold,
    )
    save_json(metrics, args.output_json)
    print(f"Saved metrics to {args.output_json}")


if __name__ == "__main__":
    main()
