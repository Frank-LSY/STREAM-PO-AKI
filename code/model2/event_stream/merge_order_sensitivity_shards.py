#!/usr/bin/env python3
"""Merge multi-GPU order-sensitivity shard CSVs into the canonical outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from event_stream.evaluate_within_bin_order_sensitivity import (
    OUT_DIR,
    _summarize,
    plot_es_hist,
    write_report,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard_dir", type=str, default=str(OUT_DIR / "shards_R20"))
    p.add_argument("--out_dir", type=str, default=str(OUT_DIR))
    p.add_argument("--n_random", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    shard_dir = Path(args.shard_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    es_files = sorted(shard_dir.glob("order_sensitivity_es_rows_*.csv"))
    ts_files = sorted(shard_dir.glob("order_sensitivity_ts_rows_*.csv"))
    if not es_files:
        raise SystemExit(f"no shard ES CSVs in {shard_dir}")

    es_df = pd.concat([pd.read_csv(p) for p in es_files], ignore_index=True)
    ts_df = (
        pd.concat([pd.read_csv(p) for p in ts_files], ignore_index=True)
        if ts_files
        else pd.DataFrame()
    )
    # Deduplicate windows if shards overlapped (should not)
    if len(es_df):
        es_df = es_df.drop_duplicates(
            subset=["person_id", "bin_j", "perm"], keep="first"
        )
    if len(ts_df):
        ts_df = ts_df.drop_duplicates(
            subset=["person_id", "bin_j", "perm"], keep="first"
        )

    es_path = out_dir / "order_sensitivity_es_rows.csv"
    ts_path = out_dir / "order_sensitivity_ts_rows.csv"
    es_df.to_csv(es_path, index=False)
    ts_df.to_csv(ts_path, index=False)

    abs_dz = es_df["abs_delta_logit"].to_numpy(dtype=float) if len(es_df) else np.array([])
    es_sum = _summarize(abs_dz)
    if len(abs_dz):
        plot_es_hist(abs_dz, out_dir / "order_sensitivity_es_hist.png")

    n_windows = (
        int(es_df[["person_id", "bin_j"]].drop_duplicates().shape[0]) if len(es_df) else 0
    )
    n_episodes = int(es_df["person_id"].nunique()) if len(es_df) else 0

    ts_sum = {}
    if len(ts_df):
        ts_sum = {
            "n": int(len(ts_df)),
            "max_feat_l2": float(ts_df["feat_l2"].max()),
            "mean_abs_dz": float(ts_df["abs_delta_logit"].mean()),
            "max_abs_dz": float(ts_df["abs_delta_logit"].max()),
        }

    summary = {
        "n_episodes": n_episodes,
        "n_windows": n_windows,
        "n_random": args.n_random,
        "n_shards": len(es_files),
        "event_stream": es_sum,
        "timeseries": ts_sum,
        "paths": {"es_rows": str(es_path), "ts_rows": str(ts_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    class _Args:
        max_episodes = 0
        max_windows = 0
        seed = 0
        n_random = args.n_random
        min_k = 3
        min_distinct = 2

    write_report(
        out_dir / "REPORT.md",
        smoke=False,
        n_episodes=n_episodes,
        n_windows=n_windows,
        es_sum=es_sum,
        ts_sum=ts_sum,
        args=_Args(),
    )
    print(json.dumps(summary, indent=2))
    print(f"merged {len(es_files)} shards -> {es_path}", flush=True)


if __name__ == "__main__":
    main()
