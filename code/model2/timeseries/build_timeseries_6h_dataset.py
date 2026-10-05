#!/usr/bin/env python3
"""Build fixed 6h grid tensors from five_folds JSONL (see ``common.fold_data.default_data_dir``)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from timeseries.timeseries_6h_common import (
    BASE_BUCKET_IDX,
    BUCKET_SIZE_H,
    NUM_POSTOP_BUCKETS,
    NUM_PREOP_BUCKETS,
    POSTOP_DAYS,
    PREOP_DAYS,
    SEQ_LEN,
    build_feature_layout,
    collect_vocab,
    episode_to_bucket_features,
    fit_lab_zscore_all_labs,
    max_postop_bucket_index,
    save_json,
)

from common.fold_data import default_data_dir, default_ts6h_data_dir, load_jsonl, resolve_fold_paths


def episodes_to_npz(
    episodes: Sequence[Dict[str, Any]],
    layout: Dict[str, Any],
    lab_stats: Dict[str, Any],
) -> Dict[str, Any]:
    n = len(episodes)
    nf = layout["n_features"]
    features = np.zeros((n, SEQ_LEN, nf), dtype=np.float16)
    labels = np.zeros(n, dtype=np.float32)
    person_ids: List[str] = []
    max_postop_bucket = np.zeros(n, dtype=np.int16)

    for i, ep in enumerate(episodes):
        features[i] = episode_to_bucket_features(ep, layout, lab_stats).astype(np.float16)
        labels[i] = float(ep.get("label_aki", 0))
        person_ids.append(str(ep.get("person_id", "")))
        max_postop_bucket[i] = int(max_postop_bucket_index(ep))

    return {
        "features": features,
        "labels": labels,
        "person_ids": np.asarray(person_ids, dtype=object),
        "max_postop_bucket": max_postop_bucket,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build five_folds_ts6h NPZ splits.")
    p.add_argument(
        "--folds_dir",
        type=str,
        default=None,
        help="JSONL folds root (default: DATA_DIR / common.fold_data.default_data_dir).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="NPZ output root (default: TS6H_DATA_DIR / five_folds_ts6h).",
    )
    p.add_argument("--fold_filename_template", type=str, default="7d_window_clm.mixed.fold{}.jsonl")
    p.add_argument("--test_filename", type=str, default="7d_window_clm.mixed.test.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    folds_dir = Path(args.folds_dir or default_data_dir())
    output_dir = Path(args.output_dir or default_ts6h_data_dir())
    fold_paths = resolve_fold_paths(folds_dir, args.fold_filename_template)
    test_path = folds_dir / args.test_filename

    print("Loading folds 1-5 for vocab / lab z-score...")
    all_eps: List[Dict[str, Any]] = []
    fold_eps: Dict[int, List[Dict[str, Any]]] = {}
    for k in range(1, 6):
        eps = load_jsonl(fold_paths[k])
        fold_eps[k] = eps
        all_eps.extend(eps)
        print(f"  fold{k}: {len(eps)}")

    vocab = collect_vocab(all_eps)
    lab_stats = fit_lab_zscore_all_labs(all_eps)
    layout = build_feature_layout(vocab)

    vocab_dir = output_dir / "vocab"
    save_json(vocab_dir / "vocab.json", vocab)
    save_json(vocab_dir / "lab_zscore.json", lab_stats)
    save_json(vocab_dir / "feature_layout.json", layout)
    save_json(
        output_dir / "grid_spec.json",
        {
            "bucket_size_hours": BUCKET_SIZE_H,
            "preop_days": PREOP_DAYS,
            "postop_days": POSTOP_DAYS,
            "num_preop_buckets": NUM_PREOP_BUCKETS,
            "num_postop_buckets": NUM_POSTOP_BUCKETS,
            "base_bucket_idx": BASE_BUCKET_IDX,
            "seq_len": SEQ_LEN,
            "n_features": layout["n_features"],
        },
    )

    splits_meta: Dict[str, Any] = {}
    for val_fold in range(1, 6):
        train_eps = []
        for k in range(1, 6):
            if k != val_fold:
                train_eps.extend(fold_eps[k])
        val_eps = fold_eps[val_fold]
        split_dir = output_dir / "splits" / f"val_fold_{val_fold}"
        split_dir.mkdir(parents=True, exist_ok=True)

        train_npz = episodes_to_npz(train_eps, layout, lab_stats)
        val_npz = episodes_to_npz(val_eps, layout, lab_stats)
        # NOTE: np.savez is a ZIP archive — NumPy cannot memmap members. Training
        # materializes features in RAM. Prefer bare .npy if mmap/workers are needed.
        np.savez(split_dir / "train.npz", **train_npz)
        np.savez(split_dir / "val.npz", **val_npz)
        print(f"val_fold_{val_fold}: train={len(train_eps)} val={len(val_eps)}")

        train_folds = [k for k in range(1, 6) if k != val_fold]
        splits_meta[f"val_fold_{val_fold}"] = {
            "val_fold": val_fold,
            "train_folds": train_folds,
            "train_rows": len(train_eps),
            "val_rows": len(val_eps),
        }

    test_eps = load_jsonl(test_path)
    test_npz = episodes_to_npz(test_eps, layout, lab_stats)
    np.savez(output_dir / "test.npz", **test_npz)
    print(f"test: {len(test_eps)}")

    manifest = {
        "source_dir": str(folds_dir),
        "output_dir": str(output_dir),
        "protocol": (
            "6h bucket concat features; BASE at index 28; "
            "LOCF across buckets for CCS, labs, and medications; "
            "block-mask Transformer/LSTM ready"
        ),
        "n_features": layout["n_features"],
        "seq_len": SEQ_LEN,
        "splits": splits_meta,
    }
    save_json(output_dir / "manifest.json", manifest)
    save_json(output_dir / "splits.json", splits_meta)
    print(f"Done -> {output_dir}")


if __name__ == "__main__":
    main()
