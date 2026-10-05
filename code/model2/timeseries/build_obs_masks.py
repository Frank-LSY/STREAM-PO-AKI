#!/usr/bin/env python3
"""Add pre-LOCF observation masks (CCS / lab values / medications) next to existing ts6h NPZ splits.

Writes ``<split>.obs.npy``: uint8 [N, SEQ_LEN, ceil(n_obs/8)], bit-packed along the last axis
(``np.unpackbits(..., axis=-1, count=n_obs)``). Column order follows ``obs_feature_indices``.
Episode order matches ``build_timeseries_6h_dataset.py``; person_ids are checked against each NPZ.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

import numpy as np

from common.fold_data import default_data_dir, default_ts6h_data_dir, resolve_fold_paths
from timeseries.timeseries_6h_common import episode_to_bucket_features, episode_to_bucket_obs, obs_feature_indices

_LAYOUT: Dict = {}
_LAB_STATS: Dict = {}


def _init(layout, lab_stats):
    _LAYOUT.update(layout)
    _LAB_STATS.update(lab_stats)


def _one(line: str):
    ep = json.loads(line)
    return str(ep.get("person_id", "")), np.packbits(episode_to_bucket_obs(ep, _LAYOUT, _LAB_STATS), axis=-1)


def obs_for_file(path: Path, pool: Pool) -> tuple[List[str], np.ndarray]:
    with path.open() as f:
        lines = [ln for ln in f if ln.strip()]
    res = pool.imap(_one, lines, chunksize=64)
    pids, packed = zip(*res)
    return list(pids), np.stack(packed)


def check_pids(npz_path: Path, pids: List[str]) -> None:
    ref = np.load(npz_path, allow_pickle=True)["person_ids"]
    if len(ref) != len(pids) or any(str(a) != b for a, b in zip(ref, pids)):
        raise RuntimeError(f"person_id order mismatch for {npz_path}")


def check_features(npz_path: Path, lines_path: Path, layout, lab_stats, n: int = 200) -> None:
    feats = np.load(npz_path, allow_pickle=True)["features"]
    with lines_path.open() as f:
        for i, ln in enumerate(f):
            if i >= n:
                break
            x = episode_to_bucket_features(json.loads(ln), layout, lab_stats).astype(np.float16)
            if not np.array_equal(x, feats[i]):
                raise RuntimeError(f"feature mismatch at row {i} of {npz_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--folds_dir", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--fold_filename_template", default="7d_window_clm.mixed.fold{}.jsonl")
    p.add_argument("--test_filename", default="7d_window_clm.mixed.test.jsonl")
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    folds_dir = Path(args.folds_dir or default_data_dir())
    root = Path(args.data_root or default_ts6h_data_dir())
    layout = json.loads((root / "vocab/feature_layout.json").read_text())
    layout["lab_to_i"] = {int(k): v for k, v in layout["lab_to_i"].items()}
    lab_stats = json.loads((root / "vocab/lab_zscore.json").read_text())
    fold_paths = resolve_fold_paths(folds_dir, args.fold_filename_template)
    test_path = folds_dir / args.test_filename
    n_obs = len(obs_feature_indices(layout))

    check_features(root / "test.npz", test_path, layout, lab_stats)
    print("feature check OK (test, first 200 episodes)")

    with Pool(args.workers, initializer=_init, initargs=(layout, lab_stats)) as pool:
        per_fold = {}
        for k in range(1, 6):
            per_fold[k] = obs_for_file(fold_paths[k], pool)
            print(f"fold{k}: {per_fold[k][1].shape}")
        test_pids, test_obs = obs_for_file(test_path, pool)

    check_pids(root / "test.npz", test_pids)
    np.save(root / "test.obs.npy", test_obs)
    for v in range(1, 6):
        split = root / "splits" / f"val_fold_{v}"
        tr = [k for k in range(1, 6) if k != v]
        tr_pids = sum((per_fold[k][0] for k in tr), [])
        check_pids(split / "train.npz", tr_pids)
        check_pids(split / "val.npz", per_fold[v][0])
        np.save(split / "train.obs.npy", np.concatenate([per_fold[k][1] for k in tr]))
        np.save(split / "val.obs.npy", per_fold[v][1])
        print(f"val_fold_{v}: saved")

    meta = {"n_obs": n_obs, "packed_axis": -1,
            "columns": "CCS (all), lab value columns, medications; see obs_feature_indices",
            "definition": "1 = observed in the 6h bucket before LOCF; BASE bucket has no observations"}
    (root / "vocab/obs_mask.json").write_text(json.dumps(meta, indent=2))
    print("done")


if __name__ == "__main__":
    main()
