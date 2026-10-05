#!/usr/bin/env python3
"""Merge full-cohort contribution shards and compute 5-fold replication tables.

Expects ``contribution_5fold_full/fold{k}/shard{s}/contribution_per_patient_cells.csv``
and writes:
  - fold{k}/contribution_per_patient_cells.csv   (concatenated shards)
  - crossfold_baseline.csv / crossfold_online.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=str,
                   default=str(PROJECT_ROOT / "result/event_stream/transformer/contribution_5fold_full"))
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--min_patients", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_merge", action="store_true")
    return p.parse_args()


def parse_cell(cell: str) -> tuple[str, str, str]:
    parts = cell.split("|")
    return parts[0], "|".join(parts[1:-1]), parts[-1]


def merge_folds(root: Path) -> None:
    for f in range(1, 6):
        parts = sorted((root / f"fold{f}").glob("shard*/contribution_per_patient_cells.csv"))
        if not parts:
            raise SystemExit(f"fold{f}: no shard*/contribution_per_patient_cells.csv found")
        print(f"fold{f}: merging {len(parts)} shards", flush=True)
        d = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
        key = ["scope", "modality", "raw_code", "time_bin", "procedure_region", "person_id"]
        dup = int(d.duplicated(key).sum())
        out = root / f"fold{f}" / "contribution_per_patient_cells.csv"
        d.to_csv(out, index=False)
        print(f"fold{f}: {len(d):,} rows (dup keys={dup}) → {out}", flush=True)


def bootstrap_scope(d: pd.DataFrame, scope: str, root: Path,
                    n_boot: int, min_patients: int, rng: np.random.Generator) -> pd.DataFrame:
    s = d[d.scope == scope].copy()
    s["cell"] = s.modality + "|" + s.raw_code + "|" + s.time_bin
    piv = s.pivot_table(index=["cell", "person_id"], columns="fold",
                        values="mean_delta_logit", aggfunc="mean").dropna()
    out = []
    cells = list(piv.groupby(level=0))
    for i, (cell, grp) in enumerate(cells, 1):
        if i % 400 == 0:
            print(f"  {scope} {i}/{len(cells)}", flush=True)
        A = grp.to_numpy()
        n = A.shape[0]
        if n < min_patients:
            continue
        fold_means = A.mean(axis=0)
        pooled = float(fold_means.mean())
        idx = rng.integers(0, n, size=(n_boot, n))
        bs = A[idx].mean(axis=1).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        m, c, t = parse_cell(cell)
        out.append(dict(
            modality=m, raw_code=c, time_bin=t, n_patients=n,
            pooled_delta_logit=pooled, ci_lo=float(lo), ci_hi=float(hi),
            ci_excludes_zero=bool(lo > 0 or hi < 0),
            folds_agreeing=int((np.sign(fold_means) == np.sign(pooled)).sum()),
            fold_min=float(fold_means.min()), fold_max=float(fold_means.max()),
            fold_sd=float(fold_means.std(ddof=1)),
        ))
    r = pd.DataFrame(out).sort_values("pooled_delta_logit", key=abs, ascending=False)
    r["replicated"] = (r.folds_agreeing == 5) & r.ci_excludes_zero
    path = root / f"crossfold_{scope}.csv"
    r.to_csv(path, index=False)
    print(f"### {scope}: {len(r)} cells (≥{min_patients} pts); "
          f"replicated={int(r.replicated.sum())} → {path}", flush=True)
    return r


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not args.skip_merge:
        merge_folds(root)
    print("loading folds …", flush=True)
    d = pd.concat(
        [pd.read_csv(root / f"fold{f}" / "contribution_per_patient_cells.csv").assign(fold=f)
         for f in range(1, 6)],
        ignore_index=True,
    )
    d = d[d.modality != "procedure"].copy()
    rng = np.random.default_rng(args.seed)
    for scope in ("baseline", "online"):
        bootstrap_scope(d, scope, root, args.n_boot, args.min_patients, rng)


if __name__ == "__main__":
    main()
