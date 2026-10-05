#!/usr/bin/env python3
"""Pool the per-fold age sweeps into one dose-response curve and plot it.

Each fold contributes one within-patient curve: for every patient, the logit at each age minus
that same patient's logit at the reference age. Folds are separate model instances trained on
different data, so agreement across all five is the replication criterion used everywhere else
in this analysis; the CI is a patient bootstrap of the pooled curve.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from common.paths import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--reference_age", type=float, default=60.0)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _reference_logit(t: pd.DataFrame, reference_age: float) -> pd.Series:
    """Map person_id → logit at reference age (exact grid hit, else linear interpolate)."""
    ages = np.array(sorted(t.age_set.unique()), dtype=float)
    if reference_age in set(ages) or np.any(np.isclose(ages, reference_age)):
        hit = ages[np.argmin(np.abs(ages - reference_age))]
        return t[np.isclose(t.age_set, hit)].set_index("person_id").baseline_logit

    if reference_age < ages.min() or reference_age > ages.max():
        raise SystemExit(
            f"reference_age={reference_age} outside sweep grid [{ages.min()}, {ages.max()}]"
        )
    lo = ages[ages <= reference_age].max()
    hi = ages[ages >= reference_age].min()
    w = (reference_age - lo) / (hi - lo)
    a = t[np.isclose(t.age_set, lo)].set_index("person_id").baseline_logit
    b = t[np.isclose(t.age_set, hi)].set_index("person_id").baseline_logit
    both = a.index.intersection(b.index)
    return (1.0 - w) * a.loc[both] + w * b.loc[both]


def main() -> None:
    args = parse_args()
    d = Path(args.input_dir).resolve()
    files = sorted(d.glob("fold*.csv"))
    if not files:
        raise SystemExit(f"no fold*.csv in {d}")

    per_fold = []
    for f in files:
        t = pd.read_csv(f)
        ref = _reference_logit(t, args.reference_age)
        t["delta"] = t.baseline_logit - t.person_id.map(ref)
        t["fold"] = f.stem
        per_fold.append(t)
    df = pd.concat(per_fold, ignore_index=True)

    # one value per (patient, age) so folds are averaged before the patient bootstrap
    cell = df.groupby(["person_id", "age_set"], as_index=False).delta.mean()
    wide = cell.pivot(index="person_id", columns="age_set", values="delta").dropna()
    ages = np.array(sorted(wide.columns))
    M = wide[ages].to_numpy()

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, M.shape[0], size=(args.n_boot, M.shape[0]))
    boot = M[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    mean = M.mean(axis=0)

    fold_mean = (df.groupby(["fold", "age_set"]).delta.mean()
                 .unstack(0).reindex(ages).to_numpy())
    monotone = [bool(np.all(np.diff(fold_mean[:, j]) > 0)) for j in range(fold_mean.shape[1])]

    out = pd.DataFrame({"age": ages, "mean_delta_logit": mean, "ci_lo": lo, "ci_hi": hi,
                        "fold_min": fold_mean.min(axis=1), "fold_max": fold_mean.max(axis=1),
                        "fold_sd": fold_mean.std(axis=1)})
    csv_path = d / "age_doseresponse_pooled.csv"
    out.to_csv(csv_path, index=False)

    # slope per decade over the range where most of the cohort actually sits
    band = (ages >= 40) & (ages <= 85)
    slope = np.polyfit(ages[band], mean[band], 1)[0] * 10.0

    print(f"  folds: {len(files)}   patients: {M.shape[0]}   reference age: {args.reference_age:.0f}")
    print(f"  every fold monotonically increasing in age: {all(monotone)}")
    print(f"  mean slope over 40-85y: {slope:+.3f} logit per decade")
    print(out.round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.fill_between(ages, fold_mean.min(axis=1), fold_mean.max(axis=1),
                    color="0.75", alpha=0.35, lw=0, zorder=1)
    ax.fill_between(ages, lo, hi, color="#c0392b", alpha=0.18, lw=0, zorder=2)
    ax.plot(ages, mean, color="#c0392b", lw=2.2, marker="o", ms=4.5, zorder=3)
    ax.axhline(0, color="0.35", lw=0.9)
    ax.axvline(args.reference_age, color="0.35", lw=0.9, ls=(0, (4, 3)))
    ax.text(args.reference_age, ax.get_ylim()[0], f"  ref: median {args.reference_age:.0f}",
            fontsize=8, style="italic", color="0.4", va="bottom")
    ax.set_xlabel("Age the patient is re-scored at (years)")
    ax.set_ylabel(f"\u0394logit reference to median age {args.reference_age:.0f}")
    ax.set_title("Directional age effect, by counterfactual value sweep\n"
                 "every patient re-scored at every age, so the curve is within-patient; "
                 "grey = individual folds",
                 fontsize=9.5)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    png = (PROJECT_ROOT / "result/figures/age_doseresponse.png")
    fig.savefig(png, dpi=170)
    print(f"Wrote {csv_path}\nWrote {png}")


if __name__ == "__main__":
    main()
