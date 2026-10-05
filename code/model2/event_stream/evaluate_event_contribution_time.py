#!/usr/bin/env python3
"""Time-stratified event contribution: what kind of event, in what period, moves risk how much.

Groups every occurrence of a code inside a time window and deletes the whole group from the
sequence, so redundant repeated measurements cannot mask each other. Rankings use Δlogit, since
falling baseline risk over the postoperative horizon would compress Δprob and fake a decay.

Not causal — model attribution under deletion.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from event_stream.dataset import load_episode_jsonl
from event_stream.eval_common import build_model_from_checkpoint, load_checkpoint_bundle
from event_stream.event_contribution import (
    POSTOP_BIN_NAMES,
    aggregate_contributions,
    compare_across_strata,
    group_contributions_for_episode,
)
from common.med_exposure import infer_num_med_exposure
from common.utils import resolve_torch_device, save_json

TIME_BIN_ORDER = ["preop_7d_48h", "preop_48h_0h", *POSTOP_BIN_NAMES]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--episode_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--max_episodes", type=int, default=200, help="Cap episodes (0 = all).")
    p.add_argument("--max_query_steps", type=int, default=25)
    p.add_argument("--lookback_hours", type=float, default=72.0)
    p.add_argument("--max_groups", type=int, default=0, help="Cap groups per episode (0 = all).")
    p.add_argument("--min_count", type=int, default=10)
    p.add_argument("--min_patients", type=int, default=5,
                   help="Min distinct patients per cell; guards against one patient filling it.")
    p.add_argument("--max_query_hour", type=float, default=168.0,
                   help="Latest prediction time to score (0 = no cap). Positives are truncated at "
                        "their AKI event while negatives run the full horizon, so late queries "
                        "describe an almost purely negative risk set.")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--plot", action="store_true", help="Code x time-bin heatmap of mean delta logit.")
    p.add_argument("--heatmap_top", type=int, default=25)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--save_records", action="store_true", help="Write all flat records (large).")
    p.add_argument("--procedure_region_map", type=str, default="",
                   help="JSON mapping procedure code to surgical region; enables stratified output.")
    p.add_argument("--per_region_episodes", type=int, default=0,
                   help="Sample up to this many episodes per region instead of taking the first N.")
    p.add_argument("--min_count_stratified", type=int, default=10,
                   help="Min measurements per (group, region) cell in the stratified comparison.")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Split the selected episodes across this many processes. Safe to shard: "
                        "the per-patient cell file that the cross-fold bootstrap consumes is "
                        "written from unfiltered records, so concatenating shards loses nothing. "
                        "The per-shard aggregate CSVs are partial and should be ignored.")
    p.add_argument("--shard_index", type=int, default=0)
    return p.parse_args()


def load_region_map(path: str) -> Dict[str, str]:
    if not path:
        return {}
    with open(path, encoding="utf-8") as f:
        return {str(k): str(v) for k, v in json.load(f).items()}


def episode_region(episode: Dict[str, Any], region_map: Dict[str, str]) -> str:
    for e in episode.get("events", []):
        if str(e.get("modality", "")).lower() == "procedure":
            return region_map.get(str(e.get("code")), "other")
    return "other"


def select_episodes(
    episodes: Sequence[Dict[str, Any]],
    region_map: Dict[str, str],
    per_region: int,
    max_total: int,
) -> List[Dict[str, Any]]:
    """Balance small surgical regions up, so every stratum can support its own ranking."""
    if per_region <= 0 or not region_map:
        return list(episodes[:max_total] if max_total > 0 else episodes)
    picked: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for ep in episodes:
        r = episode_region(ep, region_map)
        if len(picked[r]) < per_region:
            picked[r].append(ep)
    out = [ep for r in sorted(picked) for ep in picked[r]]
    return out[:max_total] if max_total > 0 else out


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  Wrote {path} (n={len(rows)})")


def write_per_patient_cells(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    """Collapse each patient to one value per cell, which is what resampling needs.

    Bootstrapping the raw records would resample query steps, treating one patient's 25 steps as
    25 independent observations and shrinking the interval far below what the patient count
    supports. Collapsing first makes the patient the unit of resampling.
    """
    cells: Dict[tuple, List[float]] = defaultdict(list)
    labels: Dict[tuple, int] = {}
    for r in records:
        key = (
            str(r.get("scope", "")), str(r.get("modality", "")), str(r.get("raw_code", "")),
            str(r.get("time_bin", "")), str(r.get("procedure_region", "")),
            str(r.get("person_id", "")),
        )
        cells[key].append(float(r["delta_logit"]))
        labels[key] = int(r.get("label_aki", 0))
    fields = ["scope", "modality", "raw_code", "time_bin", "procedure_region", "person_id",
              "label_aki", "n_measurements", "mean_delta_logit"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for key, vals in cells.items():
            w.writerow({
                **dict(zip(fields[:6], key)),
                "label_aki": labels[key],
                "n_measurements": len(vals),
                "mean_delta_logit": sum(vals) / len(vals),
            })
    print(f"  Wrote {path} ({len(cells)} patient-cell rows)")


def plot_code_time_heatmap(
    ranked: Sequence[Dict[str, Any]], out_path: Path, top_n: int, dpi: int
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    strength: Dict[str, float] = {}
    cells: Dict[str, Dict[str, float]] = {}
    for r in ranked:
        code = f"{r['modality'][:3]}:{r['raw_code']}"
        cells.setdefault(code, {})[r["time_bin"]] = r["mean_delta_logit"]
        strength[code] = max(strength.get(code, 0.0), abs(r["mean_delta_logit"]))
    codes = sorted(strength, key=lambda c: strength[c], reverse=True)[:top_n]
    if not codes:
        return

    bins = [b for b in TIME_BIN_ORDER if any(b in cells[c] for c in codes)]
    mat = np.full((len(codes), len(bins)), np.nan)
    for i, c in enumerate(codes):
        for j, b in enumerate(bins):
            if b in cells[c]:
                mat[i, j] = cells[c][b]

    lim = float(np.nanmax(np.abs(mat))) or 1.0
    fig, ax = plt.subplots(figsize=(1.05 * len(bins) + 4.5, 0.34 * len(codes) + 2.0))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([b.replace("postop_", "").replace("preop_", "pre ") for b in bins],
                       rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels([c[:40] for c in codes], fontsize=7)
    ax.set_xlabel("Time window of the event")
    ax.set_title("Mean Δlogit from deleting all occurrences of a code within a window\n"
                 "red = event raises predicted AKI risk, blue = lowers it", fontsize=9)
    fig.colorbar(im, ax=ax, label="Mean Δlogit", fraction=0.025)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"  Wrote {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, code_vocab, lab_stats, age_stats, num_med = load_checkpoint_bundle(args.checkpoint)
    episodes = load_episode_jsonl(args.episode_path)
    if num_med < 0:
        num_med = infer_num_med_exposure(episodes)
    region_map = load_region_map(args.procedure_region_map)
    episodes = select_episodes(episodes, region_map, args.per_region_episodes, args.max_episodes)
    if args.num_shards > 1:
        episodes = episodes[args.shard_index :: args.num_shards]     # strided, so every shard
        print(f"  shard {args.shard_index}/{args.num_shards}: {len(episodes)} episodes")
    if region_map:
        print(f"  region mix: {Counter(episode_region(e, region_map) for e in episodes).most_common()}")

    model = build_model_from_checkpoint(code_vocab, ckpt.get("args", {}) or {}, num_med)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    device = resolve_torch_device(args.device)
    model.to(device)
    model.eval()

    records: List[Dict[str, Any]] = []
    for i, ep in enumerate(episodes, 1):
        if i % 50 == 0 or i == 1:
            print(f"  contribution {i}/{len(episodes)} (records so far: {len(records)})")
        rows = group_contributions_for_episode(
            model,
            ep,
            code_vocab,
            lab_stats,
            num_med,
            device,
            max_query_steps=args.max_query_steps,
            lookback_hours=args.lookback_hours,
            max_groups=args.max_groups,
            max_query_hour=args.max_query_hour,
            age_stats=age_stats,
        )
        if region_map:
            region = episode_region(ep, region_map)
            for r in rows:
                r["procedure_region"] = region
        records.extend(rows)

    settings = {
        "max_episodes": args.max_episodes,
        "max_query_steps": args.max_query_steps,
        "lookback_hours": args.lookback_hours,
        "max_groups": args.max_groups,
        "min_count": args.min_count,
        "min_patients": args.min_patients,
        "max_query_hour": args.max_query_hour,
        "top_k": args.top_k,
    }

    by_code_time = aggregate_contributions(
        records, by=("modality", "raw_code", "time_bin"),
        scope="online", min_count=args.min_count, min_patients=args.min_patients, top_k=args.top_k,
    )
    by_code_time_lag = aggregate_contributions(
        records, by=("modality", "raw_code", "time_bin", "lag_bin"),
        scope="online", min_count=args.min_count, min_patients=args.min_patients, top_k=args.top_k,
    )
    baseline_by_code_time = aggregate_contributions(
        records, by=("modality", "raw_code", "time_bin"),
        scope="baseline", min_count=args.min_count, min_patients=args.min_patients, top_k=args.top_k,
    )
    by_time_only = aggregate_contributions(
        records, by=("time_bin",), scope="online", min_count=args.min_count, min_patients=args.min_patients, top_k=50,
    )
    by_lag_only = aggregate_contributions(
        records, by=("lag_bin",), scope="online", min_count=args.min_count, min_patients=args.min_patients, top_k=50,
    )

    stratified: List[Dict[str, Any]] = []
    per_region_rankings: Dict[str, Any] = {}
    if region_map:
        stratified = compare_across_strata(
            records, "procedure_region", ("modality", "raw_code", "time_bin"),
            scope="online", min_count=args.min_count_stratified, min_patients=args.min_patients,
        )
        for region in sorted({r.get("procedure_region", "") for r in records}):
            subset = [r for r in records if r.get("procedure_region") == region]
            agg = aggregate_contributions(
                subset, by=("modality", "raw_code", "time_bin"),
                scope="online", min_count=args.min_count_stratified, min_patients=args.min_patients, top_k=args.top_k,
            )
            per_region_rankings[region] = {
                "n_episodes": len({r["person_id"] for r in subset}),
                "n_measurements": len(subset),
                "risk_increasing": agg["risk_increasing"][:20],
                "risk_decreasing": agg["risk_decreasing"][:20],
                "n_groups_ranked": agg["n_groups_ranked"],
            }

    people = {r["person_id"] for r in records}
    report = {
        "analysis_type": "event_contribution_time_stratified",
        "method": "group_deletion_logit_difference",
        "disclaimer": (
            "Each measurement deletes every occurrence of one code inside one time window from "
            "the sequence and re-runs the model. delta = logit(with group) - logit(without), so "
            "positive means the events raise predicted risk. Model attribution under an "
            "off-manifold intervention, not a causal effect."
        ),
        "definitions": {
            "time_bin": "When the deleted events occurred, on the same 6h-aligned grid as evaluation",
            "lag_bin": "Gap between the deleted events and the prediction time; from_preop for preoperative groups",
            "delta_logit": "Primary ranking metric; unlike delta_prob it does not shrink as baseline risk falls",
            "baseline_scope": "Effect on the [BASE] token, reported for preoperative groups only",
            "online_scope": "Effect on postoperative query tokens at or after the deleted events",
        },
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "episode_path": str(Path(args.episode_path).resolve()),
        "settings": settings,
        "n_episodes": len(people),
        "n_aki_positive_episodes": len({r["person_id"] for r in records if r["label_aki"] == 1}),
        "n_deletion_measurements": len(records),
        "measurements_by_time_bin": dict(Counter(r["time_bin"] for r in records)),
        "online_by_code_and_time": by_code_time,
        "online_by_code_time_and_lag": by_code_time_lag,
        "baseline_by_code_and_time": baseline_by_code_time,
        "online_by_time_bin": by_time_only["all_ranked"],
        "online_by_lag_bin": by_lag_only["all_ranked"],
    }
    if region_map:
        consistent = [r for r in stratified if r["consistent_sign"]]
        report["stratification"] = {
            "stratum_field": "procedure_region",
            "note": (
                "A contribution holding its sign across surgical regions is a property of the "
                "event; one confined to a single region reflects case mix, because the index "
                "operation dominates baseline risk and is present in every episode."
            ),
            "region_mix": dict(Counter(r.get("procedure_region", "") for r in records)),
            "n_groups_testable": len(stratified),
            "n_groups_sign_consistent": len(consistent),
            "most_case_mix_dependent": sorted(
                stratified, key=lambda r: r["across_stratum_range"], reverse=True
            )[:25],
            "robust_across_regions": sorted(
                consistent, key=lambda r: abs(r["pooled_mean_delta_logit"]), reverse=True
            )[:25],
            "per_region_rankings": per_region_rankings,
        }
    save_json(report, out_dir / "event_contribution_time_summary.json")

    if stratified:
        regions = sorted({r.get("procedure_region", "") for r in records})
        strat_fields = (
            ["modality", "raw_code", "time_bin", "n_strata", "n_measurements",
             "pooled_mean_delta_logit", "across_stratum_range", "consistent_sign",
             "n_patients_total"]
            + [f"mean__{r}" for r in regions] + [f"pat__{r}" for r in regions]
            + [f"n__{r}" for r in regions]
        )
        write_csv(out_dir / "contribution_stratified_by_region.csv", stratified, strat_fields)

    base_fields = [
        "modality", "raw_code", "time_bin", "n_measurements", "n_patients",
        "mean_delta_logit", "median_delta_logit", "std_delta_logit",
        "mean_delta_prob", "fraction_positive",
    ]
    write_csv(out_dir / "contribution_by_code_and_time.csv", by_code_time["all_ranked"], base_fields)
    write_csv(
        out_dir / "contribution_by_code_time_and_lag.csv",
        by_code_time_lag["all_ranked"],
        base_fields[:3] + ["lag_bin"] + base_fields[3:],
    )
    write_csv(
        out_dir / "contribution_baseline_by_code_and_time.csv",
        baseline_by_code_time["all_ranked"],
        base_fields,
    )
    write_csv(
        out_dir / "contribution_by_time_bin.csv",
        by_time_only["all_ranked"],
        ["time_bin", "n_measurements", "n_patients", "mean_delta_logit", "std_delta_logit"],
    )

    write_per_patient_cells(out_dir / "contribution_per_patient_cells.csv", records)

    if args.save_records:
        save_json(records, out_dir / "event_contribution_records.json")
    if args.plot:
        plot_code_time_heatmap(
            by_code_time["all_ranked"],
            out_dir / "contribution_code_by_time_heatmap.png",
            args.heatmap_top,
            args.dpi,
        )

    print("=== Time-stratified event contribution (group deletion, Δlogit) ===")
    print(f"  episodes: {report['n_episodes']} (AKI+: {report['n_aki_positive_episodes']})")
    print(f"  deletion measurements: {report['n_deletion_measurements']}")
    print(f"  ranked (code x time) groups: {by_code_time['n_groups_ranked']}")
    print("  top risk-increasing (code x time):")
    for i, r in enumerate(by_code_time["risk_increasing"][:10], 1):
        print(f"    {i:2d}. [{r['modality']:9s}] {r['raw_code'][:26]:26s} {r['time_bin']:15s} "
              f"Δlogit={r['mean_delta_logit']:+.4f}  n={r['n_measurements']}")
    print("  top risk-decreasing (code x time):")
    for i, r in enumerate(by_code_time["risk_decreasing"][:10], 1):
        print(f"    {i:2d}. [{r['modality']:9s}] {r['raw_code'][:26]:26s} {r['time_bin']:15s} "
              f"Δlogit={r['mean_delta_logit']:+.4f}  n={r['n_measurements']}")
    if stratified:
        s = report["stratification"]
        print(f"  testable in >=2 surgical regions: {s['n_groups_testable']}  "
              f"(sign-consistent: {s['n_groups_sign_consistent']})")
        print("  robust across regions (real event effect):")
        for i, r in enumerate(s["robust_across_regions"][:8], 1):
            print(f"    {i}. [{r['modality']:9s}] {r['raw_code'][:22]:22s} {r['time_bin']:15s} "
                  f"pooled={r['pooled_mean_delta_logit']:+.3f}  range={r['across_stratum_range']:.3f}"
                  f"  pts={r['n_patients_total']}")
        print("  most case-mix dependent (pooled value is unreliable):")
        for i, r in enumerate(s["most_case_mix_dependent"][:8], 1):
            print(f"    {i}. [{r['modality']:9s}] {r['raw_code'][:22]:22s} {r['time_bin']:15s} "
                  f"pooled={r['pooled_mean_delta_logit']:+.3f}  range={r['across_stratum_range']:.3f}"
                  f"  pts={r['n_patients_total']}")
    print(f"Wrote {out_dir / 'event_contribution_time_summary.json'}")


if __name__ == "__main__":
    main()
