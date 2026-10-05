#!/usr/bin/env python3
"""Evaluate & visualize bucket-ts6h grid predictions (mirrors check_prediction.ipynb)."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from math import pi
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import t
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from event_stream.dataset import load_episode_jsonl
from timeseries.dataset import Ts6hEpisodeDataset
from timeseries.checkpoint import load_bucket_checkpoint
from timeseries.utils_eval import forward_bucket_model
from common.fold_data import default_test_jsonl, default_ts6h_data_dir
from common.surgery_category_map import primary_surgery_type
from timeseries.timeseries_6h_common import BASE_BUCKET_IDX, BUCKET_SIZE_H, NUM_POSTOP_BUCKETS
from common.utils import resolve_torch_device, save_json


def load_model(checkpoint: Path, device: torch.device):
    model, _ckpt, _train_args, _layout = load_bucket_checkpoint(checkpoint, device)
    return model


@torch.no_grad()
def export_fold_predictions(
    checkpoint: Path,
    test_npz: Path,
    episode_meta: Dict[str, Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    bucket_size: float,
) -> List[Dict[str, Any]]:
    model = load_model(checkpoint, device)
    test_ds = Ts6hEpisodeDataset(test_npz, load_obs=bool(getattr(model, "uses_obs_mask", False)))
    loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_ds.collate_fn,
    )
    records: List[Dict[str, Any]] = []
    for batch in loader:
        batch_t = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        out = forward_bucket_model(model, batch_t)
        token_probs = torch.sigmoid(out["token_logits"]).cpu().numpy()
        base_probs = torch.sigmoid(out["baseline_logits_all"]).cpu().numpy()
        labels = batch_t["labels"].cpu().numpy()
        max_postop = batch_t["max_postop_bucket"].cpu().numpy()
        person_ids = batch["person_ids"]

        B = token_probs.shape[0]
        for i in range(B):
            pid = str(person_ids[i])
            meta = episode_meta.get(pid, {})
            last_j = min(int(max_postop[i]), NUM_POSTOP_BUCKETS - 1)
            baseline_prob = float(base_probs[i, BASE_BUCKET_IDX])

            bucket_preds: List[Dict[str, Any]] = []
            for j in range(last_j + 1):
                idx = BASE_BUCKET_IDX + 1 + j
                bh = float(j * bucket_size)
                be = bh + bucket_size
                bucket_preds.append(
                    {
                        "bucket_hour": int(bh),
                        "bucket_hour_end": float(be),
                        "hour_range": f"[{int(bh)},{int(be)})",
                        "aki_risk_prob": float(token_probs[i, idx]),
                        "carry_source": "bucket_transformer_direct",
                    }
                )

            records.append(
                {
                    "person_id": pid,
                    "label_aki": int(labels[i]),
                    "aki_time": meta.get("aki_time"),
                    "t0": meta.get("t0"),
                    "surgery_type": meta.get("surgery_type", "Unknown"),
                    "baseline_aki_risk_prob": baseline_prob,
                    "bucket_size_hours": float(bucket_size),
                    "postop_horizon_hours": float(NUM_POSTOP_BUCKETS * bucket_size),
                    "max_bucket_hour_evaluated": int(last_j * bucket_size),
                    "n_buckets": len(bucket_preds),
                    "bucket_predictions": bucket_preds,
                }
            )
    return records


def write_jsonl(records: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _cls_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return {
            "auroc": None,
            "auprc": None,
            "f1": None,
            "recall": None,
            "precision": None,
            "accuracy": None,
            "sensitivity": None,
            "n": int(len(y_true)),
        }
    y_pred = (y_score >= threshold).astype(int)
    return {
        "auroc": float(roc_auc_score(y_true, y_score)),
        "auprc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred)),
        "n": int(len(y_true)),
    }


def evaluate_baseline_five_folds(
    fold_records: List[Tuple[str, List[Dict[str, Any]]]],
    threshold: float,
    out_dir: Path,
) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, List[float]]], set]:
    overall_metrics: Dict[str, List[float]] = defaultdict(list)
    per_type_fold_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    surgery_types_set: set = set()
    lines: List[str] = []

    lines.append("===== 五折总体 Baseline Performance =====")
    for fold_name, recs in fold_records:
        y_true = np.array([r["label_aki"] for r in recs], dtype=np.float64)
        y_score = np.array([r["baseline_aki_risk_prob"] for r in recs], dtype=np.float64)
        m = _cls_metrics(y_true, y_score, threshold)
        lines.append(f"\n== {fold_name} ==")
        if m["auroc"] is None:
            lines.append("标签只有一种，无法计算 AUROC 等分数。")
        else:
            for k in ("auroc", "auprc", "f1", "recall", "precision", "sensitivity", "accuracy"):
                lines.append(f"{k.upper()}: {m[k]:.4f}")
                overall_metrics[k].append(float(m[k]))

        type_y: Dict[str, List[float]] = defaultdict(list)
        type_s: Dict[str, List[float]] = defaultdict(list)
        for r in recs:
            typ = r.get("surgery_type", "Unknown")
            surgery_types_set.add(typ)
            type_y[typ].append(float(r["label_aki"]))
            type_s[typ].append(float(r["baseline_aki_risk_prob"]))

        total = sum(len(v) for v in type_y.values())
        for typ in sorted(type_y.keys()):
            yt = np.array(type_y[typ])
            ys = np.array(type_s[typ])
            tm = _cls_metrics(yt, ys, threshold)
            pct = len(yt) / total if total else 0.0
            lines.append(f"\n[{fold_name} | {typ}] count: {len(yt)}, percentage: {pct:.2%}")
            if tm["auroc"] is None:
                lines.append(f"[{fold_name} | {typ}] 只有一种 label，跳过。")
                continue
            for k in ("auroc", "auprc", "f1", "recall", "precision", "sensitivity", "accuracy"):
                lines.append(f"{k.upper()}: {tm[k]:.4f}")
                per_type_fold_metrics[typ][k].append(float(tm[k]))

    lines.append("\n====== 五折整体指标 均值 [最小, 最大] ======")
    for metric, vals in overall_metrics.items():
        if vals:
            lines.append(f"{metric.upper()}: {np.mean(vals):.4f}  [{np.min(vals):.4f}, {np.max(vals):.4f}]")
        else:
            lines.append(f"{metric}: 无法计算")

    lines.append("\n====== 分 surgery_type 五折指标（均值 [最小, 最大]） ======")
    for typ in sorted(surgery_types_set):
        lines.append(f"\n--- Surgery_type: {typ} ---")
        for metric in ("auroc", "auprc", "f1", "recall", "precision", "accuracy", "sensitivity"):
            vals = per_type_fold_metrics[typ][metric]
            if vals:
                lines.append(f"{metric.upper()}: {np.mean(vals):.4f} [{np.min(vals):.4f}, {np.max(vals):.4f}]")
            else:
                lines.append(f"{metric}: 无法计算")

    summary_path = out_dir / "five_fold_baseline_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary_path.read_text(encoding="utf-8"))
    return overall_metrics, per_type_fold_metrics, surgery_types_set


def plot_surgery_type_baseline(
    per_type_fold_metrics: Dict[str, Dict[str, List[float]]],
    surgery_types_set: set,
    out_dir: Path,
    dpi: int,
) -> None:
    metrics_to_show = ["auroc", "auprc", "f1", "recall", "precision", "accuracy"]
    records = []
    for typ in sorted(surgery_types_set):
        for metric in metrics_to_show:
            for v in per_type_fold_metrics[typ][metric]:
                records.append({"surgery_type": typ, "metric": metric.upper(), "score": v})
    if not records:
        return
    df_folds = pd.DataFrame.from_records(records)

    plt.figure(figsize=(14, 6))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("tab10", n_colors=len(df_folds["surgery_type"].unique()))
    sns.barplot(
        data=df_folds,
        x="metric",
        y="score",
        hue="surgery_type",
        palette=palette,
        capsize=0.15,
        err_kws={"color": "gray", "linewidth": 2},
        errorbar="ci",
    )
    plt.ylabel("Score")
    plt.title("surgery_type--baseline performance (5-fold)")
    plt.ylim(0, 1.05)
    plt.legend(title="surgery_type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "surgery_type_barplot.png", dpi=dpi)
    plt.close()

    type_metric_avg = {typ: {} for typ in sorted(surgery_types_set)}
    for typ in sorted(surgery_types_set):
        for metric in metrics_to_show:
            vals = per_type_fold_metrics[typ][metric]
            type_metric_avg[typ][metric] = float(np.mean(vals)) if vals else np.nan

    N = len(metrics_to_show)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    plt.figure(figsize=(8, 8))
    colors = sns.color_palette("tab10", n_colors=len(sorted(surgery_types_set)))
    for idx, typ in enumerate(sorted(surgery_types_set)):
        values = [
            type_metric_avg[typ][m] if not np.isnan(type_metric_avg[typ][m]) else 0.0
            for m in metrics_to_show
        ]
        values += values[:1]
        plt.polar(angles, values, label=typ, marker="o", alpha=0.85, color=colors[idx], linewidth=2)
    plt.xticks(angles[:-1], [m.upper() for m in metrics_to_show])
    plt.title("surgery_type--baseline performance", y=1.05)
    plt.legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    plt.savefig(out_dir / "surgery_type_radar.png", dpi=dpi)
    plt.close()


def _bucket_row_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(int)
    if len(y_true) == 0:
        return {k: np.nan for k in ("auc", "average_precision", "brier_score", "precision", "recall", "f1", "specificity", "pos_neg_ratio")}
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    ap = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_prob)
    if np.sum(y_pred) > 0 and np.sum(y_true) > 0:
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    else:
        prec = rec = f1 = np.nan
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    except ValueError:
        spec = np.nan
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    pnr = (n_pos / n_neg) if n_neg > 0 else np.nan
    return {
        "auc": auc,
        "average_precision": ap,
        "brier_score": brier,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "specificity": spec,
        "pos_neg_ratio": pnr,
        "n": len(y_true),
    }


def build_bucket_perf_df(fold_records: List[Tuple[str, List[Dict[str, Any]]]], threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold_name, recs in fold_records:
        buckets_data: List[Dict[str, Any]] = []
        for rec in recs:
            for b in rec.get("bucket_predictions", []):
                buckets_data.append(
                    {
                        "person_id": rec.get("person_id"),
                        "label_aki": rec["label_aki"],
                        "bucket_hour": b.get("bucket_hour"),
                        "aki_risk_prob": b.get("aki_risk_prob"),
                        "fold": fold_name,
                        "surgery_type": rec.get("surgery_type"),
                    }
                )
        df_buckets = pd.DataFrame(buckets_data)
        if df_buckets.empty:
            continue
        for surg_type in sorted(df_buckets["surgery_type"].dropna().unique()):
            for bh in sorted(df_buckets["bucket_hour"].dropna().unique()):
                sub = df_buckets[(df_buckets["bucket_hour"] == bh) & (df_buckets["surgery_type"] == surg_type)]
                yt = sub["label_aki"].values.astype(np.float64)
                yp = sub["aki_risk_prob"].values.astype(np.float64)
                m = _bucket_row_metrics(yt, yp, threshold)
                rows.append({"bucket_hour": bh, "fold": fold_name, "surgery_type": surg_type, **m})
    return pd.DataFrame(rows)


def plot_bucket_metrics_by_surgery_type(df_perf: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if df_perf.empty:
        return
    metrics_to_show = [
        "auc",
        "average_precision",
        "brier_score",
        "precision",
        "recall",
        "f1",
        "specificity",
        "pos_neg_ratio",
    ]
    n_metrics = len(metrics_to_show)
    n_cols = math.ceil(n_metrics / 2)
    n_rows = 2
    all_types = sorted(df_perf["surgery_type"].dropna().unique())
    type_colors = {typ: c for typ, c in zip(all_types, plt.cm.tab10.colors)}
    bucket_hours = sorted(df_perf["bucket_hour"].dropna().unique())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), sharex=True)
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    for i, metric in enumerate(metrics_to_show):
        ax = axes_flat[i]
        for surg_type in all_types:
            means, lo, hi = [], [], []
            for bh in bucket_hours:
                vals = df_perf[(df_perf["bucket_hour"] == bh) & (df_perf["surgery_type"] == surg_type)][metric].values
                vals = vals[~np.isnan(vals)]
                if len(vals) == 0:
                    means.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    continue
                mean = float(np.mean(vals))
                means.append(mean)
                if len(vals) > 1:
                    sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                    margin = sem * float(t.ppf(0.975, df=len(vals) - 1))
                    lo.append(mean - margin)
                    hi.append(mean + margin)
                else:
                    lo.append(np.nan)
                    hi.append(np.nan)
            color = type_colors.get(surg_type)
            ax.plot(bucket_hours, means, marker="o", label=surg_type, color=color, alpha=0.9)
            ax.fill_between(bucket_hours, lo, hi, color=color, alpha=0.17)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_xlabel("Bucket Hour")
        ax.set_title(f"{metric.replace('_', ' ').title()} by Time Bucket (5-fold mean ± 95% CI)")
        ax.grid(True)
        if i == 0:
            ax.legend(title="Surgery Type")

    for j in range(n_metrics, n_rows * n_cols):
        fig.delaxes(axes_flat[j])
    plt.tight_layout()
    plt.savefig(out_dir / "bucket_metrics_by_surgery_type.png", dpi=dpi)
    plt.close()


def build_episode_meta(test_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    episodes = load_episode_jsonl(str(test_jsonl))
    meta: Dict[str, Dict[str, Any]] = {}
    for ep in episodes:
        pid = str(ep.get("person_id", ""))
        meta[pid] = {
            "aki_time": ep.get("aki_time"),
            "t0": ep.get("t0"),
            "surgery_type": primary_surgery_type(ep),
        }
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Check bucket grid predictions (notebook parity).")
    p.add_argument("--result_root", type=str, default="../../result")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--test_npz", type=str, default=None)
    p.add_argument("--test_jsonl", type=str, default=None)
    p.add_argument("--folds", type=str, default="1,2,3,4,5")
    p.add_argument("--checkpoint_pattern", type=str, default="run_fold_bucket_ts6h_{fold}/best_model.pt")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--bucket_size", type=float, default=6.0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--skip_export", action="store_true", help="Use existing fold{k}_predictions.jsonl in output_dir.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_root = Path(args.result_root).resolve()
    out_dir = Path(args.output_dir or (result_root / "bucket_grid_check")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    test_npz = Path(args.test_npz or (default_ts6h_data_dir() / "test.npz")).resolve()
    test_jsonl = Path(args.test_jsonl or default_test_jsonl()).resolve()
    folds = [int(x.strip()) for x in args.folds.split(",") if x.strip()]
    device = resolve_torch_device(args.device)
    episode_meta = build_episode_meta(test_jsonl)

    fold_records: List[Tuple[str, List[Dict[str, Any]]]] = []
    for k in folds:
        fold_name = f"fold{k}"
        pred_path = out_dir / f"fold{k}_predictions.jsonl"
        if args.skip_export and pred_path.is_file():
            recs = read_jsonl(pred_path)
            print(f"Loaded {pred_path} ({len(recs)} episodes)")
        else:
            ckpt = result_root / args.checkpoint_pattern.format(fold=k)
            if not ckpt.is_file():
                raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
            print(f"Exporting {fold_name} from {ckpt} ...")
            recs = export_fold_predictions(
                ckpt, test_npz, episode_meta, device, args.batch_size, args.bucket_size
            )
            write_jsonl(recs, pred_path)
            print(f"Wrote {pred_path} ({len(recs)} episodes)")
        fold_records.append((fold_name, recs))

    _, per_type_fold_metrics, surgery_types_set = evaluate_baseline_five_folds(
        fold_records, args.cls_threshold, out_dir
    )
    plot_surgery_type_baseline(per_type_fold_metrics, surgery_types_set, out_dir, args.dpi)

    df_perf = build_bucket_perf_df(fold_records, args.cls_threshold)
    if not df_perf.empty:
        df_perf.to_csv(out_dir / "bucket_perf_by_surgery_type.csv", index=False)
        plot_bucket_metrics_by_surgery_type(df_perf, out_dir, args.dpi)

    manifest = {
        "output_dir": str(out_dir),
        "folds": folds,
        "test_npz": str(test_npz),
        "test_jsonl": str(test_jsonl),
        "cls_threshold": args.cls_threshold,
        "bucket_size_hours": args.bucket_size,
        "artifacts": [
            "five_fold_baseline_summary.txt",
            "surgery_type_barplot.png",
            "surgery_type_radar.png",
            "bucket_metrics_by_surgery_type.png",
            "bucket_perf_by_surgery_type.csv",
            "fold{k}_predictions.jsonl",
        ],
    }
    save_json(manifest, out_dir / "manifest.json")
    print(f"Done -> {out_dir}")


if __name__ == "__main__":
    main()
