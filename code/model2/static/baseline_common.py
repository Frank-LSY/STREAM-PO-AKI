"""Shared helpers for static baseline models (logistic, XGBoost, …)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from common.fold_data import default_test_jsonl
from common.utils import _subset_classification_metrics, save_json


def load_feature_columns(data_root: Path) -> List[str]:
    path = data_root / "vocab" / "feature_columns.json"
    with path.open(encoding="utf-8") as f:
        return list(json.load(f))


def load_xy(
    parquet_path: Path,
    feature_columns: List[str],
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(parquet_path)
    X = df[feature_columns].to_numpy(dtype=np.float32)
    y = df["label_aki"].to_numpy(dtype=np.float64)
    return X, y, df


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cls_threshold: float,
) -> Dict[str, Any]:
    return _subset_classification_metrics(y_true, y_prob, cls_threshold)


def print_metrics_table(m: Dict[str, Any], tau: float, title: str) -> None:
    cm = m.get("confusion_matrix") or {}
    print(f"\n=== {title} (τ={tau}) ===")
    print(f"  n={m.get('n')}  n_pos={m.get('n_pos')}  n_neg={m.get('n_neg')}  pos_rate={m.get('pos_rate')}")
    print(f"  AUROC={m.get('auroc')}  AUPRC={m.get('auprc')}")
    print(
        f"  accuracy={m.get('accuracy')}  precision={m.get('precision')}  "
        f"sensitivity={m.get('sensitivity')}  recall={m.get('recall')}  "
        f"specificity={m.get('specificity')}  f1={m.get('f1')}"
    )
    print(
        f"  confusion (tn, fp, fn, tp)=({cm.get('tn')}, {cm.get('fp')}, {cm.get('fn')}, {cm.get('tp')})"
    )


def write_baseline_metrics_json(
    out_dir: Path,
    metrics: Dict[str, Any],
    *,
    evaluation: str,
    model_type: str,
    val_fold: int,
    model_dir: Path,
    data_root: Path,
    test_parquet: Path,
    episode_path: Path,
    cls_threshold: float,
    train_params: Dict[str, Any],
    checkpoint_path: Path,
    best_iteration: Any = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "evaluation": evaluation,
        "prediction_unit": "one row per episode (episode-level AKI label)",
        "model_type": model_type,
        "val_fold": int(val_fold),
        "checkpoint": str(checkpoint_path.resolve()),
        "episode_path": str(episode_path.resolve()),
        "model_dir": str(model_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "test_parquet": str(test_parquet.resolve()),
        "cls_threshold": float(cls_threshold),
        "train_params": train_params,
        "best_iteration": best_iteration,
        "metrics": metrics,
    }
    json_path = out_dir / "baseline_metrics.json"
    save_json(report, json_path)
    return json_path


def default_episode_path(data_root: Path | None = None) -> Path:
    return default_test_jsonl()
