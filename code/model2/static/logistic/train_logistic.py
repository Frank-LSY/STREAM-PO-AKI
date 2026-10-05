#!/usr/bin/env python3
"""Train L2 logistic regression on five_folds_xgboost and evaluate baseline AKI risk on test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from common.utils import save_json
from static.baseline_common import (
    compute_metrics,
    default_episode_path,
    load_feature_columns,
    load_xy,
    print_metrics_table,
    write_baseline_metrics_json,
)
from common.paths import PROJECT_ROOT


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float,
    max_iter: int,
    seed: int,
    solver: str,
) -> LogisticRegression:
    clf = LogisticRegression(
        penalty="l2",
        C=float(C),
        class_weight="balanced",
        max_iter=int(max_iter),
        solver=solver,
        random_state=int(seed),
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    return clf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train/eval logistic regression static preop AKI baseline."
    )
    p.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data/five_folds_xgboost4"),
        help="Parquet dataset built by build_xgboost_dataset.py (shared features).",
    )
    p.add_argument("--val_fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Checkpoint dir (default: result/static/logistic/fold{k}).",
    )
    p.add_argument(
        "--eval_output_dir",
        type=str,
        default=None,
        help="Test metrics dir (default: result/static/logistic/eval_fold{k}_test).",
    )
    p.add_argument(
        "--episode_path",
        type=str,
        default=None,
        help="Test JSONL path for report parity with evaluate_baseline.py.",
    )
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--print_table", action="store_true")
    p.add_argument("--skip_train", action="store_true", help="Load existing model.joblib only.")
    p.add_argument("--C", type=float, default=1.0, help="Inverse L2 regularization strength.")
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument(
        "--solver",
        type=str,
        default="saga",
        choices=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    val_fold = int(args.val_fold)
    model_dir = Path(
        args.model_dir
        or f"{PROJECT_ROOT}/result/static/logistic/fold{val_fold}"
    ).resolve()
    eval_dir = Path(
        args.eval_output_dir
        or f"{PROJECT_ROOT}/result/static/logistic/eval_fold{val_fold}_test"
    ).resolve()

    feature_columns = load_feature_columns(data_root)
    split_dir = data_root / "splits" / f"val_fold_{val_fold}"
    train_path = split_dir / "train.parquet"
    val_path = split_dir / "val.parquet"
    test_path = data_root / "test.parquet"
    episode_path = Path(args.episode_path or default_episode_path(data_root)).resolve()

    X_train, y_train, _ = load_xy(train_path, feature_columns)
    X_val, y_val, _ = load_xy(val_path, feature_columns)
    X_test, y_test, test_df = load_xy(test_path, feature_columns)

    train_params: Dict[str, Any] = {
        "penalty": "l2",
        "C": float(args.C),
        "class_weight": "balanced",
        "max_iter": int(args.max_iter),
        "solver": str(args.solver),
        "seed": int(args.seed),
        "n_features": len(feature_columns),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
    }

    model_path = model_dir / "model.joblib"
    config_path = model_dir / "train_config.json"

    if args.skip_train:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        clf = joblib.load(model_path)
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                train_params = json.load(f).get("train_params", train_params)
        print(f"Loaded {model_path}")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        clf = train_classifier(
            X_train,
            y_train,
            C=args.C,
            max_iter=args.max_iter,
            seed=args.seed,
            solver=args.solver,
        )
        val_prob = clf.predict_proba(X_val)[:, 1]
        val_loss = float(log_loss(y_val, val_prob, labels=[0, 1]))
        train_params["val_log_loss"] = val_loss
        joblib.dump(clf, model_path)
        save_json(
            {
                "val_fold": val_fold,
                "train_parquet": str(train_path),
                "val_parquet": str(val_path),
                "feature_columns": str(data_root / "vocab" / "feature_columns.json"),
                "train_params": train_params,
            },
            config_path,
        )
        print(f"Saved model -> {model_path} (val_log_loss={val_loss:.5f})")

    y_prob = clf.predict_proba(X_test)[:, 1].astype(np.float64)
    metrics = compute_metrics(y_test, y_prob, args.cls_threshold)
    json_path = write_baseline_metrics_json(
        eval_dir,
        metrics,
        evaluation="baseline_preoperative_aki_risk",
        model_type="logistic",
        val_fold=val_fold,
        model_dir=model_dir,
        data_root=data_root,
        test_parquet=test_path,
        episode_path=episode_path,
        cls_threshold=args.cls_threshold,
        train_params=train_params,
        checkpoint_path=model_path,
        best_iteration=None,
    )

    pred_path = eval_dir / "test_predictions.parquet"
    out_pred = test_df[["person_id", "label_aki"]].copy()
    out_pred["pred_prob"] = y_prob
    out_pred.to_parquet(pred_path, index=False)

    if args.print_table:
        print_metrics_table(metrics, args.cls_threshold, "Logistic regression preop AKI (test set)")

    print(f"Wrote {json_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
