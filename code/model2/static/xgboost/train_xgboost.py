#!/usr/bin/env python3
"""Train XGBoost on five_folds_xgboost and evaluate on test (baseline-aligned metrics)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.fold_data import default_test_jsonl
from common.utils import _subset_classification_metrics, save_json
from common.paths import PROJECT_ROOT


def _import_real_xgboost():
    """Import site-packages xgboost; local ``model2/xgboost`` shadows the library on PYTHONPATH."""
    import importlib
    import sys
    from pathlib import Path

    model2_root = Path(__file__).resolve().parents[2]

    def _is_shadow_path(p: str) -> bool:
        if not p or p == "":
            # cwd may be model2/ (contains local xgboost/ package)
            try:
                return Path.cwd().resolve() == model2_root.resolve()
            except OSError:
                return False
        try:
            return Path(p).resolve() == model2_root.resolve()
        except OSError:
            return False

    for name in list(sys.modules):
        if name == "xgboost" or name.startswith("xgboost."):
            mod = sys.modules.get(name)
            f = (getattr(mod, "__file__", None) or "").replace("\\", "/")
            paths = getattr(mod, "__path__", None)
            if "code/model2" in f or (
                paths and any("code/model2" in str(Path(x)).replace("\\", "/") for x in paths)
            ):
                del sys.modules[name]

    old_path = sys.path[:]
    sys.path[:] = [p for p in sys.path if not _is_shadow_path(p)]
    try:
        return importlib.import_module("xgboost")
    finally:
        sys.path[:] = old_path


try:
    xgb = _import_real_xgboost()
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is required. Install with: pip install xgboost"
    ) from exc
if not hasattr(xgb, "DMatrix"):
    raise ImportError(
        f"Imported a non-library xgboost module from {getattr(xgb, '__file__', None)}; "
        "check that site-packages xgboost is installed and not shadowed."
    )

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


def train_booster(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_columns: List[str],
    params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=50,
    )
    return booster


def write_baseline_metrics_json(
    out_dir: Path,
    metrics: Dict[str, Any],
    *,
    evaluation: str,
    val_fold: int,
    model_dir: Path,
    data_root: Path,
    test_parquet: Path,
    episode_path: Path,
    cls_threshold: float,
    train_params: Dict[str, Any],
    best_iteration: Optional[int],
    checkpoint_path: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "evaluation": evaluation,
        "prediction_unit": "one row per episode (episode-level AKI label)",
        "model_type": "xgboost",
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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/eval XGBoost static preop AKI model.")
    p.add_argument(
        "--data_root",
        type=str,
        default=str(PROJECT_ROOT / "data/five_folds_xgboost4"),
    )
    p.add_argument("--val_fold", type=int, required=True, choices=[1, 2, 3, 4, 5])
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Checkpoint dir (default: result/static/xgboost/fold{k}).",
    )
    p.add_argument(
        "--eval_output_dir",
        type=str,
        default=None,
        help="Test metrics dir (default: result/static/xgboost/eval_fold{k}_test).",
    )
    p.add_argument(
        "--episode_path",
        type=str,
        default=None,
        help="Test JSONL path for report parity with evaluate_baseline.py.",
    )
    p.add_argument("--cls_threshold", type=float, default=0.5)
    p.add_argument("--print_table", action="store_true")
    p.add_argument("--skip_train", action="store_true", help="Load existing model.json only.")
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--num_boost_round", type=int, default=500)
    p.add_argument("--early_stopping_rounds", type=int, default=30)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--min_child_weight", type=float, default=1.0)
    p.add_argument("--reg_lambda", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    val_fold = int(args.val_fold)
    model_dir = Path(
        args.model_dir
        or f"{PROJECT_ROOT}/result/static/xgboost/fold{val_fold}"
    ).resolve()
    eval_dir = Path(
        args.eval_output_dir
        or f"{PROJECT_ROOT}/result/static/xgboost/eval_fold{val_fold}_test"
    ).resolve()

    feature_columns = load_feature_columns(data_root)
    split_dir = data_root / "splits" / f"val_fold_{val_fold}"
    train_path = split_dir / "train.parquet"
    val_path = split_dir / "val.parquet"
    test_path = data_root / "test.parquet"
    default_jsonl = default_test_jsonl()
    episode_path = Path(args.episode_path or default_jsonl).resolve()

    X_train, y_train, _ = load_xy(train_path, feature_columns)
    X_val, y_val, _ = load_xy(val_path, feature_columns)
    X_test, y_test, test_df = load_xy(test_path, feature_columns)

    n_pos = int(np.sum(y_train >= 0.5))
    n_neg = int(np.sum(y_train < 0.5))
    scale_pos_weight = float(n_neg / n_pos) if n_pos > 0 else 1.0

    train_params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr", "logloss"],
        "max_depth": int(args.max_depth),
        "eta": float(args.learning_rate),
        "subsample": float(args.subsample),
        "colsample_bytree": float(args.colsample_bytree),
        "min_child_weight": float(args.min_child_weight),
        "lambda": float(args.reg_lambda),
        "scale_pos_weight": scale_pos_weight,
        "seed": int(args.seed),
        "tree_method": "hist",
    }

    model_path = model_dir / "model.json"
    config_path = model_dir / "train_config.json"
    best_iteration: Optional[int] = None

    if args.skip_train:
        if not model_path.is_file():
            raise FileNotFoundError(f"Missing model: {model_path}")
        booster = xgb.Booster()
        booster.load_model(str(model_path))
        if config_path.is_file():
            with config_path.open(encoding="utf-8") as f:
                cfg = json.load(f)
            best_iteration = cfg.get("best_iteration")
        if best_iteration is None:
            best_iteration = max(0, booster.num_boosted_rounds() - 1)
        print(f"Loaded {model_path} (trees={booster.num_boosted_rounds()}, best_iteration={best_iteration})")
    else:
        model_dir.mkdir(parents=True, exist_ok=True)
        booster = train_booster(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_columns,
            train_params,
            num_boost_round=int(args.num_boost_round),
            early_stopping_rounds=int(args.early_stopping_rounds),
        )
        bi = getattr(booster, "best_iteration", None)
        if bi is None:
            bi = booster.num_boosted_rounds() - 1
        best_iteration = int(bi)
        booster.save_model(str(model_path))
        save_json(
            {
                "val_fold": val_fold,
                "train_parquet": str(train_path),
                "val_parquet": str(val_path),
                "feature_columns": str(data_root / "vocab" / "feature_columns.json"),
                "train_params": train_params,
                "num_boost_round": int(args.num_boost_round),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "best_iteration": best_iteration,
                "scale_pos_weight": scale_pos_weight,
            },
            config_path,
        )
        print(f"Saved model -> {model_path} (best_iteration={best_iteration})")

    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)
    if best_iteration is not None and best_iteration >= 0:
        ntree = int(best_iteration) + 1
        try:
            y_prob = booster.predict(dtest, iteration_range=(0, ntree))
        except TypeError:
            y_prob = booster.predict(dtest, ntree_limit=ntree)
    else:
        y_prob = booster.predict(dtest)

    metrics = compute_metrics(y_test, y_prob, args.cls_threshold)
    json_path = write_baseline_metrics_json(
        eval_dir,
        metrics,
        evaluation="baseline_preoperative_aki_risk",
        val_fold=val_fold,
        model_dir=model_dir,
        data_root=data_root,
        test_parquet=test_path,
        episode_path=episode_path,
        cls_threshold=args.cls_threshold,
        train_params=train_params,
        best_iteration=best_iteration,
        checkpoint_path=model_path,
    )

    pred_path = eval_dir / "test_predictions.parquet"
    out_pred = test_df[["person_id", "label_aki"]].copy()
    out_pred["pred_prob"] = y_prob.astype(np.float64)
    out_pred.to_parquet(pred_path, index=False)

    if args.print_table:
        print_metrics_table(metrics, args.cls_threshold, "XGBoost preop AKI (test set)")

    print(f"Wrote {json_path}")
    print(f"Wrote {pred_path}")


if __name__ == "__main__":
    main()
