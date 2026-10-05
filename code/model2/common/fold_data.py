"""Shared JSONL loading and fold path helpers."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

FOLD_FILENAME_TEMPLATE = "7d_window_clm.mixed.fold{}.jsonl"
TEST_FILENAME = "7d_window_clm.mixed.test.jsonl"


def model2_root() -> Path:
    """``model2/`` directory (parent of ``common/``)."""
    return Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    """Canonical JSONL folds root (``scripts/common.sh`` ``DATA_DIR``)."""
    env = os.environ.get("DATA_DIR")
    if env:
        return Path(env).resolve()
    return (model2_root() / ".." / ".." / "data" / "five_folds_deduplicate4").resolve()


def default_test_jsonl(data_dir: Path | None = None) -> Path:
    return (data_dir or default_data_dir()) / TEST_FILENAME


def default_xgb_data_dir() -> Path:
    env = os.environ.get("XGB_DATA_DIR")
    if env:
        return Path(env).resolve()
    return default_data_dir().parent / "five_folds_xgboost4"


def default_ts6h_data_dir() -> Path:
    env = os.environ.get("TS6H_DATA_DIR")
    if env:
        return Path(env).resolve()
    return default_data_dir().parent / "five_folds_ts6h4"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def resolve_fold_paths(folds_dir: Path, template: str) -> Dict[int, Path]:
    return {k: folds_dir / template.format(k) for k in range(1, 6)}
