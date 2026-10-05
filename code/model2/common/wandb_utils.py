"""Safe wandb import (NumPy 2.x + old wandb) and run initialization."""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from common._numpy_legacy_patch import patch_numpy_legacy_aliases


def patch_numpy_for_legacy_wandb() -> None:
    """Older wandb releases reference removed NumPy 2.0 aliases."""
    patch_numpy_legacy_aliases()


def wandb_executable_path() -> Path:
    return Path(__file__).resolve().parent / "wandb_executable.sh"


def configure_wandb_subprocess() -> str:
    """Point wandb's service subprocess at a NumPy-patched Python entrypoint."""
    exe = str(wandb_executable_path())
    os.environ["WANDB_REAL_PYTHON"] = os.environ.get("PYTHON", "") or sys.executable
    os.environ["WANDB__EXECUTABLE"] = exe
    return exe


def safe_import_wandb() -> Optional[Any]:
    try:
        patch_numpy_for_legacy_wandb()
        import wandb

        return wandb
    except Exception as exc:
        warnings.warn(f"wandb unavailable; continuing without W&B logging: {exc}", stacklevel=2)
        return None


def init_wandb_run(
    *,
    project: str,
    entity: Optional[str],
    name: str,
    tags: Optional[List[str]],
    config: Dict[str, Any],
    dir: Path,
) -> Tuple[Optional[Any], Optional[Any]]:
    wandb_mod = safe_import_wandb()
    if wandb_mod is None:
        return None, None
    try:
        executable = configure_wandb_subprocess()
        run = wandb_mod.init(
            project=project,
            entity=entity,
            name=name,
            tags=tags,
            config=config,
            dir=str(dir),
            settings=wandb_mod.Settings(_executable=executable),
        )
        return run, wandb_mod
    except Exception as exc:
        warnings.warn(f"wandb.init failed; continuing without W&B logging: {exc}", stacklevel=2)
        return None, None


def log_wandb_metrics(
    wandb_mod: Optional[Any],
    wandb_run: Optional[Any],
    metrics: Dict[str, Any],
    *,
    step: int,
) -> None:
    if wandb_run is None or wandb_mod is None:
        return
    wandb_mod.log(metrics, step=step)


def update_wandb_summary(
    wandb_mod: Optional[Any],
    wandb_run: Optional[Any],
    summary: Dict[str, Any],
) -> None:
    if wandb_run is None or wandb_mod is None:
        return
    for key, value in summary.items():
        wandb_mod.run.summary[key] = value


def finish_wandb(wandb_mod: Optional[Any], wandb_run: Optional[Any]) -> None:
    if wandb_run is None or wandb_mod is None:
        return
    wandb_mod.finish()
