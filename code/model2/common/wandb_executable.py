#!/usr/bin/env python3
"""W&B service subprocess logic (invoked via wandb_executable.sh)."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

_MODEL2_ROOT = Path(__file__).resolve().parent.parent
if str(_MODEL2_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL2_ROOT))

import common._numpy_legacy_patch  # noqa: F401


def main() -> None:
    argv = sys.argv[1:]
    if len(argv) >= 2 and argv[0] == "-m":
        module = argv[1]
        sys.argv = [module, *argv[2:]]
        runpy.run_module(module, run_name="__main__", alter_sys=True)
        return

    real_python = os.environ.get("WANDB_REAL_PYTHON") or sys.executable
    os.execv(real_python, [real_python, *argv])


if __name__ == "__main__":
    main()
