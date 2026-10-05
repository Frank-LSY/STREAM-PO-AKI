"""Repository root used to resolve data/ and result/ paths.

Defaults to the repository checkout; set PERIOP_ROOT to point at a different working tree.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("PERIOP_ROOT", Path(__file__).resolve().parents[3]))
