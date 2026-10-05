"""NumPy 2.x compatibility aliases for packages that still use removed names."""
from __future__ import annotations

import numpy as np

_ALIASES = {
    "float_": np.float64,
    "int_": np.int64,
    "bool_": np.bool_,
    "object_": np.object_,
    "str_": np.str_,
    "complex_": np.complex128,
    "long": np.int64,
    "unicode_": np.str_,
}


def patch_numpy_legacy_aliases() -> None:
    for name, target in _ALIASES.items():
        if not hasattr(np, name):
            setattr(np, name, target)


patch_numpy_legacy_aliases()
