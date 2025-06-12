"""AutoML Engine Discovery

This package exposes utilities to discover and import whichever AutoML
engine wrappers are actually available in the runtime environment.  The
wrappers themselves live alongside this module (e.g. ``auto_sklearn_wrapper.py``).
"""
from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict, List

# ---------------------------------------------------------------------------
# The *canonical* order we will attempt to use the engines.  Earlier entries
# get first dibs at the allocated wall-clock budget.
# ---------------------------------------------------------------------------
_ENGINE_ORDER: List[str] = [
    "auto_sklearn_wrapper",
    "tpot_wrapper",
    "autogluon_wrapper",
]


def _import_wrapper(module_basename: str) -> ModuleType | None:
    """Best-effort import of a single wrapper module inside ``engines``.

    We *only* catch ``ModuleNotFoundError`` because any other exception means
    the module existed but raised at import-time, in which case the user must
    fix the failure explicitly (Fail-Fast-On-Errors rule).
    """
    try:
        return importlib.import_module(f"{__name__}.{module_basename}")
    except ModuleNotFoundError:
        return None


def discover_available() -> Dict[str, ModuleType]:
    """Return a mapping of ``{engine_name: wrapper_module}`` for those wrappers
    that can be successfully imported on *this* machine.
    """
    available: Dict[str, ModuleType] = {}
    for mod in _ENGINE_ORDER:
        wrapper = _import_wrapper(mod)
        if wrapper is not None:
            available[mod] = wrapper
    return available

__all__ = ["discover_available"] 