#!/usr/bin/env python
"""Verify Auto-Sklearn installation.

Run this after activating the automl-py310 environment.
It prints the installed version or an error message if the module
is missing.
"""
from __future__ import annotations
import importlib
import sys


def main() -> None:
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
    if sys.version_info < (3, 10):
        print(f"\u2717 Python {py_ver} is too old for Auto-Sklearn")
        sys.exit(1)

    try:
        autosklearn = importlib.import_module("autosklearn")
        print(f"\u2713 Auto-Sklearn {autosklearn.__version__} detected under Python {py_ver}")
    except Exception as exc:  # noqa: BLE001
        print(f"\u2717 Auto-Sklearn import failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()

