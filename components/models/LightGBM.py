from __future__ import annotations

try:
    from lightgbm import LGBMRegressor as _LGBM
except ModuleNotFoundError:
    _LGBM = None  # type: ignore


class LightGBM:
    """Thin wrapper exposing LGBMRegressor when lightgbm is installed."""

    def __new__(cls, *args, **kwargs):
        if _LGBM is None:
            raise ModuleNotFoundError("lightgbm library is not installed.")
        return _LGBM(*args, **kwargs) 