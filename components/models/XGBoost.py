from __future__ import annotations

try:
    from xgboost import XGBRegressor as _XGB
except ModuleNotFoundError:
    _XGB = None  # type: ignore


class XGBoost:
    """Thin wrapper exposing XGBRegressor when xgboost is installed."""

    def __new__(cls, *args, **kwargs):
        if _XGB is None:
            raise ModuleNotFoundError("xgboost library is not installed.")
        return _XGB(*args, **kwargs) 