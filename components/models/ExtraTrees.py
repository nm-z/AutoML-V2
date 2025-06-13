from __future__ import annotations

from typing import Any

from components.base import BaseEstimatorBlock


class ExtraTreesBlock(BaseEstimatorBlock):
    """Adapter for ``sklearn.ensemble.ExtraTreesRegressor``."""

    signature = {
        "type": "model",
        "name": "ExtraTrees",
        "hyperparameters": {
            "n_estimators": "int",
            "max_depth": "int_or_none",
        },
    }

    def __init__(self, **kwargs):
        from sklearn.ensemble import ExtraTreesRegressor as _ETR

        self._params = kwargs.copy()
        self._impl = _ETR(random_state=42, **kwargs)

    def fit(self, X, y):
        self._impl.fit(X, y)
        return self

    def predict(self, X):
        return self._impl.predict(X)

    def score(self, X, y, sample_weight: Any | None = None):
        return self._impl.score(X, y, sample_weight=sample_weight)

    def get_params(self, deep: bool = True):
        return self._impl.get_params(deep=deep)

    def set_params(self, **params):
        self._impl.set_params(**params)
        return self 
