from __future__ import annotations

from typing import Any

from components.base import BaseEstimatorBlock


class RandomForestBlock(BaseEstimatorBlock):
    """Adapter for ``sklearn.ensemble.RandomForestRegressor``."""

    signature = {
        "type": "model",
        "name": "RandomForest",
        "hyperparameters": {
            "n_estimators": "int",
            "max_depth": "int_or_none",
            "min_samples_split": "int",
            "max_features": "str",
        },
    }

    def __init__(self, **kwargs):
        from sklearn.ensemble import RandomForestRegressor as _RFR

        self._params = kwargs.copy()
        self._impl = _RFR(random_state=42, **kwargs)

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