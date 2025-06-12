from __future__ import annotations

from typing import Any

from components.base import BaseEstimatorBlock


class ElasticNetBlock(BaseEstimatorBlock):
    """Adapter for ``sklearn.linear_model.ElasticNet``."""

    signature = {
        "type": "model",
        "name": "ElasticNet",
        "hyperparameters": {
            "alpha": "float",
            "l1_ratio": "float",
            "max_iter": "int",
        },
    }

    def __init__(self, **kwargs):
        from sklearn.linear_model import ElasticNet as _EN

        self._params = kwargs.copy()
        self._impl = _EN(**kwargs)

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