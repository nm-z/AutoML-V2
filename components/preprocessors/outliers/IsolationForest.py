from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest as _SkIsolationForest

from components.base import BaseTransformerBlock


class IsolationForestBlock(BaseTransformerBlock, BaseEstimator):
    """Wrapper around ``sklearn.ensemble.IsolationForest`` that drops detected outliers."""

    signature = {
        "type": "preprocessor",
        "name": "IsolationForest",
        "hyperparameters": {
            "n_estimators": "int",
            "contamination": "float",
            "max_features": "float",
        },
    }

    def __init__(self, **kwargs):
        self._params = kwargs.copy()
        self._impl = _SkIsolationForest(**kwargs)
        self._mask: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None):
        self._impl.fit(X)
        self._mask = self._impl.predict(X) == 1  # 1 for inlier, -1 for outlier
        return self

    def transform(self, X: np.ndarray):
        if self._mask is None:
            # If not fitted transform acts as identity
            return X
        return X[self._mask]

    def get_support_mask(self) -> np.ndarray:  # noqa: D401
        if self._mask is None:
            raise RuntimeError("IsolationForestBlock has not been fitted yet.")
        return self._mask 
