from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor as _SkLOF

from components.base import BaseTransformerBlock


class LOFBlock(BaseTransformerBlock, BaseEstimator):
    """Local Outlier Factor based sample pruning."""

    signature = {
        "type": "preprocessor",
        "name": "LocalOutlierFactor",
        "hyperparameters": {
            "n_neighbors": "int",
            "contamination": "float",
        },
    }

    def __init__(self, **kwargs):
        # LOF does not support `predict` unless novelty=True; we use fit_predict.
        self._params = kwargs.copy()
        # Force novelty=False to allow fit_predict (on training data) but we only care train set
        self._impl = _SkLOF(**kwargs)
        self._mask: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: Any = None):
        labels = self._impl.fit_predict(X)
        self._mask = labels == 1  # 1 inlier, -1 outlier
        return self

    def transform(self, X: np.ndarray):
        if self._mask is None:
            return X
        return X[self._mask]

    def get_support_mask(self) -> np.ndarray:  # noqa: D401
        if self._mask is None:
            raise RuntimeError("LOFBlock has not been fitted yet.")
        return self._mask 
