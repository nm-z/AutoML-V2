from __future__ import annotations

from components.base import BaseTransformerBlock


class QuantileTransformBlock(BaseTransformerBlock):
    """Adapter for ``sklearn.preprocessing.QuantileTransformer``."""

    signature = {
        "type": "preprocessor",
        "name": "QuantileTransform",
        "hyperparameters": {},
    }

    def __init__(self, **kwargs):
        from sklearn.preprocessing import QuantileTransformer as _QT

        self._params = kwargs.copy()
        self._impl = _QT(**kwargs)

    def fit(self, X, y=None):
        self._impl.fit(X, y)
        return self

    def transform(self, X):
        return self._impl.transform(X)

    def get_params(self, deep: bool = True):
        return self._impl.get_params(deep=deep)

    def set_params(self, **params):
        self._impl.set_params(**params)
        return self 
