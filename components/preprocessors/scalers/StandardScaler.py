from __future__ import annotations

from components.base import BaseTransformerBlock


class StandardScalerBlock(BaseTransformerBlock):
    """Adapter for ``sklearn.preprocessing.StandardScaler``."""

    signature = {
        "type": "preprocessor",
        "name": "StandardScaler",
        "hyperparameters": {},
    }

    def __init__(self, **kwargs):
        from sklearn.preprocessing import StandardScaler as _StandardScaler

        self._params = kwargs.copy()
        self._impl = _StandardScaler(**kwargs)

    # API -----------------------------------------------------------------
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