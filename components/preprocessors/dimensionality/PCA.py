from __future__ import annotations

from components.base import BaseTransformerBlock


class PCABlock(BaseTransformerBlock):
    """Adapter for ``sklearn.decomposition.PCA`` with project-standard API."""

    signature = {
        "type": "preprocessor",
        "name": "PCA",
        "hyperparameters": {},
    }

    def __init__(self, **kwargs):
        from sklearn.decomposition import PCA as _skPCA

        self._params = kwargs.copy()
        self._impl = _skPCA(**kwargs)

    def fit(self, X, y=None):
        self._impl.fit(X, y)
        return self

    def transform(self, X):
        return self._impl.transform(X)

    def fit_transform(self, X, y=None):
        return self._impl.fit_transform(X, y)

    def get_params(self, deep: bool = True):
        return self._impl.get_params(deep=deep)

    def set_params(self, **params):
        self._impl.set_params(**params)
        return self

    @property
    def explained_variance_ratio_(self):
        return self._impl.explained_variance_ratio_ 