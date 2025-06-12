from __future__ import annotations

from sklearn.preprocessing import RobustScaler as _Robust
from components.base import BaseTransformerBlock


class RobustScaler(_Robust):
    """Expose sklearn's RobustScaler under project namespace.""" 


class RobustScalerBlock(BaseTransformerBlock):
    """Project adapter for ``sklearn.preprocessing.RobustScaler``.

    This wrapper ensures that every scaler fits the *uniform* transformer
    interface required by the orchestrator while still delegating the heavy
    lifting to the battle-tested scikit-learn implementation.
    """

    # Static descriptor consumed by orchestrator during provenance logging
    signature = {
        "type": "preprocessor",
        "name": "RobustScaler",
        "hyperparameters": {},
    }

    def __init__(self, **kwargs):
        from sklearn.preprocessing import RobustScaler as _RobustScaler

        # Store kwargs for provenance; scikit-learn clones them internally
        self._params = kwargs.copy()
        self._impl = _RobustScaler(**kwargs)

    # ---------------------------------------------------------------------
    # scikit-learn compatible API
    # ---------------------------------------------------------------------
    def fit(self, X, y=None):  # noqa: D401, N802
        self._impl.fit(X, y)
        return self

    def transform(self, X):  # noqa: D401, N802
        return self._impl.transform(X)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def get_params(self, deep: bool = True):
        return self._impl.get_params(deep=deep)

    def set_params(self, **params):
        self._impl.set_params(**params)
        return self 