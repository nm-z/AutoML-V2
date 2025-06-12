from __future__ import annotations

from sklearn.neural_network import MLPRegressor as _MLP


class MLP(_MLP):
    """Expose sklearn's MLPRegressor under project namespace.""" 