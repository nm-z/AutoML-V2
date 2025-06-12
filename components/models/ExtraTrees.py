from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor as _ET


class ExtraTrees(_ET):
    """Expose sklearn's ExtraTreesRegressor under project namespace.""" 