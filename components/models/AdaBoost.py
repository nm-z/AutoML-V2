from __future__ import annotations

from sklearn.ensemble import AdaBoostRegressor as _AB


class AdaBoost(_AB):
    """Expose sklearn's AdaBoostRegressor under project namespace.""" 