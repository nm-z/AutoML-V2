from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor as _GB


class GradientBoosting(_GB):
    """Expose sklearn's GradientBoostingRegressor under project namespace.""" 