from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor as _RF


class RandomForest(_RF):
    """Expose sklearn's RandomForestRegressor under project namespace.""" 