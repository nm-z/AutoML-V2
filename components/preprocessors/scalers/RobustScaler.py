from __future__ import annotations

from sklearn.preprocessing import RobustScaler as _Robust


class RobustScaler(_Robust):
    """Expose sklearn's RobustScaler under project namespace.""" 