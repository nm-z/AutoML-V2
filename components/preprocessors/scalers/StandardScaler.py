from __future__ import annotations

from sklearn.preprocessing import StandardScaler as _Std


class StandardScaler(_Std):
    """Expose sklearn's StandardScaler under project namespace.""" 