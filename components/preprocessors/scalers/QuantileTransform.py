from __future__ import annotations

from sklearn.preprocessing import QuantileTransformer as _QT


class QuantileTransform(_QT):
    """Expose sklearn's QuantileTransformer under project namespace.""" 