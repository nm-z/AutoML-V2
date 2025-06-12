from __future__ import annotations

from sklearn.neighbors import LocalOutlierFactor as _LOF


class LocalOutlierFactor(_LOF):
    """Expose sklearn's LocalOutlierFactor under project namespace.""" 