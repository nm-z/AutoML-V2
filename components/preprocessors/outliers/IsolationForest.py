from __future__ import annotations

from sklearn.ensemble import IsolationForest as _IF


class IsolationForest(_IF):
    """Expose sklearn's IsolationForest under project namespace.""" 