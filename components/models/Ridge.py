from __future__ import annotations

from sklearn.linear_model import Ridge as _Ridge


class Ridge(_Ridge):
    """Expose sklearn's Ridge under project namespace.""" 