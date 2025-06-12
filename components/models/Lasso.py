from __future__ import annotations

from sklearn.linear_model import Lasso as _Lasso


class Lasso(_Lasso):
    """Expose sklearn's Lasso under project namespace.""" 