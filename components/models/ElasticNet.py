from __future__ import annotations

from sklearn.linear_model import ElasticNet as _EN


class ElasticNet(_EN):
    """Expose sklearn's ElasticNet under project namespace.""" 