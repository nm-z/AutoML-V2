from __future__ import annotations

from sklearn.decomposition import PCA as _PCA


class PCA(_PCA):
    """Expose sklearn's PCA under project namespace.""" 