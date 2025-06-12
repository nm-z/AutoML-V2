from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, TransformerMixin


class KMeansOutlier(BaseEstimator, TransformerMixin):
    """Flag observations far from their nearest KMeans centroid.

    Simple transformer that appends a boolean mask column `is_outlier`.
    """

    def __init__(self, n_clusters: int = 2, min_cluster_size_ratio: float = 0.05, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: Any = None):  # noqa: D417
        self._kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self._kmeans.fit(X)
        return self

    def transform(self, X: np.ndarray):  # noqa: D401
        labels = self._kmeans.predict(X)
        counts = np.bincount(labels)
        small_clusters = {idx for idx, cnt in enumerate(counts) if cnt < self.min_cluster_size_ratio * len(X)}
        is_outlier = np.isin(labels, list(small_clusters)).astype(int)
        return np.hstack([X, is_outlier.reshape(-1, 1)]) 