from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans

from components.base import BaseTransformerBlock


class KMeansOutlierBlock(BaseTransformerBlock, BaseEstimator):
    """Remove observations belonging to *tiny* K-means clusters.

    The heuristic assumes that clusters representing fewer than
    ``min_cluster_size_ratio * n_samples`` are anomalies and should be purged
    prior to model training.
    """

    signature = {
        "type": "preprocessor",
        "name": "KMeansOutlier",
        "hyperparameters": {
            "n_clusters": "int",
            "min_cluster_size_ratio": "float",
        },
    }

    def __init__(self, n_clusters: int = 2, min_cluster_size_ratio: float = 0.05, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.min_cluster_size_ratio = min_cluster_size_ratio
        self.random_state = random_state

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: Any = None):  # noqa: D417
        self._impl = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        self._impl.fit(X)
        self._labels = self._impl.labels_
        return self

    def transform(self, X: np.ndarray):  # noqa: D401
        labels = getattr(self, "_labels", self._impl.predict(X))
        counts = np.bincount(labels)
        small_clusters = {idx for idx, cnt in enumerate(counts) if cnt < self.min_cluster_size_ratio * len(X)}
        mask = ~np.isin(labels, list(small_clusters))
        # Drop rows corresponding to small clusters
        return X[mask]

    # Expose mask for downstream components if needed
    def get_support_mask(self) -> np.ndarray:  # noqa: D401
        """Return boolean mask of retained samples after `.fit()`."""
        labels = self._labels
        counts = np.bincount(labels)
        small_clusters = {idx for idx, cnt in enumerate(counts) if cnt < self.min_cluster_size_ratio * len(labels)}
        return ~np.isin(labels, list(small_clusters)) 
