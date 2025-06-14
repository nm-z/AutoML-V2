from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def engineer_features(X: pd.DataFrame) -> tuple[pd.DataFrame, Pipeline]:
    """Apply standard scaling and PCA to numeric columns.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.

    Returns
    -------
    Tuple[pd.DataFrame, Pipeline]
        Transformed features and the fitted pipeline.
    """
    numeric_cols = X.select_dtypes(include="number").columns
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ])
    X_transformed = pipeline.fit_transform(X[numeric_cols])
    X_fe = pd.DataFrame(
        X_transformed,
        index=X.index,
        columns=[f"pc{i+1}" for i in range(X_transformed.shape[1])],
    )
    # Preserve non-numeric columns, if any
    if len(numeric_cols) != X.shape[1]:
        X_non = X.drop(columns=numeric_cols)
        X_fe = pd.concat([X_non.reset_index(drop=True), X_fe.reset_index(drop=True)], axis=1)
    return X_fe, pipeline


__all__ = ["engineer_features"]
