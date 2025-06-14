from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline


def engineer_features(X: pd.DataFrame, y: pd.Series | None = None) -> tuple[pd.DataFrame, Pipeline]:
    """Apply lightweight feature engineering and dimensionality reduction.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    y : pd.Series | None, optional
        Target vector for target encoding of categorical variables.

    Returns
    -------
    Tuple[pd.DataFrame, Pipeline]
        Transformed features and the fitted pipeline.
    """
    df = X.copy()

    # ------------------------------------------------------------------
    # Target Encoding for Categorical Columns
    # ------------------------------------------------------------------
    if y is not None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        global_mean = y.mean()
        for col in cat_cols:
            means = y.groupby(df[col]).mean()
            df[col] = df[col].map(means).fillna(global_mean)

    # ------------------------------------------------------------------
    # Interaction Terms (first 3 numeric columns)
    # ------------------------------------------------------------------
    num_cols = df.select_dtypes(include="number").columns
    top_numeric = list(num_cols)[:3]
    for i, col1 in enumerate(top_numeric):
        for col2 in top_numeric[i + 1 :]:
            df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

    numeric_cols = df.select_dtypes(include="number").columns
    pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=0.95)),
    ])
    X_transformed = pipeline.fit_transform(df[numeric_cols])
    X_fe = pd.DataFrame(
        X_transformed,
        index=df.index,
        columns=[f"pc{i+1}" for i in range(X_transformed.shape[1])],
    )
    if len(numeric_cols) != df.shape[1]:
        X_non = df.drop(columns=numeric_cols).reset_index(drop=True)
        X_fe = pd.concat([X_non, X_fe.reset_index(drop=True)], axis=1)
    return X_fe, pipeline


__all__ = ["engineer_features"]
