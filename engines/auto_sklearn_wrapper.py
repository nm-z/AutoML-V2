"""auto_sklearn_wrapper – Stub Implementation

This placeholder satisfies the orchestrator while the full wrapper is under
construction.  It *tries* to run AutoSklearn if the library is present; if
not, it falls back to a simple baseline that at least returns a fitted model
with a ``best_score_`` attribute so downstream code can proceed.
"""
from __future__ import annotations

import random
from typing import Any, Sequence

import numpy as np
import pandas as pd
from rich.console import Console
from rich.tree import Tree

console = Console(highlight=False)


# ---------------------------------------------------------------------------
# Public API expected by orchestrator
# ---------------------------------------------------------------------------

def fit_engine(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_families: Sequence[str],
    prep_steps: Sequence[str],
    seed: int,
    timeout_sec: int,
) -> Any:  # noqa: ANN401 – unknown model object type until real impl.
    """Fit an AutoSklearnRegressor constrained to the approved search-space.

    The current implementation is a *stub*; it will either:
    1.  Use ``autosklearn.regression.AutoSklearnRegressor`` if available, or
    2.  Fit a trivial baseline model that predicts the mean.
    """

    root = Tree("[AutoSklearn]")
    rng = random.Random(seed)

    # Map generic names to auto-sklearn internal names. This is a placeholder
    # and should be expanded with a more comprehensive mapping if needed.
    # For now, only a few exact matches are supported for demonstration.
    as_estimators = []
    for family in model_families:
        # Simple direct mapping for now, assuming auto-sklearn uses similar names
        # In a real scenario, this would be a more robust dictionary lookup
        if family == "LinearRegression":
            as_estimators.append("linear_regression")
        elif family == "RandomForest":
            as_estimators.append("random_forest")
        # Add more mappings as needed

    as_preprocessors = []
    for prep in prep_steps:
        # Simple direct mapping for now
        if prep == "PCA":
            as_preprocessors.append("pca")
        elif prep == "StandardScaler":
            as_preprocessors.append("standard_scaler")
        # Add more mappings

    try:
        from autosklearn.regression import AutoSklearnRegressor  # type: ignore

        root.add("library detected – running real AutoSklearn")
        model = AutoSklearnRegressor(
            time_left_for_this_task=timeout_sec,
            seed=seed,
            n_jobs=1,
            metric="r2",
            include_estimators=as_estimators if as_estimators else None,
            include_preprocessors=as_preprocessors if as_preprocessors else None,
        )
        model.fit(X, y)
    except ModuleNotFoundError:
        root.add("library missing – falling back to LinearRegression")
        from sklearn.linear_model import LinearRegression  # type: ignore

        model = LinearRegression(n_jobs=1)
        model.fit(X, y)
        # Attach best_score_ on training data (since no CV here)
        preds = model.predict(X)
        ss_res = ((y - preds) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        model.best_score_ = 1 - ss_res / ss_tot  # type: ignore[attr-defined]

    console.print(root)
    return model

__all__ = ["fit_engine"] 