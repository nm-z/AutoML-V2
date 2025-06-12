from __future__ import annotations

"""auto_sklearn_wrapper – Light wrapper around AutoSklearnRegressor

This adapter conforms to the orchestrator's strict API contract:
::
    fit_engine(X, y, *, model_families, prep_steps, seed, timeout_sec)

It restricts the allowed estimator pool to the immutable search-space defined
in :pymod:`scripts.config` while delegating hyper-parameter optimisation to
Auto-Sklearn itself.
"""

from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree

console = Console(highlight=False)

# ---------------------------------------------------------------------------
# Mapping – project generic names → Auto-Sklearn estimator identifiers
# ---------------------------------------------------------------------------
_AUTOSKLEARN_MAP = {
    # Models
    "Ridge": "ridge_regression",
    "Lasso": "lasso_regression",
    "ElasticNet": "elasticnet",
    "SVR": "svr",
    "DecisionTree": "decision_tree",
    "RandomForest": "random_forest",
    "ExtraTrees": "extra_trees",
    "GradientBoosting": "gradient_boosting",
    "AdaBoost": "adaboost",
    "MLP": "mlp",
    "XGBoost": "xgradient_boosting",
    "LightGBM": "lightgbm",
    # RPOP has no Auto-Sklearn equivalent – omit gracefully
}


def _build_include_list(families: Sequence[str]):
    """Return a *distinct* list of Auto-Sklearn estimator identifiers."""

    include = [
        _AUTOSKLEARN_MAP[f]
        for f in families
        if f in _AUTOSKLEARN_MAP and _AUTOSKLEARN_MAP[f] is not None
    ]
    # Auto-Sklearn expects a list – duplicates hurt but we remove them anyway
    return list(dict.fromkeys(include))


def fit_engine(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_families: Sequence[str],
    prep_steps: Sequence[str],  # noqa: ARG001 – reserved for future use
    seed: int,
    timeout_sec: int,
) -> Any:  # noqa: ANN401 – polymorphic return type
    """Fit AutoSklearnRegressor or fall back to LinearRegression."""

    root = Tree("[Auto-Sklearn]")

    include_estimators = _build_include_list(model_families)

    try:
        from autosklearn.regression import AutoSklearnRegressor  # type: ignore

        root.add("library detected – running real Auto-Sklearn")

        automl = AutoSklearnRegressor(
            time_left_for_this_task=timeout_sec,
            per_run_time_limit=min(900, max(30, timeout_sec // 10)),
            include_estimators=include_estimators if include_estimators else None,
            resampling_strategy="holdout",
            resampling_strategy_arguments={"train_size": 0.75},
            metric="r2",
            n_jobs=1,  # prevent nested parallelism
            seed=seed,
        )

        automl.fit(X, y)
        model = automl

    except ModuleNotFoundError:
        root.add("library missing – fallback LinearRegression")

        from sklearn.linear_model import LinearRegression  # type: ignore

        linreg = LinearRegression(n_jobs=1)
        linreg.fit(X, y)
        preds = linreg.predict(X)
        ss_res = ((y - preds) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        linreg.best_score_ = 1 - ss_res / ss_tot  # type: ignore[attr-defined]
        model = linreg

    console.print(root)
    return model

__all__ = ["fit_engine"] 