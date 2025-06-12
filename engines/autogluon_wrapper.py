"""autogluon_wrapper – Stub Implementation"""
from __future__ import annotations

from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree

console = Console(highlight=False)


def fit_engine(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_families: Sequence[str],
    prep_steps: Sequence[str],
    seed: int,
    timeout_sec: int,
) -> Any:  # noqa: ANN401
    """Fit AutoGluon TabularPredictor or fallback DummyRegressor."""

    root = Tree("[AutoGluon]")

    # Map our generic model names to AutoGluon's internal model keys
    autogluon_model_map = {
        "Ridge": "LR",
        "Lasso": "LR",
        "ElasticNet": "LR",
        "SVR": None,  # AutoGluon doesn't have a direct SVR equivalent in default models
        "DecisionTree": None,  # AutoGluon uses tree-based models, but not directly 'DecisionTree'
        "RandomForest": "RF",
        "ExtraTrees": "XT",
        "GradientBoosting": "GBM",  # LightGBM is often used for Gradient Boosting
        "AdaBoost": None,  # No direct AdaBoost model in default AutoGluon
        "MLP": "NN_TORCH",  # Multi-layer Perceptron
        "XGBoost": "XGB",
        "LightGBM": "GBM",
        "RPOP": None, # RPOP is not a standard AutoGluon model
    }

    ag_included_models = []
    for family in model_families:
        ag_key = autogluon_model_map.get(family)
        if ag_key:
            ag_included_models.append(ag_key)

    # AutoGluon handles preprocessing internally. We won't constrain it directly
    # based on prep_steps, as that would interfere with its automated feature engineering.
    # However, if we needed to disable certain types of features, we would use
    # the `feature_generator` argument in TabularPredictor.

    try:
        from autogluon.tabular import TabularPredictor  # type: ignore

        root.add("library detected – running real AutoGluon")
        train_data = X.copy()
        train_data["target"] = y
        predictor = TabularPredictor(
            label="target",
            problem_type="regression",
            eval_metric="r2",
        )
        predictor.fit(
            train_data=train_data,
            presets="lite",
            time_limit=timeout_sec,
            seed=seed,
            # Only include models explicitly requested by the orchestrator from our allowed list
            included_model_types=list(set(ag_included_models)) if ag_included_models else None, # Use set to remove duplicates
        )
        predictor.best_score_ = predictor.leaderboard(silent=True)["score_val"].iloc[0]  # type: ignore[attr-defined]
        model = predictor
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