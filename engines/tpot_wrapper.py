"""tpot_wrapper – Stub Implementation"""
from __future__ import annotations

import random
from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree

console = Console(highlight=False)

# Map generic names to TPOT internal names. This mapping should be comprehensive
# and cover all models and preprocessors defined in config.py that TPOT can use.
# For now, a representative subset is provided.
TPOT_COMPONENT_MAP = {
    # Models
    "Ridge": "sklearn.linear_model.Ridge",
    "Lasso": "sklearn.linear_model.Lasso",
    "ElasticNet": "sklearn.linear_model.ElasticNet",
    "SVR": "sklearn.svm.SVR",
    "DecisionTree": "sklearn.tree.DecisionTreeRegressor",
    "RandomForest": "sklearn.ensemble.RandomForestRegressor",
    "ExtraTrees": "sklearn.ensemble.ExtraTreesRegressor",
    "GradientBoosting": "sklearn.ensemble.GradientBoostingRegressor",
    "AdaBoost": "sklearn.ensemble.AdaBoostRegressor",
    "MLP": "sklearn.neural_network.MLPRegressor",
    "XGBoost": "xgboost.XGBRegressor",
    "LightGBM": "lightgbm.LGBMRegressor",
    # Preprocessors
    "PCA": "sklearn.decomposition.PCA",
    "RobustScaler": "sklearn.preprocessing.RobustScaler",
    "StandardScaler": "sklearn.preprocessing.StandardScaler",
    "QuantileTransform": "sklearn.preprocessing.QuantileTransformer",
    "KMeansOutlier": "sklearn.cluster.MiniBatchKMeans",  # Example mapping, adjust based on how TPOT uses outliers
    "IsolationForest": "sklearn.ensemble.IsolationForest",
    "LocalOutlierFactor": "sklearn.neighbors.LocalOutlierFactor",
}


def fit_engine(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    model_families: Sequence[str],
    prep_steps: Sequence[str],
    seed: int,
    timeout_sec: int,
) -> Any:  # noqa: ANN401
    """Fit TPOTRegressor or fall back to DummyRegressor."""

    root = Tree("[TPOT]")

    custom_tpot_config = {}
    for family in model_families:
        tpot_name = TPOT_COMPONENT_MAP.get(family)
        if tpot_name:
            custom_tpot_config[tpot_name] = {}  # Use empty dict for default TPOT search space

    for prep in prep_steps:
        tpot_name = TPOT_COMPONENT_MAP.get(prep)
        if tpot_name:
            custom_tpot_config[tpot_name] = {}  # Use empty dict for default TPOT search space

    try:
        from tpot import TPOTRegressor  # type: ignore

        root.add("library detected – running real TPOT")
        model = TPOTRegressor(
            generations=50,
            population_size=50,
            verbosity=2,
            random_state=seed,
            n_jobs=1,
            max_time_mins=timeout_sec / 60,
            scoring="r2",
            config_dict=custom_tpot_config, # Pass the custom config
        )
        model.fit(X, y)
        # TPOT attaches score_ after fit
        model.best_score_ = model.score(X, y)  # type: ignore[attr-defined]
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