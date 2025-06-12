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


def build_frozen_config(model_families: Sequence[str], prep_steps: Sequence[str]):
    """Return a TPOT‐compatible search‐space restricted to *approved* components.

    The orchestrator guarantees that *model_families* and *prep_steps* only
    contain names approved in the immutable spec (Tables 1 & 2).  We map those
    names to TPOT identifiers via *TPOT_COMPONENT_MAP* and construct an *empty*
    hyper-parameter dict for each entry – TPOT will then use its default search
    ranges for that primitive.  This keeps the search-space frozen to exactly
    the allowed components without manual tuning.
    """

    frozen: dict[str, dict] = {}
    for fam in model_families:
        tpot_name = TPOT_COMPONENT_MAP.get(fam)
        if tpot_name:
            frozen[tpot_name] = {}
    for prep in prep_steps:
        tpot_name = TPOT_COMPONENT_MAP.get(prep)
        if tpot_name:
            frozen[tpot_name] = {}
    return frozen


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

    custom_tpot_config = build_frozen_config(model_families, prep_steps)

    try:
        from tpot import TPOTRegressor  # type: ignore

        root.add("library detected – running real TPOTRegressor")

        tpot = TPOTRegressor(
            generations=100,
            population_size=100,
            config_dict=custom_tpot_config if custom_tpot_config else "TPOT light",
            early_stop=20,
            scoring="r2",
            n_jobs=1,  # Prevent nested parallelism – orchestrator handles outer level
            random_state=seed,
            max_time_mins=max(1, timeout_sec // 60),
            verbosity=2,
        )

        tpot.fit(X, y)

        model = tpot.fitted_pipeline_
        try:
            model.best_score_ = float(tpot._optimized_pipeline_score)  # type: ignore[attr-defined]
        except Exception:
            pass

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