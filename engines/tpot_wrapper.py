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


def build_frozen_config(model_families: Sequence[str], _prep_steps: Sequence[str]):
    """Return a TPOT *config_dict* limited to supported Regressors & Transformers.

    IMPORTANT:
        •  TPOT validates every entry against a whitelist of operator *types*.
           Passing an unsupported estimator (e.g. unsupervised outlier models)
           triggers ``ValueError: optype must be one of …`` during runtime.

        •  Therefore we *only* forward components that TPOT can positively
           identify as either *Regressor* or *Transformer*.  At the moment this
           includes all **model families** plus the benign scaling / PCA steps.

    The orchestrator still sees the full search-space – TPOT merely ignores the
    risky entries that violate its operator registry.
    """

    SAFE_PREPROCESSORS = {  # Plain feature transformers accepted by TPOT
        "PCA",
        "RobustScaler",
        "StandardScaler",
        "QuantileTransform",
    }

    frozen: dict[str, dict] = {}

    # 1️⃣ Add *Regressor* primitives (model families)
    for fam in model_families:
        tpot_name = TPOT_COMPONENT_MAP.get(fam)
        if tpot_name:
            frozen[tpot_name] = {}

    # 2️⃣ Add only *safe* pre-processing primitives that TPOT classifies as
    #     Transformer.  Skip outlier detectors (IsolationForest, LOF, …) because
    #     they do **not** satisfy TPOT's operator-type constraints.
    for prep in _prep_steps:
        if prep not in SAFE_PREPROCESSORS:
            continue
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