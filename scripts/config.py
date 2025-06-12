"""AutoML Orchestrator – Global Configuration

This module centralises every *immutable* knob that governs the AutoML
search-space and execution behaviour.  **Never** import these constants
into a function just to mutate them.
"""
from __future__ import annotations

from typing import Tuple

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_STATE: int = 42  # Global seed – applies to every RNG-capable component

# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------
# n_jobs for outer CV / AutoML engines – prevents nested parallel explosions
N_JOBS_CV: int = 12  # Top-level CV or AutoML parallelism
# n_jobs to be forwarded into *individual* models/transformers
N_JOBS_MODEL: int = 1

# ---------------------------------------------------------------------------
# Search-Space Definition (Tables 1 & 2)
# ---------------------------------------------------------------------------
# MODEL_FAMILIES: Tuple[str, ...] = (
#     "Ridge",
#     "Lasso",
#     "ElasticNet",
#     "SVR",
#     "DecisionTree",
#     "RandomForest",
#     "ExtraTrees",
#     "GradientBoosting",
#     "AdaBoost",
#     "MLP",
#     "XGBoost",
#     "LightGBM",
# )
MODEL_FAMILIES: Tuple[str, ...] = tuple(_MODEL_SPACE.keys())

# PREP_STEPS: Tuple[str, ...] = (
#     "PCA",
#     "RobustScaler",
#     "StandardScaler",
#     "KMeansOutlier",
#     "IsolationForest",
#     "LocalOutlierFactor",
#     "QuantileTransform",
# )
PREP_STEPS: Tuple[str, ...] = tuple(_PREPROCESSOR_SPACE.keys())

# ---------------------------------------------------------------------------
# Engine Budgets
# ---------------------------------------------------------------------------
WALLCLOCK_LIMIT_SEC: int = 3_600  # default per AutoML engine

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEFAULT_METRIC: str = "r2"  # Do *not* hard-code a target – metric only.

# ---------------------------------------------------------------------------
# Search-Space – concrete hyper-parameter grids (immutable)
# ---------------------------------------------------------------------------
# NOTE: These ranges are intentionally **narrow** and free of manual tuning –
# they merely expose *representative* values so that each engine can exercise
# every primitive without exploding the Cartesian search volume.  Update with
# care: any change here alters the deterministic search-space guaranteed by
# the spec.

_MODEL_SPACE: dict[str, dict] = {
    "Ridge": {"alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"alpha": [0.001, 0.01, 0.1, 1.0], "max_iter": [2000]},
    "ElasticNet": {
        "alpha": [0.001, 0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.5, 0.9],
        "max_iter": [2000],
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "gamma": ["scale", "auto"],
        "kernel": ["rbf", "poly"],
    },
    "DecisionTree": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
    },
    "RandomForest": {
        "n_estimators": [100, 300, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "max_features": ["sqrt", "log2"],
    },
    "ExtraTrees": {
        "n_estimators": [100, 300],
        "max_depth": [None, 10, 20],
    },
    "GradientBoosting": {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5],
    },
    "AdaBoost": {"n_estimators": [50, 100, 200], "learning_rate": [0.05, 0.1, 0.2]},
    "MLP": {
        "hidden_layer_sizes": [(50,), (100,), (100, 50)],
        "activation": ["relu", "tanh"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001],
        "learning_rate": ["adaptive"],
    },
    "XGBoost": {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [100, 300],
        "learning_rate": [0.05, 0.1, 0.2],
        "num_leaves": [31, 63, 127],
        "max_depth": [-1, 10],
    },
    "MyNewModel": {
        "parameter1": ["a", "b", "c"],
        "parameter2": [10, 20, 30],
    },
}

_PREPROCESSOR_SPACE: dict[str, dict] = {
    "PCA": {"n_components": list(range(5, 55, 5))},
    "RobustScaler": {"quantile_range": [[10.0, 90.0], [25.0, 75.0]]},
    "StandardScaler": {},
    "KMeansOutlier": {"n_clusters": [3, 5, 8]},
    "IsolationForest": {
        "n_estimators": [100, 200],
        "contamination": [0.05, 0.1],
        "max_features": [0.5, 1.0],
    },
    "LocalOutlierFactor": {"n_neighbors": [20, 35], "contamination": [0.05, 0.1]},
    "QuantileTransform": {"output_distribution": ["uniform", "normal"]},
}

# ---------------------------------------------------------------------------
# Helper – expose hyper-parameter grids for orchestrator / wrappers
# ---------------------------------------------------------------------------

def get_space(kind: str):
    """Return the approved hyper-parameter search-space for *kind*.

    Parameters
    ----------
    kind
        Either ``"model"`` or ``"preprocessor"``.

    Returns
    -------
    dict
        A mapping suitable for AutoML engine consumption.  This function
        returns the detailed hyper-parameter grids from Tables 1 & 2.
    """
    if kind == "model":
        return _MODEL_SPACE.copy()
    elif kind == "preprocessor":
        return _PREPROCESSOR_SPACE.copy()
    else:
        raise ValueError("kind must be 'model' or 'preprocessor'")

__all__ = [
    "RANDOM_STATE",
    "N_JOBS_CV",
    "N_JOBS_MODEL",
    "MODEL_FAMILIES",
    "PREP_STEPS",
    "WALLCLOCK_LIMIT_SEC",
    "DEFAULT_METRIC",
    "get_space",
] 