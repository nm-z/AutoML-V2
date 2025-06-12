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
MODEL_FAMILIES: Tuple[str, ...] = (
    "Ridge",
    "RPOP",
    "Lasso",
    "ElasticNet",
    "SVR",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "GradientBoosting",
    "AdaBoost",
    "MLP",
    "XGBoost",
    "LightGBM",
)

PREP_STEPS: Tuple[str, ...] = (
    "PCA",
    "RobustScaler",
    "StandardScaler",
    "KMeansOutlier",
    "IsolationForest",
    "LocalOutlierFactor",
    "QuantileTransform",
)

# ---------------------------------------------------------------------------
# Engine Budgets
# ---------------------------------------------------------------------------
WALLCLOCK_LIMIT_SEC: int = 3_600  # default per AutoML engine

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
DEFAULT_METRIC: str = "r2"  # Do *not* hard-code a target – metric only.

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
        A mapping suitable for AutoML engine consumption.  For now this
        function returns **empty dicts** as placeholders so that interfaces
        do not break while the detailed hyper-parameter grids from Tables 1 &
        2 are still under construction.
    """
    if kind not in {"model", "preprocessor"}:
        raise ValueError("kind must be 'model' or 'preprocessor'")
    # TODO – Implement full hyper-parameter grids from Tables 1 & 2.
    return {}

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