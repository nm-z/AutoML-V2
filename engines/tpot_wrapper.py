"""tpot_wrapper – Stub Implementation"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree
from sklearn.base import BaseEstimator

from components.base import BaseEngine
from scripts.config import _MODEL_SPACE, _PREPROCESSOR_SPACE, DEFAULT_METRIC

console = Console(highlight=False)
logger = logging.getLogger(__name__)

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
    # TPOT has specific components for outlier detection, but they are not
    # direct one-to-one mappings for KMeansOutlier, IsolationForest, LOF
    # so we will omit them from the config_dict for now.
}

# Map generic metric names to TPOT internal names
TPOT_METRIC_MAP = {
    "r2": "r2",
    "neg_mean_squared_error": "neg_mean_squared_error",
    "neg_mean_absolute_error": "neg_mean_absolute_error",
}


def _build_frozen_config(model_families: Sequence[str], prep_steps: Sequence[str]) -> dict:
    """Return a TPOT *config_dict* limited to supported Regressors & Transformers."""
    frozen: dict[str, dict] = {}

    # Add Regressor primitives (model families)
    for fam in model_families:
        tpot_name = TPOT_COMPONENT_MAP.get(fam)
        if tpot_name:
            frozen[tpot_name] = _MODEL_SPACE.get(fam, {})

    # Add pre-processing primitives
    for prep in prep_steps:
        tpot_name = TPOT_COMPONENT_MAP.get(prep)
        if tpot_name:
            # Only include preprocessors that TPOT supports as transformers
            # For simplicity, we are assuming all components in PREP_STEPS are safe
            # for TPOT in terms of type. More robust check might be needed for a live system.
            frozen[tpot_name] = _PREPROCESSOR_SPACE.get(prep, {})

    return frozen


class TPOTEngine(BaseEngine):
    """TPOT adapter conforming to the orchestrator's API."""

    def __init__(self, seed: int, timeout_sec: int, run_dir: Path):
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.run_dir = run_dir
        self._tpot: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BaseEstimator:
        root = Tree("[TPOT]")
        logger.info("[%s] search-start", self.__class__.__name__)

        model_families = kwargs.get("model_families", _MODEL_SPACE.keys())
        prep_steps = kwargs.get("prep_steps", _PREPROCESSOR_SPACE.keys())
        metric = kwargs.get("metric", DEFAULT_METRIC)

        custom_tpot_config = _build_frozen_config(model_families, prep_steps)
        tpot_metric = TPOT_METRIC_MAP.get(metric, "r2")

        try:
            from tpot import TPOTRegressor

            root.add("library detected – running real TPOTRegressor")

            self._tpot = TPOTRegressor(
                generations=100,
                population_size=100,
                config_dict=custom_tpot_config if custom_tpot_config else "TPOT light",
                early_stop=20,  # Early stopping after 20 generations without improvement
                scoring=tpot_metric,
                n_jobs=1,  # Prevent nested parallelism – orchestrator handles outer level
                random_state=self.seed,
                max_time_mins=max(1, self.timeout_sec // 60),
                verbosity=2,
                template="Regression", # enforce regression template
            )

            self._tpot.fit(X, y)
            logger.info(
                "[%s] best-score: %s", self.__class__.__name__, getattr(self._tpot, "_optimized_pipeline_score", "N/A")
            )

        except ModuleNotFoundError as e:
            logger.warning("[%s] library missing – fallback LinearRegression: %s", self.__class__.__name__, e)
            from sklearn.linear_model import LinearRegression

            linreg = LinearRegression(n_jobs=1)
            linreg.fit(X, y)
            self._tpot = linreg

        console.print(root)
        logger.info("[%s] search-end", self.__class__.__name__)
        return self._tpot.fitted_pipeline_ if hasattr(self._tpot, "fitted_pipeline_") else self._tpot

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._tpot is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._tpot.predict(X)

    def export(self, path: Path):
        if self._tpot is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Export the best pipeline as a Python script
        exported_pipeline_file = path / "exported_pipeline.py"
        self._tpot.export(exported_pipeline_file)
        logger.info("[%s] Saved exported pipeline to %s", self.__class__.__name__, exported_pipeline_file)

        # Save evaluation metrics (simple JSON for now)
        evaluation_data = {
            "best_score": getattr(self._tpot, "_optimized_pipeline_score", "N/A"),
            "metric": self._tpot.scoring_function if hasattr(self._tpot, "scoring_function") else "N/A",
            "generations": self._tpot.generations_left if hasattr(self._tpot, "generations_left") else "N/A",
            "evaluated_pipelines": len(self._tpot.evaluated_individuals_) if hasattr(self._tpot, "evaluated_individuals_") else "N/A",
        }
        evaluation_file = path / "evaluation.json"
        with open(evaluation_file, "w") as f:
            json.dump(evaluation_data, f, indent=4)
        logger.info("[%s] Saved evaluation summary to %s", self.__class__.__name__, evaluation_file)

__all__ = ["TPOTEngine"] 