"""autogluon_wrapper – Stub Implementation"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree
from sklearn.base import BaseEstimator

from components.base import BaseEngine
from scripts.config import _MODEL_SPACE, DEFAULT_METRIC

console = Console(highlight=False)
logger = logging.getLogger(__name__)

# Map our generic model names to AutoGluon's internal model keys
AUTOGLUON_MODEL_MAP = {
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
}


class AutoGluonEngine(BaseEngine):
    """AutoGluon adapter conforming to the orchestrator's API."""

    def __init__(self, seed: int, timeout_sec: int, run_dir: Path):
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.run_dir = run_dir
        self._predictor: Any = None
        self._ag_output_path = self.run_dir / "autogluon_output"

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BaseEstimator:
        root = Tree("[AutoGluon]")
        logger.info("[%s] search-start", self.__class__.__name__)

        model_families = kwargs.get("model_families", _MODEL_SPACE.keys())
        metric = kwargs.get("metric", DEFAULT_METRIC)

        ag_included_models = []
        for family in model_families:
            ag_key = AUTOGLUON_MODEL_MAP.get(family)
            if ag_key:
                ag_included_models.append(ag_key)

        try:
            from autogluon.tabular import TabularPredictor

            root.add("library detected – running real AutoGluon")
            train_data = X.copy()
            train_data["target"] = y

            self._predictor = TabularPredictor(
                label="target",
                problem_type="regression",
                eval_metric=metric, # AutoGluon understands 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
                path=str(self._ag_output_path),
            )
            self._predictor.fit(
                train_data=train_data,
                presets="medium_quality_faster_train",  # Specific preset as required
                time_limit=self.timeout_sec,
                seed=self.seed,
                # Only include models explicitly requested by the orchestrator from our allowed list
                included_model_types=list(set(ag_included_models)) if ag_included_models else None,
            )
            best_score = self._predictor.leaderboard(silent=True)["score_val"].iloc[0]
            logger.info("[%s] best-score: %s", self.__class__.__name__, best_score)

        except ModuleNotFoundError as e:
            logger.warning("[%s] library missing – fallback LinearRegression: %s", self.__class__.__name__, e)
            from sklearn.linear_model import LinearRegression

            linreg = LinearRegression(n_jobs=1)
            linreg.fit(X, y)
            self._predictor = linreg

        console.print(root)
        logger.info("[%s] search-end", self.__class__.__name__)
        return self._predictor

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._predictor is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._predictor.predict(X)

    def export(self, path: Path):
        if self._predictor is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # AutoGluon saves its output to a directory. We need to zip it.
        zip_file_name = path / "autogluon"
        shutil.make_archive(str(zip_file_name), "zip", self._ag_output_path)
        logger.info("[%s] Zipped AutoGluon output to %s.zip", self.__class__.__name__, zip_file_name)


__all__ = ["AutoGluonEngine"] 