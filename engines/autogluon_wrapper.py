"""autogluon_wrapper – Stub Implementation"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree
from sklearn.base import BaseEstimator, RegressorMixin

from components.base import BaseEngine

console = Console(highlight=False)
logger = logging.getLogger(__name__)

# --- Configuration for AutoGluonEngine ---
# These are now class attributes inside AutoGluonEngine
# _MODEL_SPACE = {
#     "Ridge": {},
#     "Lasso": {},
#     "ElasticNet": {},
#     "SVR": {},
#     "DecisionTree": {},
#     "RandomForest": {},
#     "ExtraTrees": {},
#     "GradientBoosting": {},
#     "AdaBoost": {},
#     "MLP": {},
#     "XGBoost": {},
#     "LightGBM": {},
# }

# DEFAULT_METRIC = "r2"

# Map our generic model names to AutoGluon's internal model keys
# These are now class attributes inside AutoGluonEngine
# AUTOGLUON_MODEL_MAP = {
#     "Ridge": "LR",
#     "Lasso": "LR",
#     "ElasticNet": "LR",
#     "SVR": None,  # AutoGluon doesn't have a direct SVR equivalent in default models
#     "DecisionTree": None,  # AutoGluon uses tree-based models, but not directly 'DecisionTree'
#     "RandomForest": "RF",
#     "ExtraTrees": "XT",
#     "GradientBoosting": "GBM",  # LightGBM is often used for Gradient Boosting
#     "AdaBoost": None,  # No direct AdaBoost model in default AutoGluon
#     "MLP": "NN_TORCH",  # Multi-layer Perceptron
#     "XGBoost": "XGB",
#     "LightGBM": "GBM",
# }


class AutoGluonEngine(BaseEngine):
    """AutoGluon adapter conforming to the orchestrator's API."""

    _MODEL_SPACE = {
        "Ridge": {}, # Default hyperparameters, will be overridden by AutoGluon's search
        "Lasso": {},
        "ElasticNet": {},
        "SVR": {},
        "DecisionTree": {},
        "RandomForest": {},
        "ExtraTrees": {},
        "GradientBoosting": {},
        "AdaBoost": {},
        "MLP": {},
        "XGBoost": {},
        "LightGBM": {},
    }

    _AUTOGLUON_MODEL_MAP = {
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

    def __init__(self, seed: int, timeout_sec: int, run_dir: Path, metric: str = "r2"):
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.run_dir = run_dir
        self._predictor: Any = None
        self._ag_output_path = self.run_dir / "autogluon_output"
        self._metric: str = metric # Store the metric for best_pipeline_info

    @property
    def name(self) -> str:
        return "AutoGluonEngine"

    @property
    def best_pipeline_info(self) -> dict:
        if self._predictor is None or not hasattr(self._predictor, '_ag_predictor'):
            return {"status": "not_fitted"}
        
        ag_predictor = self._predictor._ag_predictor
        try:
            leaderboard = ag_predictor.leaderboard(silent=True)
            if not leaderboard.empty:
                best_model_info = leaderboard.iloc[0].to_dict()
                return {
                    "score": best_model_info.get("score_val", "N/A"),
                    "metric": self._metric,
                    "model_name": best_model_info.get("model", "N/A"),
                    "pipeline_description": "AutoGluon ensemble/stacking",
                    "leaderboard_entry": best_model_info,
                }
            return {"status": "fitted", "details": "No detailed pipeline info available from AutoGluon predictor"}
        except Exception as e:
            logger.error(f"Error extracting best_pipeline_info for AutoGluonEngine: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @property
    def run_info(self) -> dict:
        if self._predictor is None:
            return {"status": "not_fitted"}
        
        return {
            "best_score": self.best_pipeline_info.get("score", "N/A"),
            "run_dir": str(self.run_dir), # The base run directory for this engine
            "log": str(self.run_dir.parent / "logs" / f"{self.name}.log"), # Orchestrator's log for this engine
            "artefact_paths": {
                "autogluon_output_zip": str(self.run_dir / "autogluon.zip"),
                "autogluon_output_folder": str(self._ag_output_path), # Original folder path
            }
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BaseEstimator:
        root = Tree("[AutoGluon]")
        logger.info("[%s] search-start", self.__class__.__name__)

        model_families = kwargs.get("model_families", self._MODEL_SPACE.keys())
        self._metric = kwargs.get("metric", self._metric) # Store the metric (if explicitly passed in fit method)

        ag_included_models = []
        for family in model_families:
            ag_key = self._AUTOGLUON_MODEL_MAP.get(family)
            if ag_key:
                ag_included_models.append(ag_key)

        try:
            from autogluon.tabular import TabularPredictor
            import numpy as np # Import numpy for set_seed if it's within autogluon itself
            # AutoGluon's set_seed was removed from autogluon.common.utils.utils in recent versions.
            # Instead, set seed globally with numpy and random before calling AutoGluon.
            # If AutoGluon still respects its own internal seed, that should be set in TabularPredictor args.
            # from autogluon.common.utils.utils import set_seed # Removed

            root.add("library detected – running real AutoGluon")
            train_data = X.copy()
            train_data["target"] = y
            
            # set_seed(self.seed) # Removed, assuming global random state is enough or AutoGluon handles it

            ag_predictor = TabularPredictor(
                label="target",
                problem_type="regression",
                eval_metric=self._metric, # AutoGluon understands 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'
                path=str(self._ag_output_path),
            )
            ag_predictor.fit(
                train_data=train_data,
                presets="medium_quality_faster_train",  # Specific preset as required
                time_limit=self.timeout_sec,
                # Only include models explicitly requested by the orchestrator from our allowed list
                included_model_types=list(set(ag_included_models)) if ag_included_models else None,
            )
            best_score = ag_predictor.leaderboard(silent=True)["score_val"].iloc[0]
            logger.info("[%s] best-score: %s", self.__class__.__name__, best_score)
            
            self._predictor = AutoGluonSklearnWrapper(ag_predictor=ag_predictor, path=self._ag_output_path)
            return self._predictor

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
        if self._predictor is None or not hasattr(self._predictor, '_ag_predictor'):
            raise RuntimeError("Model not fitted or not an AutoGluon model. Call fit() first.")

        ag_predictor = self._predictor._ag_predictor
        # AutoGluon saves its output to a directory. We need to zip it.
        zip_file_name = path / "autogluon"
        shutil.make_archive(str(zip_file_name), "zip", self._ag_output_path)
        logger.info("[%s] Zipped AutoGluon output to %s.zip", self.__class__.__name__, zip_file_name)


class AutoGluonSklearnWrapper(BaseEstimator, RegressorMixin):
    """A scikit-learn compatible wrapper for AutoGluon's TabularPredictor.

    This allows AutoGluon models to be used with scikit-learn's cross_validate
    and other utilities that expect the scikit-learn estimator API.
    """
    def __init__(self, ag_predictor: Any = None, path: Path = None):
        # We store the actual AutoGluon TabularPredictor instance
        self._ag_predictor = ag_predictor
        self.path = path # Store path for potential loading if needed

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        # In cross_validate, fit is called on cloned estimators.
        # If _ag_predictor is None, it means this is a cloned instance that needs to be loaded.
        if self._ag_predictor is None and self.path is not None:
            from autogluon.tabular import TabularPredictor
            self._ag_predictor = TabularPredictor.load(str(self.path))
        elif self._ag_predictor is None:
            raise RuntimeError("AutoGluonPredictor not provided and path is not set for loading.")

        # No need to call fit on _ag_predictor here, as it's already fitted from the original AutoGluonEngine.fit call
        # This fit method is primarily for scikit-learn's cloning/validation purposes.
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self._ag_predictor is None:
            raise RuntimeError("AutoGluonPredictor is not initialized. Call fit() first.")
        return self._ag_predictor.predict(X).values

    def get_params(self, deep: bool = True) -> dict:
        # Required by scikit-learn's BaseEstimator for cloning.
        # Return parameters that would be needed to re-instantiate the wrapper.
        return {"ag_predictor": self._ag_predictor, "path": self.path}

    def set_params(self, **parameters):
        # Required by scikit-learn's BaseEstimator for cloning.
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


__all__ = ["AutoGluonEngine"] 