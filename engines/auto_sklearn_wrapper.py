from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree
from sklearn.base import BaseEstimator

from components.base import BaseEngine

console = Console(highlight=False)
logger = logging.getLogger(__name__)

# --- Configuration for AutoSklearnEngine ---
_MODEL_SPACE = {
    "Ridge": {}, # Default hyperparameters, will be overridden by Auto-Sklearn's search
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

_PREPROCESSOR_SPACE = {
    "PCA": {},
    "RobustScaler": {},
    "StandardScaler": {},
    "QuantileTransform": {},
    "KMeansOutlier": {},
    "IsolationForest": {},
    "LocalOutlierFactor": {},
}

DEFAULT_METRIC = "r2"

# ---------------------------------------------------------------------------
# Mapping – project generic names → Auto-Sklearn estimator identifiers
# ---------------------------------------------------------------------------
_AUTOSKLEARN_MODEL_MAP = {
    "Ridge": "ridge_regression",
    "RPOP": None,
    "Lasso": "lasso_regression",
    "ElasticNet": "elastic_net",  # Auto-Sklearn uses elastic_net
    "SVR": "svr",
    "DecisionTree": "decision_tree",
    "RandomForest": "random_forest",
    "ExtraTrees": "extra_trees",
    "GradientBoosting": "gradient_boosting",
    "AdaBoost": "adaboost",
    "MLP": "mlp",
    "XGBoost": "xgradient_boosting",
    "LightGBM": "lightgbm",
}

_AUTOSKLEARN_PREPROCESSOR_MAP = {
    "PCA": "pca",
    "RobustScaler": "robust_scaler",
    "StandardScaler": "standard_scaler",
    "QuantileTransform": "quantile_transformer",
    # Outlier detection not directly mapped to Auto-Sklearn's preprocessors
}


class AutoSklearnEngine(BaseEngine):
    """Auto-Sklearn adapter conforming to the orchestrator's API."""

    def __init__(self, seed: int, timeout_sec: int, run_dir: Path, metric: str = DEFAULT_METRIC):
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.run_dir = run_dir
        self._automl: Any = None
        self._metric: str = metric # Store the metric for best_pipeline_info

    @property
    def name(self) -> str:
        return "AutoSklearnEngine"

    @property
    def best_pipeline_info(self) -> dict:
        if self._automl is None:
            return {"status": "not_fitted"}
        try:
            # Auto-Sklearn's best model info is usually available in performance_statistics
            # or by inspecting the ensemble.
            # For simplicity, we'll return the performance statistics of the best model.
            # A more detailed implementation would parse the actual pipeline steps.
            if hasattr(self._automl, 'performance_statistics'):
                return {
                    "score": self._automl.performance_statistics.get(self._metric, "N/A"),
                    "metric": self._metric,
                    "pipeline_description": "Auto-Sklearn internal ensemble/pipeline",
                    # You might add more details here by parsing self._automl.show_models()
                }
            return {"status": "fitted", "details": "No detailed pipeline info available from Auto-Sklearn instance"}
        except Exception as e:
            logger.error(f"Error extracting best_pipeline_info for AutoSklearnEngine: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    @property
    def run_info(self) -> dict:
        if self._automl is None:
            return {"status": "not_fitted"}
        
        # Auto-Sklearn writes its own logs and artifacts to tmp_folder and output_folder
        # which are already within the engine-specific run_dir.
        # We need to ensure these paths are returned relative to the run_dir if needed,
        # or as absolute paths.
        return {
            "best_score": self.best_pipeline_info.get("score", "N/A"),
            "run_dir": str(self.run_dir), # The base run directory for this engine
            "log": str(self.run_dir.parent / "logs" / f"{self.name}.log"), # Orchestrator's log for this engine
            "artefact_paths": {
                "model_pickle": str(self.run_dir / "model.pkl"), # The champion model saved by export
                "autosklearn_stats_csv": str(self.run_dir / "autosklearn_score_details.csv"),
                "autosklearn_tmp_folder": str(self.run_dir / "autosklearn_tmp"),
                "autosklearn_output_folder": str(self.run_dir / "autosklearn_out"),
            }
        }

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BaseEstimator:
        root = Tree("[Auto-Sklearn]")
        logger.info("[%s] search-start", self.__class__.__name__)

        model_families = kwargs.get("model_families", _MODEL_SPACE.keys())
        prep_steps = kwargs.get("prep_steps", _PREPROCESSOR_SPACE.keys())
        self._metric = kwargs.get("metric", DEFAULT_METRIC) # Store the metric

        include_estimators, include_preprocessors = self._build_search_space(
            model_families, prep_steps
        )
        translated_metric = self._translate_metric(self._metric)

        try:
            from autosklearn.regression import AutoSklearnRegressor

            root.add("library detected – running real Auto-Sklearn")

            self._automl = AutoSklearnRegressor(
                time_left_for_this_task=self.timeout_sec,
                per_run_time_limit=max(30, self.timeout_sec // 16),
                include_estimators=include_estimators,
                include_preprocessors=include_preprocessors,
                resampling_strategy="holdout",
                resampling_strategy_arguments={"train_size": 0.75},
                metric=translated_metric,
                n_jobs=1,  # prevent nested parallelism
                seed=self.seed,
                tmp_folder=self.run_dir / "autosklearn_tmp",
                output_folder=self.run_dir / "autosklearn_out",
            )

            self._automl.fit(X, y)
            logger.info(
                "[%s] best-score: %s", self.__class__.__name__, self._automl.performance_statistics[self._metric]
            )

        except ModuleNotFoundError as e:
            logger.warning("[%s] library missing – fallback LinearRegression: %s", self.__class__.__name__, e)
            from sklearn.linear_model import LinearRegression

            linreg = LinearRegression(n_jobs=1)
            linreg.fit(X, y)
            self._automl = linreg

        console.print(root)
        logger.info("[%s] search-end", self.__class__.__name__)
        return self._automl

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._automl is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return self._automl.predict(X)

    def export(self, path: Path):
        if self._automl is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Auto-Sklearn saves models and run statistics to its output_folder
        # We just need to make sure the output folder is within the run_dir
        # and copy relevant files if needed. For now, we rely on the tmp_folder/output_folder arguments.

        # Save performance statistics to CSV
        stats_df = pd.DataFrame([self._automl.performance_statistics])
        stats_file = path / "autosklearn_score_details.csv"
        stats_df.to_csv(stats_file, index=False)
        logger.info("[%s] Saved performance statistics to %s", self.__class__.__name__, stats_file)

        # Save the best model (champion pipeline) using Auto-Sklearn's own export mechanism
        # Auto-Sklearn's `save_model` exports a .pkl file by default.
        model_file = path / "model.pkl"
        self._automl.save_model(model_file)
        logger.info("[%s] Saved champion model to %s", self.__class__.__name__, model_file)

    def _build_search_space(self, models: Sequence[str], preprocessors: Sequence[str]) -> tuple[list, list]:
        include_estimators = [
            _AUTOSKLEARN_MODEL_MAP[f] for f in models if f in _AUTOSKLEARN_MODEL_MAP
        ]
        include_preprocessors = [
            _AUTOSKLEARN_PREPROCESSOR_MAP[f] for f in preprocessors if f in _AUTOSKLEARN_PREPROCESSOR_MAP
        ]

        # Filter out duplicates and None values
        include_estimators = list(dict.fromkeys(filter(None, include_estimators)))
        include_preprocessors = list(dict.fromkeys(filter(None, include_preprocessors)))
        return include_estimators, include_preprocessors

    def _translate_metric(self, metric: str) -> Any:
        from autosklearn.metrics import mean_absolute_error, mean_squared_error, r2

        metric_map = {
            "r2": r2,
            "neg_mean_squared_error": mean_squared_error,  # Auto-Sklearn's MSE is negative
            "neg_mean_absolute_error": mean_absolute_error, # Auto-Sklearn's MAE is negative
            # Add other metrics as needed
        }
        if metric not in metric_map:
            logger.warning("[%s] Metric '%s' not directly translatable to Auto-Sklearn. Using r2.", self.__class__.__name__, metric)
            return r2
        return metric_map[metric]


__all__ = ["AutoSklearnEngine"] 