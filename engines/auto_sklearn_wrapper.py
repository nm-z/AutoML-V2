from __future__ import annotations

import logging
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

# ---------------------------------------------------------------------------
# Mapping – project generic names → Auto-Sklearn estimator identifiers
# ---------------------------------------------------------------------------
_AUTOSKLEARN_MODEL_MAP = {
    "Ridge": "ridge_regression",
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

    def __init__(self, seed: int, timeout_sec: int, run_dir: Path):
        self.seed = seed
        self.timeout_sec = timeout_sec
        self.run_dir = run_dir
        self._automl: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> BaseEstimator:
        root = Tree("[Auto-Sklearn]")
        logger.info("[%s] search-start", self.__class__.__name__)

        model_families = kwargs.get("model_families", _MODEL_SPACE.keys())
        prep_steps = kwargs.get("prep_steps", _PREPROCESSOR_SPACE.keys())
        metric = kwargs.get("metric", DEFAULT_METRIC)

        include_estimators, include_preprocessors = self._build_search_space(
            model_families, prep_steps
        )
        translated_metric = self._translate_metric(metric)

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
                "[%s] best-score: %s", self.__class__.__name__, self._automl.performance_statistics[metric]
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