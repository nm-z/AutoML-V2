"""AutoML Orchestrator – Meta-Search Controller

This module coordinates multiple AutoML engines, ensuring each one searches
*exactly* the approved search-space and then selects the highest-R² champion
under a shared validation split.

Only the orchestration logic lives here; engine-specific glue code resides in
``engines/``.
"""
from __future__ import annotations

import logging
import time
import os # Import os for environment variable checking
import sys # Import sys for sys.exit
from pathlib import Path
import argparse
import json
import random
from datetime import datetime
import traceback
import sys
from typing import Any, Dict, Tuple, Sequence, Optional

import pandas as pd
from rich.console import Console
from rich.tree import Tree
import subprocess # Added subprocess

from scripts.data_loader import load_data # Import the new data_loader
from engines import discover_available
from engines.auto_sklearn_wrapper import AutoSklearnEngine
from engines.tpot_wrapper import TPOTEngine
from engines.autogluon_wrapper import AutoGluonEngine

import numpy as np
from sklearn.pipeline import Pipeline
import multiprocessing as _mp
from multiprocessing.queues import Queue as _MPQueue

# Scoring & CV utilities
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
    r2_score,
    mean_squared_error,
)

# Define the project version
__version__ = "0.1.0" # Added version attribute

# ---------------------------------------------------------------------------
# Global Constants and Configuration
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
N_SPLITS_CROSS_VALIDATION = 5
N_REPEATS_CROSS_VALIDATION = 3

# ---------------------------------------------------------------------------
# Component Discovery
# ---------------------------------------------------------------------------
# Build lists of available model families and preprocessing steps by inspecting
# the modules under ``components/models`` and ``components/preprocessors``.
_MODELS_DIR = Path(__file__).parent / "components" / "models"
_PREPROCESSORS_DIR = Path(__file__).parent / "components" / "preprocessors"

# Names correspond to the module file stems without the ``.py`` extension.
MODEL_FAMILIES = sorted(
    p.stem for p in _MODELS_DIR.glob("*.py") if p.stem != "__init__"
)

PREP_STEPS = sorted(
    p.stem for p in _PREPROCESSORS_DIR.rglob("*.py") if p.stem != "__init__"
)

# Wallclock limit for each engine, in seconds. This is a default and can be overridden by CLI.
WALLCLOCK_LIMIT_SEC = 3600  # 1 hour

# Default metric for evaluation
DEFAULT_METRIC = "r2"

# ---------------------------------------------------------------------------
# Global Logging Setup
# ---------------------------------------------------------------------------
# Set base logging level; can be overridden by AUTOML_VERBOSE
logging_level = logging.INFO
if "AUTOML_VERBOSE" in os.environ:
    logging_level = logging.DEBUG

# Log to both stdout and a persistent log file
log_file = Path(__file__).with_name("main.log")
logging.basicConfig(
    level=logging_level,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Optional Logstash integration for centralized logging
if os.getenv("LOGSTASH_HOST"):
    try:
        from logstash_async.handler import AsynchronousLogstashHandler
        from logstash_async.formatter import LogstashFormatter

        ls_host = os.environ.get("LOGSTASH_HOST")
        ls_port = int(os.environ.get("LOGSTASH_PORT", "5959"))
        ls_handler = AsynchronousLogstashHandler(ls_host, ls_port, database_path=None)
        ls_handler.setFormatter(LogstashFormatter())
        logging.getLogger().addHandler(ls_handler)
        logger.info("Logging to Logstash at %s:%s", ls_host, ls_port)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not configure Logstash handler: %s", exc)

# ---------------------------------------------------------------------------
# Reproducibility – set global seeds immediately at import-time
# ---------------------------------------------------------------------------
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# Rich console with recording enabled so we can persist logs afterwards
console = Console(highlight=False, record=True)


# ---------------------------------------------------------------------------
# Helper functions for runtime manifest
# ---------------------------------------------------------------------------

def _extract_pipeline_info(model: Any) -> list[dict]:
    """
    Extracts pipeline steps and hyperparameters from a fitted model object,
    handling different AutoML engine outputs.
    """
    pipeline_info = []
    if hasattr(model, "show_models") and callable(model.show_models):
        # This path is for Auto-Sklearn
        for rank, model_dict in model.show_models().items():
            for step_name, step_details in model_dict["configuration"].items():
                if step_name == "regressor":
                    # Handle the final estimator
                    pipeline_info.append({
                        "step": step_details["name"],
                        "params": step_details["hyperparameters"],
                    })
                elif step_name.startswith("data_preprocessor"):
                    # Handle preprocessors
                    pipeline_info.append({
                        "step": step_details["name"],
                        "params": step_details["hyperparameters"],
                    })
    elif hasattr(model, "export") and callable(model.export) and hasattr(model, "evaluated_individuals"):
        # This path is for TPOT (after it exports a pipeline and has evaluated_individuals)
        # This is a simplified extraction; a full implementation would parse the exported .py file
        if model.evaluated_individuals:
            best_individual = max(model.evaluated_individuals.values(), key=lambda ind: ind.fitness.values[0])
            # Extract steps and parameters from the best_individual pipeline
            # This requires knowledge of TPOT's internal representation, which is complex
            # For now, return a placeholder
            return [{"step": "TPOT Pipeline", "params": {"complexity": best_individual.complexity}}]
        return []
    elif hasattr(model, "leaderboard") and callable(model.leaderboard):
        # This path is for AutoGluon
        leaderboard = model.leaderboard(extra_info=True)
        if not leaderboard.empty:
            best_model_name = leaderboard.loc[0, 'model']
            # AutoGluon models are often complex ensembles, difficult to represent as simple steps
            # For now, return the best model name and its score
            return [{"step": best_model_name, "params": {"score_val": leaderboard.loc[0, 'score_val']}}]
        return []
    elif isinstance(model, Pipeline):
        # Handle scikit-learn pipelines directly if the champion is a simple sklearn pipeline
        pipeline_info = []
        for name, step in model.steps:
            params = {}
            if hasattr(step, 'get_params'):
                # Filter out non-hyperparameter parameters by convention (no leading underscore)
                params = {k: v for k, v in step.get_params().items() if not k.startswith('_') and '__' not in k}
            pipeline_info.append({"step": name, "params": params})
        return pipeline_info
    else:
        # Default for simple models, or if the model type is not recognized
        return [{"step": model.__class__.__name__, "params": {}}]


def _write_runtime_manifest(
    run_dir: Path,
    initial_cli_args: argparse.Namespace,
    static_config_data: Optional[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    champion_model: Any,
    champion_engine_name: str,
    champion_cv_score: float,
    per_engine_metrics: Dict[str, Dict[str, float]],
    per_engine_fitted_models: Dict[str, Any],
    total_duration_seconds: float,
):
    """
    Writes the runtime manifest config.json to the run directory, detailing the run's metadata,
    dataset, champion pipeline, and individual engine results.
    """
    timestamp_utc = datetime.utcnow().isoformat(timespec='seconds') + 'Z'

    dataset_info = {
        "path": str(Path(initial_cli_args.data).resolve()),
        "n_rows": X.shape[0],
        "n_features": X.shape[1],
        "target": Path(initial_cli_args.target).name.split('.')[0]
    }

    champion_pipeline_info = _extract_pipeline_info(champion_model)

    runners_data = {}
    for eng_name, metrics in per_engine_metrics.items():
        # Ensure log path is relative to the run_dir
        # The log file is now within the run_dir, e.g., 05_outputs/dataset_name/timestamp/run.log
        # So we don't need a separate 'logs' directory outside the run_dir.
        relative_log_path = "run.log" # Assuming all engine logs are consolidated into run.log
        runners_data[eng_name] = {
            "r2_mean": metrics.get("r2_mean", float('nan')),
            "r2_std": metrics.get("r2_std", float('nan')),
            "rmse_mean": metrics.get("rmse_mean", float('nan')),
            "rmse_std": metrics.get("rmse_std", float('nan')),
            "mae_mean": metrics.get("mae_mean", float('nan')),
            "mae_std": metrics.get("mae_std", float('nan')),
            "log": str(relative_log_path),
            "duration_seconds": metrics.get("duration_seconds", float('nan')),
            "pipeline": _extract_pipeline_info(per_engine_fitted_models.get(eng_name)) # Include pipeline info for each engine
        }

    manifest_data = {
        "run_meta": {
            "timestamp_utc": timestamp_utc,
            "budget_seconds": initial_cli_args.time,
            "metric": initial_cli_args.metric,
            "engines_invoked": list(per_engine_metrics.keys()),
            "blend_enabled": initial_cli_args.ensemble and not initial_cli_args.no_ensemble,
            "total_duration_seconds": total_duration_seconds,
            "data_path": str(Path(initial_cli_args.data).resolve()), # Add data_path to run_meta
            "target_path": str(Path(initial_cli_args.target).resolve()), # Add target_path to run_meta
        },
        "dataset": dataset_info,
        "champion": {
            "engine": champion_engine_name,
            "cv_score_r2": champion_cv_score, # Renamed to be specific to R2
            "pipeline": champion_pipeline_info,
            "artefact_paths": {
                "overall_champion_pickle": "overall_champion.pkl", # Updated key
                "feature_importance": "fi.csv" # Placeholder, not yet implemented
            }
        },
        "runners": runners_data
    }

    manifest_path = run_dir / "metrics.json" # Changed filename to metrics.json
    try:
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Runtime manifest saved to: {manifest_path}")
    except Exception as e:
        logger.error(f"Failed to write runtime manifest: {e}", exc_info=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _rmse(y_true, y_pred):
    """Root Mean Squared Error helper – always returns positive value."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _meta_search_sequential(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    run_dir: Path | str = "05_outputs",
    timeout_per_engine: int | None = None,
    metric: str = DEFAULT_METRIC,
    enable_ensemble: bool = False,
    n_cpus: int,
) -> Tuple[Any, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Run each available AutoML engine sequentially and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    run_dir
        Directory where all artifacts (models, logs, metrics) will be saved.
    timeout_per_engine
        Time limit in seconds for each AutoML engine. If None, uses
        WALLCLOCK_LIMIT_SEC.
    metric
        The primary metric to optimize for (e.g., 'r2', 'neg_mean_squared_error').
    enable_ensemble
        If True, and multiple engines are run, an ensemble of their champion
        pipelines will be created.
    n_cpus
        Number of CPU threads available inside the container. Passed to each
        AutoML engine and used to limit BLAS threading.

    Returns
    -------
    Tuple[Any, Dict[str, Any], Dict[str, Dict[str, float]]]
        - The champion fitted model (best performing across all engines).
        - A dictionary of fitted models, keyed by engine name.
        - A dictionary of performance metrics per engine, keyed by engine name.
          Each value is a dictionary containing mean and std dev for r2, rmse, mae.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    timeout_sec = timeout_per_engine or WALLCLOCK_LIMIT_SEC

    engines = {}
    fitted_models: Dict[str, Any] = {}
    per_engine_metrics: Dict[str, Dict[str, float]] = {}

    # Discover available engines
    discovered_engines = discover_available()
    if not discovered_engines:
        logger.error("No AutoML engines found. Please ensure engine wrappers are in the 'engines/' directory.")
        raise RuntimeError("No AutoML engines found.")

    root = Tree("[bold cyan]AutoML Meta-Search (Sequential)[/bold cyan]")

    for name, wrapper_module in discovered_engines.items():
        engine_node = root.add(f"[bold blue]Processing Engine: {name}[/bold blue]")
        engine_start_time = time.perf_counter()
        logger.info("[Orchestrator] Starting %s engine training...", name)

        try:
            engine_class_name = wrapper_module.__all__[0] # Assuming the class name is the first in __all__
            engine_class = getattr(wrapper_module, engine_class_name)
            # Instantiate the engine wrapper
            engine = engine_class(
                seed=RANDOM_STATE,
                timeout_sec=timeout_sec,
                run_dir=run_dir,
                metric=metric,
                n_cpus=n_cpus,
            )

            # Fit the model
            logger.info(f"[Orchestrator|{name}] Fitting model for {name}...")
            fitted_model = engine.fit(X, y)
            logger.info(f"[Orchestrator|{name}] Model fitting complete for {name}.")
            fitted_models[name] = fitted_model

            # Perform 5x3 Repeated Cross-Validation
            cv_node = engine_node.add("5×3 Repeated K-Fold evaluation…")
            rkf = RepeatedKFold(n_splits=N_SPLITS_CROSS_VALIDATION, n_repeats=N_REPEATS_CROSS_VALIDATION, random_state=RANDOM_STATE)
            scoring = {
                "r2": make_scorer(r2_score),
                "rmse": make_scorer(_rmse, greater_is_better=False),
                "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            }
            logger.info(f"[Orchestrator|{name}] Starting {N_REPEATS_CROSS_VALIDATION}x{N_SPLITS_CROSS_VALIDATION} Repeated K-Fold Cross-Validation for {name}...")
            cv_results = cross_validate(
                fitted_model,
                X,
                y,
                cv=rkf,
                scoring=scoring,
                return_train_score=False,
                n_jobs=1,
            )
            logger.info(f"[Orchestrator|{name}] Cross-Validation complete for {name}.")

            # Process CV results
            r2_scores = cv_results["test_r2"]
            rmse_scores = np.abs(cv_results["test_rmse"]) # RMSE is typically positive
            mae_scores = np.abs(cv_results["test_mae"])   # MAE is typically positive

            engine_duration = time.perf_counter() - engine_start_time

            per_engine_metrics[name] = {
                "r2_mean": np.mean(r2_scores),
                "r2_std": np.std(r2_scores),
                "rmse_mean": np.mean(rmse_scores),
                "rmse_std": np.std(rmse_scores),
                "mae_mean": np.mean(mae_scores),
                "mae_std": np.std(mae_scores),
                "duration_seconds": engine_duration,
            }
            logger.info(f"[Orchestrator|{name}] Metrics: R²={per_engine_metrics[name]['r2_mean']:.4f} (±{per_engine_metrics[name]['r2_std']:.4f}), RMSE={per_engine_metrics[name]['rmse_mean']:.4f} (±{per_engine_metrics[name]['rmse_std']:.4f}), MAE={per_engine_metrics[name]['mae_mean']:.4f} (±{per_engine_metrics[name]['mae_std']:.4f})")
            cv_node.add(f"R²: {per_engine_metrics[name]['r2_mean']:.4f} (±{per_engine_metrics[name]['r2_std']:.4f})")
            cv_node.add(f"RMSE: {per_engine_metrics[name]['rmse_mean']:.4f} (±{per_engine_metrics[name]['rmse_std']:.4f})")
            cv_node.add(f"MAE: {per_engine_metrics[name]['mae_mean']:.4f} (±{per_engine_metrics[name]['mae_std']:.4f})")

        except Exception as e:
            logger.error(f"[Orchestrator|{name}] Error running engine {name}: {e}", exc_info=True)
            engine_node.add(f"[bold red]Error: {e}[/bold red]")
            # Do not re-raise, allow other engines to run

    console.print(root)

    if not fitted_models:
        logger.error("No models were successfully fitted across all engines.")
        raise RuntimeError("No models were successfully fitted.")

    # Select the champion model based on the mean R2 score from CV results
    champion_engine_name = None
    champion_model = None
    best_r2_score = -np.inf

    for name, metrics in per_engine_metrics.items():
        if metrics.get("r2_mean", -np.inf) > best_r2_score:
            best_r2_score = metrics["r2_mean"]
            champion_engine_name = name
            champion_model = fitted_models[name]

    if champion_model is None:
        logger.error("Could not determine a champion model.")
        raise RuntimeError("Could not determine a champion model.")

    logger.info(f"Champion model selected: {champion_engine_name} with mean R² of {best_r2_score:.4f}")

    return champion_model, fitted_models, per_engine_metrics

def meta_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    artifacts_dir: Path | str = "05_outputs",
    timeout_per_engine: int | None = None,
    metric: str = DEFAULT_METRIC,
    enable_ensemble: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """This is a deprecated function and should not be used.
    Please use _meta_search_sequential or _meta_search_concurrent instead.
    """
    logger.warning("meta_search is deprecated. Use _meta_search_sequential or _meta_search_concurrent.")
    # For backward compatibility, call sequential by default
    champion_model, _, _ = _meta_search_sequential(
        X=X,
        y=y,
        run_dir=artifacts_dir,
        timeout_per_engine=timeout_per_engine,
        metric=metric,
        enable_ensemble=enable_ensemble,
        n_cpus=os.cpu_count() or 1,
    )
    return champion_model, {}

class _MeanEnsembleRegressor:  # noqa: D401 – simple averaging ensemble
    """A simple ensemble that averages predictions from multiple models."""

    def __init__(self, models: list[Any]):
        self.models = models

    def fit(self, *_args, **_kwargs):  # noqa: D401 – no-op fit
        return self

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)

def _blend(champions: list[Any]):
    """Create a simple averaging ensemble from a list of champion models."""
    if not champions:
        return None
    return _MeanEnsembleRegressor(models=champions)

def _runner(name: str, X_obj, y_obj, t_sec, r_dir, met, n_cpus, q_child: _mp.Queue):
    # Dynamically import the wrapper class within the child process
    from orchestrator import _get_automl_engine, RANDOM_STATE, N_SPLITS_CROSS_VALIDATION, N_REPEATS_CROSS_VALIDATION, _rmse, logging_level, DEFAULT_METRIC
    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
    from sklearn.model_selection import RepeatedKFold, cross_validate
    import time
    import random
    import numpy as np
    import traceback
    import sys
    import logging
    from pathlib import Path

    wrapper_class = _get_automl_engine(name)

    # Configure logging for the child process to the main run.log file
    if not logging.root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(r_dir / "run.log", mode="a"),
                logging.StreamHandler(sys.stdout),
            ],
        )
    child_logger = logging.getLogger(f"orchestrator.runner.{name}")
    child_logger.setLevel(logging_level)

    child_logger.info("[Orchestrator|%s] Child process started.", name)
    # Set seeds for child process to ensure reproducibility
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    engine_start_time = time.perf_counter()
    fitted_model = None
    engine_metrics = {}

    try:
        # Instantiate the engine wrapper within the child process
        wrapper_instance = wrapper_class(
            seed=RANDOM_STATE,
            timeout_sec=t_sec,
            run_dir=r_dir,
            metric=met,
            n_cpus=n_cpus,
        )
        fitted_model = wrapper_instance.fit(X_obj, y_obj)
        child_logger.info(f"[Orchestrator|{name}] Model fitting complete for {name}.")

        # Perform 5x3 Repeated Cross-Validation
        rkf = RepeatedKFold(n_splits=N_SPLITS_CROSS_VALIDATION, n_repeats=N_REPEATS_CROSS_VALIDATION, random_state=RANDOM_STATE)
        scoring = {
            "r2": make_scorer(r2_score),
            "rmse": make_scorer(_rmse, greater_is_better=False),
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        }
        child_logger.info(f"[Orchestrator|{name}] Starting {N_REPEATS_CROSS_VALIDATION}x{N_SPLITS_CROSS_VALIDATION} Repeated K-Fold Cross-Validation for {name}...")
        cv_results = cross_validate(
            fitted_model, X_obj, y_obj, cv=rkf, scoring=scoring, return_train_score=False, n_jobs=1 # n_jobs=1 to avoid nesting multiprocessing
        )
        child_logger.info(f"[Orchestrator|{name}] Cross-Validation complete for {name}.")

        # Process CV results
        r2_scores = cv_results["test_r2"]
        rmse_scores = np.abs(cv_results["test_rmse"])
        mae_scores = np.abs(cv_results["test_mae"])

        engine_duration = time.perf_counter() - engine_start_time

        engine_metrics = {
            "r2_mean": np.mean(r2_scores),
            "r2_std": np.std(r2_scores),
            "rmse_mean": np.mean(rmse_scores),
            "rmse_std": np.std(rmse_scores),
            "mae_mean": np.mean(mae_scores),
            "mae_std": np.std(mae_scores),
            "duration_seconds": engine_duration,
        }
        child_logger.info(f"[Orchestrator|{name}] Metrics: R²={engine_metrics['r2_mean']:.4f} (±{engine_metrics['r2_std']:.4f}), RMSE={engine_metrics['rmse_mean']:.4f} (±{engine_metrics['rmse_std']:.4f}), MAE={engine_metrics['mae_mean']:.4f} (±{engine_metrics['mae_std']:.4f})")
        q_child.put((name, fitted_model, engine_metrics)) # Put fitted model AND metrics
    except Exception as e: # Catch any error in the child process
        error_traceback = traceback.format_exc()
        child_logger.error("[Orchestrator|%s] Engine crashed: %s\n%s", name, e, error_traceback)
        q_child.put((name, None, {"error": str(e), "traceback": error_traceback})) # Indicate error, pass traceback

def _meta_search_concurrent(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str,
    timeout_per_engine: int,
    run_dir: Path | str = "05_outputs",
    enable_ensemble: bool,
    n_cpus: int,
) -> Tuple[Any, Dict[str, Any], Dict[str, Dict[str, float]]]:
    """Run each available AutoML engine in parallel and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    run_dir
        Directory where all artifacts (models, logs, metrics) will be saved.
    timeout_per_engine
        Time limit in seconds for each AutoML engine.
    metric
        The primary metric to optimize for (e.g., 'r2', 'neg_mean_squared_error').
    enable_ensemble
        If True, and multiple engines are run, an ensemble of their champion
        pipelines will be created.

    Returns
    -------
    Tuple[Any, Dict[str, Any], Dict[str, Dict[str, float]]]
        - The champion fitted model (best performing across all engines).
        - A dictionary of fitted models, keyed by engine name.
        - A dictionary of performance metrics per engine, keyed by engine name.
          Each value is a dictionary containing mean and std dev for r2, rmse, mae.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]AutoML Meta-Search (Concurrent)[/bold cyan]")

    # Discover available engines
    discovered_engines = discover_available()
    if not discovered_engines:
        logger.error("No AutoML engines found. Please ensure engine wrappers are in the 'engines/' directory.")
        raise RuntimeError("No AutoML engines found.")

    ctx = _mp.get_context("spawn")  # "spawn" is safer for multiprocessing
    q = ctx.Queue() # type: ignore
    workers = []

    for name in discovered_engines.keys():  # Iterate over names only
        worker = ctx.Process(
            target=_runner,
            args=(
                name,
                X,
                y,
                timeout_per_engine,
                run_dir,
                metric,
                n_cpus,
                q,
            ),
        )
        worker.start()
        workers.append(worker)

    per_engine_fitted_models: Dict[str, Any] = {}
    per_engine_metrics: Dict[str, Dict[str, float]] = {}

    for _ in workers: # Iterate as many times as there are workers
        eng_name, fitted_model, metrics = q.get()
        if fitted_model is not None: # Only store successful results
            per_engine_fitted_models[eng_name] = fitted_model
            per_engine_metrics[eng_name] = metrics
        else:
            error_msg = metrics.get('error', 'Unknown error')
            error_tb = metrics.get('traceback', 'No traceback available')
            console.print(f"[red]✗ {eng_name} error: {error_msg}[/]")
            logger.error(f"[Orchestrator] Error from {eng_name} child process:\n%s", error_tb)

    for worker in workers: # Wait for all processes to finish
        worker.join()

    if not per_engine_fitted_models:
        logger.error("All AutoML engines failed in concurrent run.")
        raise RuntimeError("All AutoML engines failed in concurrent run.")

    # Select the champion model based on the mean R2 score from CV results
    champion_engine_name = None
    champion_model = None
    best_r2_score = -np.inf

    for name, metrics in per_engine_metrics.items():
        if metrics.get("r2_mean", -np.inf) > best_r2_score:
            best_r2_score = metrics["r2_mean"]
            champion_engine_name = name
            champion_model = per_engine_fitted_models[name]

    if champion_model is None:
        logger.error("Could not determine a champion model from concurrent run.")
        raise RuntimeError("Could not determine a champion model from concurrent run.")

    logger.info(f"Champion model selected from concurrent run: {champion_engine_name} with mean R² of {best_r2_score:.4f}")

    return champion_model, per_engine_fitted_models, per_engine_metrics

def _get_automl_engine(name: str):
    """Dynamically import and return the specified AutoML engine wrapper."""
    if name == "auto_sklearn_wrapper":
        return AutoSklearnEngine
    elif name == "tpot_wrapper":
        return TPOTEngine
    elif name == "autogluon_wrapper":
        return AutoGluonEngine
    else:
        raise ValueError(f"Unknown AutoML engine: {name}")


def _score(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric: str = DEFAULT_METRIC,
):
    """Score a trained model using the specified metric."""
    if metric == "r2":
        return r2_score(y_valid, model.predict(X_valid))
    elif metric == "neg_mean_squared_error":
        return -mean_squared_error(y_valid, model.predict(X_valid))
    elif metric == "neg_mean_absolute_error":
        return -mean_absolute_error(y_valid, model.predict(X_valid))
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def _validate_components_availability() -> None:
    """Ensure that all component names correspond to actual module files."""
    missing: list[str] = []

    for name in MODEL_FAMILIES:
        if not (_MODELS_DIR / f"{name}.py").is_file():
            missing.append(f"model '{name}'")

    for name in PREP_STEPS:
        pattern = list(_PREPROCESSORS_DIR.rglob(f"{name}.py"))
        if not pattern:
            missing.append(f"preprocessor '{name}'")

    if missing:
        raise FileNotFoundError(
            "Missing components: " + ", ".join(sorted(missing))
        )

def _cli() -> None:
    """Parses command-line arguments and orchestrates the AutoML pipeline."""
    parser = argparse.ArgumentParser(
        description="\nAutoML Orchestrator – Meta-Search Controller",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the predictors CSV file (e.g., DataSets/3/predictors.csv)",
    )
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Path to the target CSV file (e.g., DataSets/3/targets.csv)",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=WALLCLOCK_LIMIT_SEC,
        help=f"Wall-clock time limit per engine in seconds (default: {WALLCLOCK_LIMIT_SEC})",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=DEFAULT_METRIC,
        help=f"Evaluation metric (e.g., r2, neg_mean_squared_error). Default: {DEFAULT_METRIC}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available AutoML engines",
    )
    parser.add_argument(
        "--autogluon",
        action="store_true",
        help="Run only the AutoGluon engine",
    )
    parser.add_argument(
        "--autosklearn",
        action="store_true",
        help="Run only the Auto-Sklearn engine",
    )
    parser.add_argument(
        "--tpot",
        action="store_true",
        help="Run only the TPOT engine",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensemble creation even if multiple engines are run",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=os.cpu_count() or 1,
        help="Number of CPU threads to use inside the container",
    )

    args = parser.parse_args()

    try:
        _validate_components_availability()
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    if not (args.all or args.autogluon or args.autosklearn or args.tpot):
        parser.error("At least one engine must be selected: --all, --autogluon, --autosklearn, or --tpot")

    selected_engines = []
    if args.all:
        selected_engines = ["autogluon", "autosklearn", "tpot"]
    else:
        if args.autogluon:
            selected_engines.append("autogluon")
        if args.autosklearn:
            selected_engines.append("autosklearn")
        if args.tpot:
            selected_engines.append("tpot")

    if not selected_engines:
        parser.error("No engines selected. Please use --all or specify at least one engine with --autogluon, --autosklearn, or --tpot.")

    # Define unique run directory for artifacts
    timestamp_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    dataset_name = Path(args.data).name  # Get dataset name from data path
    run_dir = Path("05_outputs") / dataset_name / timestamp_str
    run_dir.mkdir(parents=True, exist_ok=True)

    # Limit CPU threading for BLAS libraries and engines
    cpus_str = str(args.cpus)
    for var in [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "OMP_THREAD_LIMIT",
    ]:
        os.environ[var] = cpus_str

    # Configure logging to also write to a file within the run_dir
    file_handler = logging.FileHandler(run_dir / "run.log", mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    console.log("[bold green]Starting AutoML Orchestrator Run[/bold green]")
    start_time = time.perf_counter() # Record the start time of the run
    console.log(f"  Dataset: {args.data}")
    console.log(f"  Target: {args.target}")
    console.log(f"  Time Limit per Engine: {args.time} seconds")
    console.log(f"  Evaluation Metric: {args.metric}")
    console.log(f"  Selected Engines: {', '.join(selected_engines)}")
    console.log(f"  Artifacts Directory: {run_dir}")

    # Load data
    try:
        # The data_loader.py function now handles resolving the exact file paths
        # if a directory is provided, so we can pass args.data and args.target directly.
        X, y = load_data(args.data, args.target)
        logger.info(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
    except Exception as e:
        logger.error(f"Failed to load data: {e}", exc_info=True)
        sys.exit(1) # Terminate pipeline immediately

    # Partition data for final hold-out set
    from sklearn.model_selection import train_test_split
    X_train_cv, X_holdout, y_train_cv, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, shuffle=True
    )
    logger.info(f"Data split into training/CV ({X_train_cv.shape[0]} rows) and hold-out ({X_holdout.shape[0]} rows) sets.")

    try:
        if len(selected_engines) > 1 and not args.no_ensemble:
            # Run engines concurrently if multiple are selected and ensembling is enabled
            champion_model, fitted_engines, per_engine_metrics = _meta_search_concurrent(
                X=X_train_cv,
                y=y_train_cv,
                run_dir=run_dir,
                timeout_per_engine=args.time,
                metric=args.metric,
                enable_ensemble=not args.no_ensemble,
                n_cpus=args.cpus,
            )
            # Blend champions if ensembling is enabled
            if fitted_engines and not args.no_ensemble:
                logger.info("Blending champion models...")
                ensemble_model = _blend(list(fitted_engines.values()))
                if ensemble_model:
                    # Re-evaluate the ensemble model on the hold-out set to see if it's better
                    y_pred_ensemble = ensemble_model.predict(X_holdout)
                    r2_ensemble = r2_score(y_holdout, y_pred_ensemble)
                    if r2_ensemble > r2_score(y_holdout, champion_model.predict(X_holdout)):
                        champion_model = ensemble_model
                        logger.info(f"Ensemble model became the new champion with R²={r2_ensemble:.4f}")
        else:
            # Run sequentially if only one engine or ensembling is disabled
            champion_model, fitted_engines, per_engine_metrics = _meta_search_sequential(
                X=X_train_cv,
                y=y_train_cv,
                run_dir=run_dir,
                timeout_per_engine=args.time,
                metric=args.metric,
                enable_ensemble=False,  # Ensemble is handled outside sequential for single engine runs
                n_cpus=args.cpus,
            )

        # Evaluate champion model on the hold-out set
        logger.info("Evaluating champion model on the hold-out set...")
        y_pred_holdout = champion_model.predict(X_holdout)
        r2_holdout = r2_score(y_holdout, y_pred_holdout)
        rmse_holdout = _rmse(y_holdout, y_pred_holdout)
        mae_holdout = mean_absolute_error(y_holdout, y_pred_holdout)

        logger.info(f"Champion Model Hold-out Metrics: R²={r2_holdout:.4f}, RMSE={rmse_holdout:.4f}, MAE={mae_holdout:.4f}")

        # Save artifacts (champion models, metrics.json)
        overall_champion_path = run_dir / "overall_champion.pkl"
        import pickle
        with open(overall_champion_path, "wb") as f:
            pickle.dump(champion_model, f)
        logger.info(f"Overall champion model saved to {overall_champion_path}")

        # Create and save metrics.json
        metrics_data = {
            "run_meta": {
                "timestamp_utc": datetime.utcnow().isoformat(timespec='seconds') + 'Z',
                "budget_seconds": args.time,
                "metric": args.metric,
                "engines_invoked": selected_engines,
                "ensemble_enabled": not args.no_ensemble and len(selected_engines) > 1,
                "total_duration_seconds": time.perf_counter() - start_time, # Approximate total duration
                "data_path": str(Path(args.data).resolve()),
                "target_path": str(Path(args.target).resolve()),
            },
            "dataset": {
                "data_path": args.data,
                "target_path": args.target,
                "n_rows": X.shape[0],
                "n_features": X.shape[1],
                "target_name": Path(args.target).name.split('.')[0],
            },
            "holdout_metrics": {
                "r2": r2_holdout,
                "rmse": rmse_holdout,
                "mae": mae_holdout,
            },
            "per_engine_cv_metrics": per_engine_metrics, # This will contain detailed CV results
            "champion_pipeline_info": _extract_pipeline_info(champion_model),
        }

        metrics_json_path = run_dir / "metrics.json"
        with open(metrics_json_path, "w") as f:
            json.dump(metrics_data, f, indent=2)
        logger.info(f"Metrics saved to {metrics_json_path}")

        # Save per-engine champion models
        for eng_name, fitted_model in fitted_engines.items():
            engine_champion_path = run_dir / f"{eng_name}_champion.pkl"
            with open(engine_champion_path, "wb") as f:
                pickle.dump(fitted_model, f)
            logger.info(f"Champion model for {eng_name} saved to {engine_champion_path}")

    except Exception as e:
        logger.error(f"An error occurred during meta-search or evaluation: {e}", exc_info=True)
        sys.exit(1) # Terminate pipeline immediately

    console.log("[bold green]AutoML Orchestrator Run Completed[/bold green]")


if __name__ == "__main__":
    _cli() 
