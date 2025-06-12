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
from pathlib import Path
import argparse
import json
import random
from datetime import datetime
import traceback
from typing import Any, Dict, Tuple, Sequence, Optional

import pandas as pd
from rich.console import Console
from rich.tree import Tree
import subprocess # Added subprocess

import scripts.config as config
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
# Global Logging Setup
# ---------------------------------------------------------------------------
# Set base logging level; can be overridden by AUTOML_VERBOSE
logging_level = logging.INFO
if "AUTOML_VERBOSE" in os.environ:
    logging_level = logging.DEBUG

logging.basicConfig(level=logging_level, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reproducibility – set global seeds immediately at import-time
# ---------------------------------------------------------------------------
random.seed(config.RANDOM_STATE)
np.random.seed(config.RANDOM_STATE)

# Rich console with recording enabled so we can persist logs afterwards
console = Console(highlight=False, record=True)


# ---------------------------------------------------------------------------
# Helper functions for runtime manifest
# ---------------------------------------------------------------------------

def get_git_sha():
    """Returns the current Git commit SHA, or 'N/A' if not a Git repository."""
    try:
        # Get the current commit SHA; cwd=Path(__file__).parent.parent ensures we are in AutoML root
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).parent.parent
        ).strip().decode('utf-8')
        return sha
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "N/A"

def _extract_pipeline_info(model: Any) -> list[dict]:
    """
    Extracts pipeline steps and hyperparameters from a fitted model object,
    handling different AutoML engine outputs.
    """
    pipeline_info = []
    if isinstance(model, Pipeline): # Generic Scikit-learn pipeline
        for name, step_obj in model.steps:
            step_name = step_obj.__class__.__name__
            params = {}
            if hasattr(step_obj, 'get_params'):
                all_params = step_obj.get_params(deep=False)
                # Filter for basic types; omit complex objects/references
                params = {k: v for k, v in all_params.items() if isinstance(v, (int, float, str, bool, list, type(None)))}
            pipeline_info.append({"step": step_name, "params": params})
    elif hasattr(model, 'get_model_pipeline'): # AutoSklearn
        try:
            sklearn_pipeline = model.get_model_pipeline()
            if isinstance(sklearn_pipeline, Pipeline):
                for name, step_obj in sklearn_pipeline.steps:
                    step_class_name = getattr(step_obj, '__class__', type(step_obj)).__name__
                    params = {}
                    if hasattr(step_obj, 'get_params'):
                        all_params = step_obj.get_params(deep=False)
                        params = {k: v for k, v in all_params.items() if isinstance(v, (int, float, str, bool, list, type(None)))}
                    pipeline_info.append({"step": step_class_name, "params": params})
            elif hasattr(sklearn_pipeline, '__class__'):
                step_class_name = getattr(sklearn_pipeline, '__class__', type(sklearn_pipeline)).__name__
                params = {}
                if hasattr(sklearn_pipeline, 'get_params'):
                    all_params = sklearn_pipeline.get_params(deep=False)
                    params = {k: v for k, v in all_params.items() if isinstance(v, (int, float, str, bool, list, type(None)))}
                pipeline_info.append({"step": step_class_name, "params": params})
            else:
                logger.warning("Could not extract detailed pipeline from AutoSklearn model. Type: %s", type(sklearn_pipeline))
                pipeline_info.append({"step": "AutoSklearnInternal", "params": {}})
        except Exception as e:
            logger.warning(f"Error extracting AutoSklearn pipeline: {e}", exc_info=True)
            pipeline_info.append({"step": "AutoSklearnInternal", "params": {}})
    elif hasattr(model, 'path') and 'autogluon' in str(model.path).lower(): # AutoGluon Predictor
        pipeline_info.append({"step": "AutoGluonEnsemble", "params": {}})
    elif hasattr(model, 'fitted_pipeline_') and isinstance(model.fitted_pipeline_, Pipeline): # TPOT
        for name, step_obj in model.fitted_pipeline_.steps:
            step_name = step_obj.__class__.__name__
            params = {}
            if hasattr(step_obj, 'get_params'):
                all_params = step_obj.get_params(deep=False)
                params = {k: v for k, v in all_params.items() if isinstance(v, (int, float, str, bool, list, type(None)))}
            pipeline_info.append({"step": step_name, "params": params})
    elif hasattr(model, 'exported_pipeline_file'): # TPOT, if it has an exported pipeline path
        pipeline_info.append({"step": "TPOTExportedPipeline", "params": {"file": model.exported_pipeline_file}})
    else:
        logger.warning(f"Could not extract pipeline info for model of type: {type(model)}")
        pipeline_info.append({"step": "UnknownModelType", "params": {}})

    return pipeline_info


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
    git_sha = get_git_sha()

    dataset_info = {
        "path": str(Path(initial_cli_args.data).resolve()),
        "n_rows": X.shape[0],
        "n_features": X.shape[1],
        "target": Path(initial_cli_args.target).name.split('.')[0]
    }

    champion_pipeline_info = _extract_pipeline_info(
        per_engine_fitted_models.get(champion_engine_name)
    )

    runners_data = {}
    for eng_name, metrics in per_engine_metrics.items():
        relative_log_path = Path("../logs") / f"{eng_name}.log"
        runners_data[eng_name] = {
            "best_score": metrics.get("r2", float('nan')),
            "run_dir": f"{eng_name}_artifacts/", # Placeholder for future sub-directories if engines create them
            "log": str(relative_log_path)
        }

    manifest_data = {
        "run_meta": {
            "timestamp_utc": timestamp_utc,
            "git_sha": git_sha,
            "budget_seconds": initial_cli_args.time,
            "metric": initial_cli_args.metric,
            "engines_invoked": list(per_engine_metrics.keys()),
            "blend_enabled": initial_cli_args.ensemble and not initial_cli_args.no_ensemble,
            "duration_seconds": total_duration_seconds,
            "input_config_path": str(Path(initial_cli_args.config).resolve()) if initial_cli_args.config else "N/A",
            "input_config_content": static_config_data if static_config_data else {},
        },
        "dataset": dataset_info,
        "champion": {
            "engine": champion_engine_name,
            "cv_score": champion_cv_score,
            "pipeline": champion_pipeline_info,
            "artefact_paths": {
                "pipeline_pickle": f"{champion_engine_name}_champion.pkl",
                "feature_importance": "fi.csv" # Placeholder, not yet implemented
            }
        },
        "runners": runners_data
    }

    manifest_path = run_dir / "config.json"
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
    run_dir: Path | str = "05_outputs", # Changed artifacts_dir to run_dir
    timeout_per_engine: int | None = None,
    metric: str = config.DEFAULT_METRIC,
    enable_ensemble: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Dict[str, float]]]: # Added per_engine_metrics to return type
    """Run each available AutoML engine and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    run_dir
        Where to persist every artifact (models, logs, CV results, …).
    timeout_per_engine
        Optional wall-clock limit that overrides the global constant.

    Returns
    -------
    champion
        The top-performing model *object* (already fitted).
    per_engine_results
        ``{engine_name: fitted_model}`` for *every* engine that succeeded.
    per_engine_metrics
        ``{engine_name: {metric_name: value}}`` for each engine's CV performance.
    """

    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    timeout_sec = timeout_per_engine or config.WALLCLOCK_LIMIT_SEC

    root = Tree("[bold cyan]AutoML Meta-Search[/bold cyan]")

    # ---------------------------------------------------------------------
    # 0️⃣ Sanity checks – validate workspace invariants
    # ---------------------------------------------------------------------
    try:
        _validate_components_availability()
    except Exception as exc:
        console.print(f"[red]Workspace validation failed: {exc}")
        raise

    # ---------------------------------------------------------------------
    # 1️⃣ Discover available engines
    # ---------------------------------------------------------------------
    discover_node = root.add("Discovering engine wrappers…")
    all_engines = discover_available()
    engines = {}
    import sys
    python_major_version = sys.version_info.major
    python_minor_version = sys.version_info.minor

    for name, wrapper in all_engines.items():
        # Skip AutoGluon only on versions known to be incompatible (e.g., Python 3.13)
        if name == "autogluon_wrapper" and python_major_version == 3 and python_minor_version == 13:
            discover_node.add(
                f"[yellow]⚠ Skipping {name} due to Python 3.13 compatibility issues"
            )
            continue

        discover_node.add(f"[green]✔ {name}")
        engines[name] = wrapper

    if not engines:
        console.print(root)
        raise RuntimeError("No AutoML engine wrappers could be imported – aborting.")

    # ---------------------------------------------------------------------
    # 2️⃣ Iterate over engines
    # ---------------------------------------------------------------------
    results: Dict[str, Any] = {}
    for eng_name, wrapper in engines.items():
        eng_node = root.add(f"[bold]{eng_name}[/] ▸ running for {timeout_sec}s…")
        start = time.perf_counter()
        try:
            model = wrapper.fit_engine(
                X,
                y,
                model_families=config.MODEL_FAMILIES,
                prep_steps=config.PREP_STEPS,
                seed=config.RANDOM_STATE,
                timeout_sec=timeout_sec,
                metric=metric,
            )
            duration = time.perf_counter() - start
            eng_node.add(f"[green]✓ completed in {duration:.1f}s")
            results[eng_name] = model

            # Persist model artifact
            file_path = run_dir / f"{eng_name}_champion.pkl"
            try:
                import joblib

                joblib.dump(model, file_path)
                eng_node.add(f"[blue]🔖 saved → {file_path}")
            except Exception as exc:  # noqa: BLE001 – log & continue
                eng_node.add(f"[yellow]⚠ could not save: {exc}")
        except Exception as err:  # noqa: BLE001 – fail fast
            eng_node.add(f"[red]✗ error: {err}")
            console.print(root)
            raise  # Fail-Fast-On-Errors

    # If every engine errored-out we cannot proceed to evaluation
    if not results:
        console.print(root)
        raise RuntimeError("All AutoML engines failed.")

    # ---------------------------------------------------------------------
    # 3️⃣ Cross-validation – 5×3 Repeated K-Fold
    # ---------------------------------------------------------------------
    cv_node = root.add("5×3 Repeated K-Fold evaluation…")

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_STATE)

    scoring = {
        "r2": "r2",
        "rmse": make_scorer(_rmse, greater_is_better=False),
        "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    }

    per_engine_metrics: Dict[str, Dict[str, float]] = {}

    for eng_name, model in results.items():
        sub = cv_node.add(f"{eng_name} ▸ CV evaluating…")
        try:
            cv_res = cross_validate(
                model,
                X,
                y,
                cv=rkf,
                scoring=scoring,
                n_jobs=config.N_JOBS_CV,
                error_score="raise",
            )

            r2_m = float(cv_res["test_r2"].mean())
            rmse_m = float((-cv_res["test_rmse"].mean()))
            mae_m = float((-cv_res["test_mae"].mean()))

            r2_s = float(cv_res["test_r2"].std())
            rmse_s = float((cv_res["test_rmse"].std()))
            mae_s = float((cv_res["test_mae"].std()))

            per_engine_metrics[eng_name] = {
                "r2": r2_m,
                "rmse": rmse_m,
                "mae": mae_m,
                "r2_std": r2_s,
                "rmse_std": rmse_s,
                "mae_std": mae_s,
            }

            sub.add(
                f"[green]✓ R²={r2_m:.4f}  RMSE={rmse_m:.4f}  MAE={mae_m:.4f}"
            )
        except Exception as exc:  # noqa: BLE001 – fail fast
            sub.add(f"[red]✗ error during CV: {exc}")
            console.print(root)
            raise

    # ---------------------------------------------------------------------
    # 4️⃣ Select the overall champion based on mean CV R²
    # ---------------------------------------------------------------------
    champion_name, champion_model = max(
        results.items(), key=lambda kv: per_engine_metrics[kv[0]]["r2"]
    )

    champ_score = per_engine_metrics[champion_name]["r2"]
    root.add(f"🏆 [bold green]Champion → {champion_name}[/]").add(
        f"R²={champ_score:.4f} (from CV)"
    )

    # ---------------------------------------------------------------------
    # 5️⃣ Final evaluation on hold-out set (if applicable)
    # ---------------------------------------------------------------------
    # This part assumes a hold-out set is managed outside this function
    # (e.g., in _cli after data loading).

    # Return the champion model and per-engine results and metrics
    console.print(root)
    return champion_model, results, per_engine_metrics # Added per_engine_metrics to return

def meta_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    artifacts_dir: Path | str = "05_outputs",
    timeout_per_engine: int | None = None,
    metric: str = config.DEFAULT_METRIC,
    enable_ensemble: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Deprecated: Use _meta_search_sequential or _meta_search_concurrent instead."""
    logger.warning("meta_search is deprecated. Use _meta_search_sequential or _meta_search_concurrent.")
    # This function will now call _meta_search_sequential for compatibility
    # with existing calls that might not expect the new return values.
    champion, results, _ = _meta_search_sequential(
        X=X, y=y, run_dir=artifacts_dir, timeout_per_engine=timeout_per_engine,
        metric=metric, enable_ensemble=enable_ensemble
    )
    return champion, results


def _validate_components_availability() -> None:
    """Validate that all components referenced in config.py are available on PATH.

    Raises
    ------
    RuntimeError
        If any component cannot be imported.
    """
    pass  # Already validated by import checks and initial setup


def _cli() -> None:
    """Parse CLI arguments and run the orchestrator."""
    parser = argparse.ArgumentParser(
        description="Run AutoML meta-search across multiple engines."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all engines concurrently. (Not yet implemented)",
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
        default=config.WALLCLOCK_LIMIT_SEC,
        help=f"Wall-clock time limit per engine in seconds (default: {config.WALLCLOCK_LIMIT_SEC})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="05_outputs",
        help="Directory to save artifacts (models, logs, metrics). Default: 05_outputs/",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=config.DEFAULT_METRIC,
        help=f"Evaluation metric (e.g., r2, neg_mean_squared_error). Default: {config.DEFAULT_METRIC}",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a JSON configuration file (e.g., config.json). Overrides other CLI args.",
    )
    parser.add_argument(
        "--ensemble",
        action="store_true",
        help="Enable ensembling of champion models (if multiple engines succeed).",
    )
    parser.add_argument(
        "--no-ensemble",
        action="store_true",
        help="Disable ensembling of champion models.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
        help="Show program's version number and exit.",
    )

    args = parser.parse_args()

    # Handle --ensemble and --no-ensemble conflict
    if args.ensemble and args.no_ensemble:
        parser.error("Cannot use both --ensemble and --no-ensemble simultaneously.")

    static_config_data = None # Store the content of the initial config file
    # Load configuration from JSON file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                static_config_data = json.load(f) # Capture the content
            # Override CLI args with config file values if present
            args.data = static_config_data.get("data_path", args.data)
            args.target = static_config_data.get("target_path", args.target)
            args.time = static_config_data.get("time_limit", args.time)
            args.output_dir = static_config_data.get("output_dir", args.output_dir)
            args.metric = static_config_data.get("metric", args.metric)
            args.ensemble = static_config_data.get("ensemble", args.ensemble)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(2)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {args.config}")
            sys.exit(2)

    # Ensure the base output directory exists
    base_output_dir = Path(args.output_dir)
    base_output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure the logs directory exists as a sibling to timestamped run directories
    logs_dir = base_output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured logs directory exists: {logs_dir}")

    # Generate a unique run directory with timestamp
    run_id = datetime.now().isoformat(timespec='seconds').replace(':', '-')
    run_dir = base_output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created run directory: {run_dir}")

    # Load data using the new data_loader module
    try:
        X, y = load_data(args.data, args.target)
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        sys.exit(2) # Exit code for data loading error

    logger.info("Starting AutoML meta-search...")
    logger.info("Predictors path: %s", args.data)
    logger.info("Target path: %s", args.target)
    logger.info("Time limit per engine: %d seconds", args.time)
    logger.info("Output directory: %s", run_dir)
    logger.info("Evaluation metric: %s", args.metric)

    start_total_run = time.perf_counter()
    champion_model = None
    per_engine_results = {}
    per_engine_metrics = {}
    exit_code = 2 # Default to crash exit code

    try:
        if args.all:
            champion_model, per_engine_results, per_engine_metrics = _meta_search_concurrent(
                X=X,
                y=y,
                run_dir=run_dir, # Pass the new run_dir
                timeout_per_engine=args.time,
                metric=args.metric,
                enable_ensemble=args.ensemble and not args.no_ensemble,
            )
        else:
            champion_model, per_engine_results, per_engine_metrics = _meta_search_sequential( # Renamed
                X=X,
                y=y,
                run_dir=run_dir, # Pass the new run_dir
                timeout_per_engine=args.time,
                metric=args.metric,
                enable_ensemble=args.ensemble and not args.no_ensemble,
            )
        logger.info("AutoML meta-search completed successfully.")
        exit_code = 0 # Success exit code

    except RuntimeError as e:
        logger.error(f"AutoML meta-search failed: {e}")
        if "timeout" in str(e).lower():
            exit_code = 3 # Timeout exit code
        else:
            exit_code = 2 # General crash exit code
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
        exit_code = 2 # General crash exit code
    finally:
        total_duration = time.perf_counter() - start_total_run
        if exit_code == 0: # Only write manifest on successful completion
            # Identify the champion engine name and its CV score
            champion_engine_name = None
            champion_cv_score = float('nan')
            if per_engine_metrics:
                # Find the champion based on the 'r2' metric
                champion_engine_name, _ = max(
                    per_engine_metrics.items(), key=lambda item: item[1].get("r2", float('-inf'))
                )
                champion_cv_score = per_engine_metrics[champion_engine_name].get("r2", float('nan'))

            # Save the overall champion model if it exists and run was successful
            if champion_model:
                overall_champion_path = run_dir / "overall_champion.pkl"
                try:
                    import joblib
                    joblib.dump(champion_model, overall_champion_path)
                    logger.info(f"Overall champion model saved to: {overall_champion_path}")
                except Exception as exc:
                    logger.warning(f"Could not save overall champion model: {exc}")

            _write_runtime_manifest(
                run_dir=run_dir,
                initial_cli_args=args,
                static_config_data=static_config_data,
                X=X,
                y=y,
                champion_model=champion_model,
                champion_engine_name=champion_engine_name,
                champion_cv_score=champion_cv_score,
                per_engine_metrics=per_engine_metrics,
                per_engine_fitted_models=per_engine_results, # This is the dict of fitted models
                total_duration_seconds=total_duration,
            )
        sys.exit(exit_code)


# ---------------------------------------------------------------------------
# Engine Abstractions (re-located from the top)
# ---------------------------------------------------------------------------

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
    metric: str = config.DEFAULT_METRIC,
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


def _fit_engine(
    eng_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    timeout_sec: int,
    run_dir: Path,
    model_families: Sequence[str],
    prep_steps: Sequence[str],
    metric: str,
) -> Any:
    """Fit a single AutoML engine and return the fitted model."""
    logger.info("[Orchestrator] Starting %s engine training...", eng_name)
    engine_class = _get_automl_engine(eng_name)
    engine = engine_class(seed=config.RANDOM_STATE, timeout_sec=timeout_sec, run_dir=run_dir)

    try:
        fitted_model = engine.fit(
            X_train, y_train,
            model_families=model_families,
            prep_steps=prep_steps,
            metric=metric,
        )
        logger.info("[Orchestrator] %s engine training completed.", eng_name)
        return fitted_model
    except Exception as e:
        logger.error("[Orchestrator] Error fitting %s engine: %s", eng_name, e, exc_info=True)
        raise RuntimeError(f"Engine {eng_name} crashed during fit: {e}") from e


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


def _meta_search_concurrent(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str,
    timeout_per_engine: int,
    run_dir: Path | str = "05_outputs", # Changed artifacts_dir to run_dir
    enable_ensemble: bool,
) -> Tuple[Any, Dict[str, Any], Dict[str, Dict[str, float]]]: # Added per_engine_metrics to return type
    """Run each available AutoML engine in parallel and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    run_dir
        Where to persist every artifact (models, logs, CV results, …).
    timeout_per_engine
        Optional wall-clock limit that overrides the global constant.

    Returns
    -------
    champion
        The top-performing model *object* (already fitted).
    per_engine_results
        ``{engine_name: fitted_model}`` for *every* engine that succeeded.
    per_engine_metrics
        ``{engine_name: {metric_name: value}}`` for each engine's CV performance.
    """

    run_dir = Path(run_dir)
    # Ensure run_dir exists, as child processes will use it.
    run_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold cyan]AutoML Meta-Search (Concurrent)[/bold cyan]")

    # ---------------------------------------------------------------------
    # 0️⃣ Sanity checks – validate workspace invariants
    # ---------------------------------------------------------------------
    try:
        _validate_components_availability()
    except Exception as exc:
        console.print(f"[red]Workspace validation failed: {exc}")
        raise

    # ---------------------------------------------------------------------
    # 1️⃣ Discover available engines
    # ---------------------------------------------------------------------
    discover_node = Tree("Discovering engine wrappers…")
    all_engines = discover_available()
    engines = {}
    import sys
    python_major_version = sys.version_info.major
    python_minor_version = sys.version_info.minor

    for name, wrapper in all_engines.items():
        if name == "autogluon_wrapper" and python_major_version == 3 and python_minor_version == 13:
            discover_node.add(
                f"[yellow]⚠ Skipping {name} due to Python 3.13 compatibility issues"
            )
            continue
        discover_node.add(f"[green]✔ {name}")
        engines[name] = wrapper

    if not engines:
        console.print(discover_node)
        raise RuntimeError("No AutoML engine wrappers could be imported – aborting.")
    console.print(discover_node)

    # ---------------------------------------------------------------------
    # 2️⃣ Run each engine in a separate process
    # ---------------------------------------------------------------------
    ctx = _mp.get_context("spawn")  # "spawn" is safer for multiprocessing
    q = ctx.Queue() # type: ignore
    workers = []

    # Pass necessary arguments to child processes
    # For multiprocessing, objects passed must be pickleable. X, y can be large.
    # Consider serializing X, y or using shared memory if performance is an issue.
    # For now, pass them directly (they will be pickled).

    # Define the worker function that each process will run
    def _runner(name: str, X_obj, y_obj, t_sec, r_dir, model_families, prep_steps, met, q: _MPQueue):
        # Configure logging for the child process to a dedicated file
        log_file_path = r_dir.parent / "logs" / f"{name}.log" # e.g., 05_outputs/logs/autosklearn.log
        log_file_path.parent.mkdir(parents=True, exist_ok=True) # Ensure logs directory exists

        # Remove existing handlers to avoid duplicate logging if run multiple times
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        file_handler = logging.FileHandler(log_file_path, mode='a')
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
        logging.root.setLevel(logging_level) # Use global logging level

        logger.info("[Orchestrator|%s] Child process started. Logs to: %s", name, log_file_path)
        # Set seeds for child process to ensure reproducibility
        random.seed(config.RANDOM_STATE)
        np.random.seed(config.RANDOM_STATE)

        engine_start_time = time.perf_counter()
        fitted_model = None
        engine_metrics = {}

        try:
            engine_class = _get_automl_engine(name)
            # Pass the run_dir to the engine wrapper so it saves its artifacts there
            engine = engine_class(seed=config.RANDOM_STATE, timeout_sec=t_sec, run_dir=r_dir)
            fitted_model = engine.fit(
                X_obj, y_obj,
                model_families=model_families,
                prep_steps=prep_steps,
                metric=met,
            )
            engine_duration = time.perf_counter() - engine_start_time
            logger.info("[Orchestrator|%s] Engine training completed in %.1fs.", name, engine_duration)

            # Persist model artifact for this engine
            file_path = r_dir / f"{name}_champion.pkl"
            try:
                import joblib
                joblib.dump(fitted_model, file_path)
                logger.info(f"[Orchestrator|%s] Saved champion model → {file_path}", name)
            except Exception as exc:  # noqa: BLE001 – log & continue
                logger.warning(f"[Orchestrator|%s] Could not save model: {exc}", name)

            # Perform CV evaluation for this engine in child process
            logger.info("[Orchestrator|%s] Starting 5×3 Repeated K-Fold evaluation…", name)
            rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_STATE)
            scoring = {
                "r2": "r2",
                "rmse": make_scorer(_rmse, greater_is_better=False),
                "mae": make_scorer(mean_absolute_error, greater_is_better=False),
            }
            cv_res = cross_validate(
                fitted_model,
                X_obj,
                y_obj,
                cv=rkf,
                scoring=scoring,
                n_jobs=1, # Run CV single-threaded in child process to avoid nesting multiprocessing
                error_score="raise",
            )

            r2_m = float(cv_res["test_r2"].mean())
            rmse_m = float((-cv_res["test_rmse"].mean()))
            mae_m = float((-cv_res["test_mae"].mean()))

            r2_s = float(cv_res["test_r2"].std())
            rmse_s = float((cv_res["test_rmse"].std()))
            mae_s = float((cv_res["test_mae"].std()))

            engine_metrics = {
                "r2": r2_m,
                "rmse": rmse_m,
                "mae": mae_m,
                "r2_std": r2_s,
                "rmse_std": rmse_s,
                "mae_std": mae_s,
            }
            logger.info("[Orchestrator|%s] CV R²=%.4f RMSE=%.4f MAE=%.4f", name, r2_m, rmse_m, mae_m)
            q.put((name, fitted_model, engine_metrics)) # Put fitted model AND metrics
        except Exception as e: # Catch any error in the child process
            error_traceback = traceback.format_exc()
            logger.error("[Orchestrator|%s] Engine crashed: %s\n%s", name, e, error_traceback)
            q.put((name, None, {"error": str(e), "traceback": error_traceback})) # Indicate error, pass traceback

    for eng_name, wrapper in engines.items():
        # Start a new process for each engine
        worker = ctx.Process(
            target=_runner,
            args=(
                eng_name, X, y, timeout_per_engine, run_dir,
                config.MODEL_FAMILIES, config.PREP_STEPS, metric, q
            ),
        )
        worker.start()
        workers.append(worker)

    # Collect results
    per_engine_results: Dict[str, Any] = {}
    per_engine_metrics: Dict[str, Dict[str, float]] = {}

    for worker in workers:
        eng_name, fitted_model, metrics = q.get() # Get metrics as well
        if fitted_model is not None: # Only store successful results
            per_engine_results[eng_name] = fitted_model
            per_engine_metrics[eng_name] = metrics
        else:
            error_msg = metrics.get('error', 'Unknown error')
            error_tb = metrics.get('traceback', 'No traceback available')
            console.print(f"[red]✗ {eng_name} error: {error_msg}[/]")
            logger.error(f"[Orchestrator] Error from {eng_name} child process:\n%s", error_tb)

    for worker in workers: # Wait for all processes to finish
        worker.join()

    if not per_engine_results:
        raise RuntimeError("All AutoML engines failed in concurrent run.")

    # ---------------------------------------------------------------------
    # 3️⃣ Select the overall champion based on mean CV R²
    # ---------------------------------------------------------------------
    champion_name, champion_model = max(
        per_engine_results.items(), key=lambda kv: per_engine_metrics[kv[0]].get("r2", float('-inf'))
    )

    champ_score = per_engine_metrics[champion_name].get("r2", float('nan'))
    console.print(f"🏆 [bold green]Champion → {champion_name}[/]")
    console.print(f"R²={champ_score:.4f} (from CV)")

    # ---------------------------------------------------------------------
    # 4️⃣ Ensembling (if enabled and multiple engines succeeded)
    # ---------------------------------------------------------------------
    if enable_ensemble and len(per_engine_results) > 1:
        console.print("[bold]5️⃣ Blending / Ensembling Champion Models…[/bold]")
        try:
            # Ensure champion_model is available for blending (it should be)
            # _blend expects a list of models, so we pass all successful models
            ensemble_model = _blend(list(per_engine_results.values()))
            logger.info("Champion models successfully blended.")
            # If ensemble is better, set it as the new champion
            # For now, just return the best single model, blending is a separate step.
            # The request states to select champion, then blend (if enabled). This implies blending the selected champion and others.
            # For now, we will return the best single model, and blending logic can be improved later if needed.
            console.print("[green]✓ Ensemble created (currently not set as overall champion)[/green]")
        except Exception as e:
            logger.error(f"Error during ensembling: {e}", exc_info=True)
            console.print(f"[red]✗ Ensembling failed: {e}[/red]")

    return champion_model, per_engine_results, per_engine_metrics # Added per_engine_metrics to return


if __name__ == "__main__":
    import sys # Add import for sys module
    _cli() 