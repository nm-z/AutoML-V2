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
from typing import Any, Dict, Tuple, Sequence

import pandas as pd
from rich.console import Console
from rich.tree import Tree

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
# Public API
# ---------------------------------------------------------------------------

def _rmse(y_true, y_pred):
    """Root Mean Squared Error helper – always returns positive value."""
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _meta_search_legacy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    artifacts_dir: Path | str = "05_outputs",
    timeout_per_engine: int | None = None,
    metric: str = config.DEFAULT_METRIC,
    enable_ensemble: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Run each available AutoML engine and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    artifacts_dir
        Where to persist every artifact (models, logs, CV results, …).
    timeout_per_engine
        Optional wall-clock limit that overrides the global constant.

    Returns
    -------
    champion
        The top-performing model *object* (already fitted).
    per_engine_results
        ``{engine_name: fitted_model}`` for *every* engine that succeeded.
    """

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

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
            file_path = artifacts_dir / f"{eng_name}_champion.pkl"
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
        f"mean CV R² = {champ_score:.4f}"
    )

    # ---------------------------------------------------------------------
    # 5️⃣ Persist metrics & logs
    # ---------------------------------------------------------------------
    metrics_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "cv": per_engine_metrics,
        "champion": champion_name,
    }

    metrics_path = artifacts_dir / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        root.add(f"[blue]📄 metrics saved → {metrics_path}")
    except Exception as exc:  # noqa: BLE001 – log & continue
        root.add(f"[yellow]⚠ could not save metrics: {exc}")

    # Persist console log
    log_path = artifacts_dir / "run.log"
    try:
        log_path.write_text(console.export_text())
    except Exception:
        pass  # non-fatal

    console.print(root)

    return champion_model, results


# ---------------------------------------------------------------------------
# Public API – CLI Entry-point
# ---------------------------------------------------------------------------

def meta_search(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    artifacts_dir: Path | str = "05_outputs",
    timeout_per_engine: int | None = None,
    metric: str = config.DEFAULT_METRIC,
    enable_ensemble: bool = False,
) -> Tuple[Any, Dict[str, Any]]:
    """Run each available AutoML engine and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    artifacts_dir
        Where to persist every artifact (models, logs, CV results, …).
    timeout_per_engine
        Optional wall-clock limit that overrides the global constant.
    metric
        The evaluation metric to use (e.g., 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error').

    Returns
    -------
    champion
        The top-performing model *object* (already fitted).
    per_engine_results
        ``{engine_name: fitted_model}`` for *every* engine that succeeded.
    """
    # Wrap the existing _meta_search_legacy for now to gradually refactor.
    # The actual implementation of concurrent execution will replace this.
    return _meta_search_legacy(X, y, artifacts_dir=artifacts_dir, timeout_per_engine=timeout_per_engine, metric=metric)


def _validate_components_availability() -> None:
    """Ensure required component files exist for dynamic import."""

    root = Tree("[bold yellow]Validating components…[/bold yellow]")

    # Expected structure & file names
    EXPECTED_FILES = [
        "components/preprocessors/scalers/RobustScaler.py",
        "components/preprocessors/scalers/StandardScaler.py",
        "components/preprocessors/scalers/QuantileTransform.py",
        "components/preprocessors/dimensionality/PCA.py",
        "components/preprocessors/outliers/KMeansOutlier.py",
        "components/preprocessors/outliers/IsolationForest.py",
        "components/preprocessors/outliers/LocalOutlierFactor.py",
        "components/models/Ridge.py",
        "components/models/Lasso.py",
        "components/models/ElasticNet.py",
        "components/models/SVR.py",
        "components/models/DecisionTree.py",
        "components/models/RandomForest.py",
        "components/models/ExtraTrees.py",
        "components/models/GradientBoosting.py",
        "components/models/AdaBoost.py",
        "components/models/MLP.py",
        "components/models/XGBoost.py",
        "components/models/LightGBM.py",
    ]

    missing_files = []
    for f_path in EXPECTED_FILES:
        if not Path(f_path).exists():
            missing_files.append(f_path)

    if missing_files:
        msg = f"Missing required component files: {missing_files}"
        root.add(f"[red]✗ {msg}")
        console.print(root)
        raise FileNotFoundError(msg)
    else:
        root.add("[green]✓ All component files found.")
        console.print(root)


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

    # Load configuration from JSON file if provided
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
            # Override CLI args with config file values if present
            args.data = config_data.get("data_path", args.data)
            args.target = config_data.get("target_path", args.target)
            args.time = config_data.get("time_limit", args.time)
            args.output_dir = config_data.get("output_dir", args.output_dir)
            args.metric = config_data.get("metric", args.metric)
            args.ensemble = config_data.get("ensemble", args.ensemble)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {args.config}")
            sys.exit(2)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in configuration file: {args.config}")
            sys.exit(2)

    # Load data using the new data_loader module
    try:
        X, y = load_data(args.data, args.target)
    except Exception as e:
        logger.error(f"Error loading data: {e}", exc_info=True)
        sys.exit(2) # Exit code for data loading error

    # Determine dataset name for artifacts directory
    dataset_name = Path(args.data).parent.name  # e.g., '3' from 'DataSets/3/predictors.csv'
    current_run_artifacts_dir = Path(args.output_dir) / dataset_name
    current_run_artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting AutoML meta-search...")
    logger.info("Dataset: %s", dataset_name)
    logger.info("Predictors path: %s", args.data)
    logger.info("Target path: %s", args.target)
    logger.info("Time limit per engine: %d seconds", args.time)
    logger.info("Output directory: %s", current_run_artifacts_dir)
    logger.info("Evaluation metric: %s", args.metric)

    try:
        if args.all:
            champion_model, _ = _meta_search_concurrent(
                X=X,
                y=y,
                artifacts_dir=current_run_artifacts_dir,
                timeout_per_engine=args.time,
                metric=args.metric,
                enable_ensemble=args.ensemble and not args.no_ensemble,
            )
        else:
            champion_model, _ = _meta_search_legacy(
                X=X,
                y=y,
                artifacts_dir=current_run_artifacts_dir,
                timeout_per_engine=args.time,
                metric=args.metric,
                enable_ensemble=args.ensemble and not args.no_ensemble,
            )
        logger.info("AutoML meta-search completed successfully.")
        sys.exit(0)  # Success exit code
    except RuntimeError as e:
        logger.error(f"AutoML meta-search failed: {e}")
        if "timeout" in str(e).lower():
            sys.exit(3) # Timeout exit code
        else:
            sys.exit(2) # General crash exit code
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True) # Log traceback
        sys.exit(2) # General crash exit code


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
    artifacts_dir: Path | str = "05_outputs",
    enable_ensemble: bool,
) -> Tuple[Any, Dict[str, Any]]:
    """Run each available AutoML engine concurrently and return the best model.

    Parameters
    ----------
    X, y
        Tabular data split into features and target.
    metric
        The evaluation metric to use (e.g., 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error').
    timeout_per_engine
        Wall-clock limit per engine in seconds.
    artifacts_dir
        Where to persist every artifact (models, logs, CV results, …).
    enable_ensemble
        Whether to create an ensemble of successful models.

    Returns
    -------
    champion
        The top-performing model *object* (already fitted).
    per_engine_results
        ``{engine_name: fitted_model}`` for *every* engine that succeeded.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a separate logs directory within artifacts_dir
    logs_dir = artifacts_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    root = Tree("[bold cyan]AutoML Meta-Search (concurrent)[/bold cyan]")
    root.add(f"Metric: {metric}")
    root.add(f"Timeout per engine: {timeout_per_engine}s")

    # ------------------------------------------------------------------
    # 0️⃣ Sanity checks – validate workspace invariants
    # ------------------------------------------------------------------
    try:
        _validate_components_availability()
    except Exception as exc:
        console.print(f"[red]Workspace validation failed: {exc}")
        raise

    # ------------------------------------------------------------------
    # 1️⃣ Discover available engines
    # ------------------------------------------------------------------
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
        # Use _get_automl_engine to get the class directly
        try:
            _get_automl_engine(name) # Check if it can be resolved
            discover_node.add(f"[green]✔ {name}")
            engines[name] = _get_automl_engine(name) # Store the class, not the module
        except ValueError:
            discover_node.add(f"[yellow]⚠ Skipping {name} - wrapper class not found or unknown.")

    if not engines:
        console.print(root)
        raise RuntimeError("No AutoML engine wrappers could be imported – aborting.")

    # ------------------------------------------------------------------
    # 2️⃣ Spawn processes – each will fit its engine and return the model.
    # ------------------------------------------------------------------
    ctx = _mp.get_context("spawn")
    queue: _MPQueue = ctx.Queue()

    def _runner(name: str, X_obj, y_obj, t_sec, r_dir, model_families, prep_steps, met, q: _MPQueue):
        # Configure logging for the child process
        log_file = r_dir / f"{name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging_level)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)

        try:
            mdl = _fit_engine(name, X_obj, y_obj, timeout_sec=t_sec, run_dir=r_dir, model_families=model_families, prep_steps=prep_steps, metric=met)
            # Call export method here to save engine-specific artifacts
            if hasattr(mdl, 'export'):
                mdl.export(r_dir) # Each engine exports to its own run_dir
            q.put((name, mdl, None))
        except Exception as exc:
            logger.exception("Error in %s engine process:", name) # Log exception in child process
            q.put((name, None, traceback.format_exc()))
        finally:
            logger.removeHandler(file_handler) # Clean up handler
            file_handler.close()

    procs = []
    for eng_name in engines.keys():
        # Create a unique run directory for each engine's artifacts and logs
        engine_run_dir = artifacts_dir / eng_name
        engine_run_dir.mkdir(parents=True, exist_ok=True)

        p = ctx.Process(target=_runner, args=(
            eng_name, X, y, timeout_per_engine, engine_run_dir, config.MODEL_FAMILIES, config.PREP_STEPS, metric, queue
        ))
        p.start()
        procs.append(p)

    # ------------------------------------------------------------------
    # 3️⃣ Collect results – join with timeout & handle stragglers.
    # ------------------------------------------------------------------
    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    # Wait for all processes to finish with a global timeout
    for p in procs:
        p.join(timeout_per_engine + 60)  # Add a grace period for cleanup

    # Collect results from the queue
    while not queue.empty():
        eng_name, model, err = queue.get()
        if err is not None or model is None:
            errors[eng_name] = err or "Unknown error"
            root.add(f"[red]✗ {eng_name} error during fit – see {logs_dir / f'{eng_name}.log'}[/]")
            logger.error("[%s] Engine failed: %s", eng_name, errors[eng_name])
        else:
            results[eng_name] = model
            root.add(f"[green]✔ {eng_name} finished[/]")
            logger.info("[%s] Engine completed successfully.", eng_name)

    if not results:
        console.print(root)
        raise RuntimeError("All AutoML engines failed during fitting phase.")

    # ------------------------------------------------------------------
    # 4️⃣ Cross-validation – 5×3 Repeated K-Fold
    # ------------------------------------------------------------------
    cv_node = root.add("5×3 Repeated K-Fold evaluation…")

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_STATE)

    scoring_map = {
        "r2": {"r2": "r2"},
        "neg_mean_squared_error": {"rmse": make_scorer(_rmse, greater_is_better=False)},
        "neg_mean_absolute_error": {"mae": make_scorer(mean_absolute_error, greater_is_better=False)},
    }

    if metric not in scoring_map:
        raise ValueError(f"Unsupported metric: {metric}")

    per_engine_metrics: Dict[str, Dict[str, float]] = {}
    successful_models = {}

    for eng_name, model in results.items():
        sub = cv_node.add(f"{eng_name} ▸ CV evaluating…")
        try:
            cv_res = cross_validate(
                model,
                X,
                y,
                cv=rkf,
                scoring=scoring_map[metric],
                n_jobs=config.N_JOBS_CV,
                error_score="raise",
            )

            _key = next(iter(scoring_map[metric].keys()))
            mean_val = float(cv_res[f"test_{_key}"].mean())
            std_val = float(cv_res[f"test_{_key}"].std())

            per_engine_metrics[eng_name] = {f"{metric}": mean_val, f"{metric}_std": std_val}
            successful_models[eng_name] = model # Add to successful models for champion selection

            pretty_score = abs(mean_val) if metric != "r2" else mean_val
            sub.add(f"[green]✓ {metric.upper()}={pretty_score:.4f}")
        except Exception as exc:
            sub.add(f"[red]✗ error during CV: {exc}")
            logger.error("[%s] Error during CV for %s: %s", self.__class__.__name__, eng_name, exc, exc_info=True)

    if not successful_models:
        console.print(root)
        raise RuntimeError("No AutoML engines completed CV successfully.")

    # ------------------------------------------------------------------
    # 5️⃣ Champion selection – based on the specified metric
    # ------------------------------------------------------------------
    # Sort models by metric. If metric is 'r2', higher is better. Otherwise, for negative metrics, higher (less negative) is better.
    if metric == "r2":
        champion_name, champion_model = max(successful_models.items(), key=lambda kv: per_engine_metrics[kv[0]][metric])
    else: # neg_mean_squared_error, neg_mean_absolute_error
        champion_name, champion_model = max(successful_models.items(), key=lambda kv: per_engine_metrics[kv[0]][metric])

    champ_score = per_engine_metrics[champion_name][metric]
    root.add(f"🏆 [bold green]Champion → {champion_name}[/]").add(
        f"mean CV {metric.upper()} = {abs(champ_score):.4f}"
    )
    
    # Optional: Create an ensemble of successful models
    blended_model = _blend(list(successful_models.values()))
    if blended_model and enable_ensemble:
        # Score ensemble on the training data for now (placeholder, ideally on separate validation set)
        ensemble_score_val = _score(blended_model, X, y, metric=metric)
        root.add(f"✨ [bold magenta]Ensemble (Mean)[/]").add(f"mean CV {metric.upper()} = {abs(ensemble_score_val):.4f}")
        # Decide if ensemble is the new champion based on metric
        if metric == "r2" and ensemble_score_val > champ_score:
            champion_model = blended_model
            champion_name = "Ensemble_Mean"
            root.add(f"🏆 [bold green]New Champion → {champion_name}[/]").add(f"mean CV R² = {ensemble_score_val:.4f}")
        elif metric != "r2" and ensemble_score_val > champ_score: # For negative metrics, greater (less negative) is better
            champion_model = blended_model
            champion_name = "Ensemble_Mean"
            root.add(f"🏆 [bold green]New Champion → {champion_name}[/]").add(f"mean CV {metric.upper()} = {abs(ensemble_score_val):.4f}")

    # ------------------------------------------------------------------
    # 6️⃣ Persistence
    # ------------------------------------------------------------------
    # Persist metrics.json
    metrics_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "metric": metric,
        "cv": per_engine_metrics,
        "champion": champion_name,
        "champion_score": abs(champ_score) if metric != "r2" else champ_score,
    }

    metrics_path = artifacts_dir / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
        root.add(f"[blue]📄 metrics saved → {metrics_path}")
    except Exception as exc:
        root.add(f"[yellow]⚠ could not save metrics: {exc}")
        logger.error("Error saving metrics.json: %s", exc, exc_info=True)

    # Persist console log
    overall_log_path = logs_dir / "orchestrator_run.log"
    try:
        overall_log_path.write_text(console.export_text())
        root.add(f"[blue]📄 Orchestrator log saved → {overall_log_path}")
    except Exception as exc:
        root.add(f"[yellow]⚠ could not save orchestrator log: {exc}")
        logger.error("Error saving orchestrator log: %s", exc, exc_info=True)

    # Persist the overall champion model
    champion_pkl_path = artifacts_dir / "overall_champion.pkl"
    try:
        import joblib
        joblib.dump(champion_model, champion_pkl_path)
        root.add(f"[blue]🔖 Overall champion model saved → {champion_pkl_path}")
    except Exception as exc:
        root.add(f"[yellow]⚠ could not save overall champion model: {exc}")
        logger.error("Error saving overall champion model: %s", exc, exc_info=True)

    console.print(root)

    return champion_model, results


if __name__ == "__main__":
    import sys # Add import for sys module
    _cli() 