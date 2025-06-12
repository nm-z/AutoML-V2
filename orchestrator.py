"""AutoML Orchestrator – Meta-Search Controller

This module coordinates multiple AutoML engines, ensuring each one searches
*exactly* the approved search-space and then selects the highest-R² champion
under a shared validation split.

Only the orchestration logic lives here; engine-specific glue code resides in
``engines/``.
"""
from __future__ import annotations

import time
from pathlib import Path
import argparse
import json
import random
from datetime import datetime
import traceback
from typing import Any, Dict, Tuple

import pandas as pd
from rich.console import Console
from rich.tree import Tree

import scripts.config as config
from engines import discover_available

import numpy as np

# Scoring & CV utilities
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    mean_absolute_error,
)

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
    from sklearn.metrics import mean_squared_error as _mse
    import numpy as _np

    return _np.sqrt(_mse(y_true, y_pred))

def _meta_search_legacy(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    artifacts_dir: Path | str = "05_outputs",
    timeout_per_engine: int | None = None,
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
# NEW – Public API shim that supports both the *old* signature used across the
#       existing codebase **and** the *new* 4-parameter signature mandated by
#       the latest specification (config_path, X, y, budget).
# ---------------------------------------------------------------------------


def meta_search(*args, **kwargs):  # noqa: D401 – flexible dispatcher
    """Dispatch to the correct implementation based on positional args.

    Supported call patterns:

    1. *Old* signature::

           meta_search(X, y, artifacts_dir="…", timeout_per_engine=…)

       This variant is kept for backward-compatibility with existing demo
       scripts and notebooks in this repository.

    2. *New* spec-compliant signature::

           meta_search(config_path, X, y, budget)

       The ``config_path`` is currently **ignored** because the orchestrator
       already imports a global immutable ``scripts.config`` module at
       import-time.  The provided ``budget`` is forwarded to the legacy
       implementation via the ``timeout_per_engine`` kwarg so that behaviour
       remains identical.
    """

    if len(args) >= 2 and isinstance(args[0], pd.DataFrame):
        # Detected legacy call: (X, y, …)
        X = args[0]
        y = args[1]
        remaining = args[2:]
        return _meta_search_legacy(X, y, *remaining, **kwargs)

    if len(args) == 4 and isinstance(args[0], (str, Path)):
        # Detected new spec call: (config_path, X, y, budget)
        cfg_path, X, y, fallback_budget = args  # noqa: N806 – positional unpack

        # ------------------------------------------------------------------
        # 1️⃣ Parse JSON configuration ← might override *budget* & *metric*
        # ------------------------------------------------------------------
        cfg_path = Path(cfg_path)
        try:
            cfg_payload = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"config_path JSON invalid: {exc}") from exc

        metric_name: str = (
            cfg_payload.get("metric")
            or cfg_payload.get("objective")
            or config.DEFAULT_METRIC
        )

        budget_sec: int = int(cfg_payload.get("budget", fallback_budget)) if fallback_budget else int(
            cfg_payload.get("budget", config.WALLCLOCK_LIMIT_SEC)
        )

        # Enforce positive, non-zero wall-clock limit
        if budget_sec <= 0:
            budget_sec = config.WALLCLOCK_LIMIT_SEC

        return _meta_search_concurrent(
            X=X,
            y=y,
            metric=metric_name,
            timeout_per_engine=budget_sec,
            **kwargs,
        )

    raise TypeError(
        "meta_search() received unsupported arguments. "
        "Expected either (X, y, …) or (config_path, X, y, budget)."
    )


# ---------------------------------------------------------------------------
# Helper – validate that mandatory model & preprocessor components exist
# ---------------------------------------------------------------------------


def _validate_components_availability() -> None:
    """Ensure that every component declared in the global ``config`` module
    physically exists inside the *components* directory tree.  Fail fast if
    any mandatory file is missing so that the user can fix the installation
    before the expensive AutoML search even starts.
    """

    missing: list[str] = []

    # Validate model wrappers
    model_root = Path("components/models")
    for model_name in config.MODEL_FAMILIES:
        if not model_root.joinpath(f"{model_name}.py").exists():
            missing.append(f"models/{model_name}.py")

    # Validate preprocessors
    preproc_root = Path("components/preprocessors")
    for prep in config.PREP_STEPS:
        # Some preprocessors live one level deeper (scalers/, outliers/, …)
        pattern = list(preproc_root.rglob(f"{prep}.py"))
        if not pattern:
            missing.append(f"preprocessors/**/{prep}.py")

    if missing:
        formatted = "\n".join(f" • {p}" for p in missing)
        raise FileNotFoundError(
            "Mandatory components are missing from the workspace:\n" + formatted
        )


__all__ = ["meta_search", "_meta_search_legacy"]


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------


def _cli() -> None:
    """Entry-point executed via ``python orchestrator.py …``.

    The CLI intentionally remains minimal – it only orchestrates dataset
    loading, the optional wall-clock limit, and the output directory.  All
    heavy-lifting is delegated to :pyfunc:`meta_search`.
    """

    parser = argparse.ArgumentParser(description="AutoML Orchestrator")
    parser.add_argument("--data", help="Path to predictors CSV file (overrides --dataset-id)")
    parser.add_argument("--target", help="Path to target CSV file (overrides --dataset-id)")
    parser.add_argument(
        "--dataset-id",
        type=int,
        help="Numeric dataset identifier to look-up inside config.json instead of --data/--target",
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="Per-engine wall-clock limit in seconds (overrides config)",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="Dataset identifier – used to name the 05_outputs/<dataset> directory",
    )

    args, _ = parser.parse_known_args()

    # ------------------------------------------------------------------
    # Dataset selection – priority order:
    #   1. --data & --target provided  → use them directly.
    #   2. --dataset-id provided      → look-up in config.json.
    # ------------------------------------------------------------------

    if args.data and args.target:
        predictors_path = Path(args.data)
        target_path = Path(args.target)

        if not predictors_path.exists() or not target_path.exists():
            parser.error("Both --data and --target must exist on disk.")

        dataset_name = args.dataset_name or predictors_path.parent.name
        out_dir = Path("05_outputs") / dataset_name

        X = pd.read_csv(predictors_path)
        y = pd.read_csv(target_path).iloc[:, 0]

    elif args.dataset_id is not None:
        cfg_path = Path(__file__).with_name("config.json")
        if not cfg_path.exists():
            parser.error("--dataset-id passed but config.json not found next to orchestrator.py")

        try:
            cfg = json.loads(cfg_path.read_text())
            dataset_entry = next(d for d in cfg["datasets"] if d["id"] == args.dataset_id)
        except (KeyError, StopIteration):
            parser.error(f"Dataset ID {args.dataset_id} not found in config.json")
        except json.JSONDecodeError as exc:
            parser.error(f"config.json is invalid: {exc}")

        predictors_csv = Path(dataset_entry["path"])
        target_path = dataset_entry.get("target_path")
        target_column = dataset_entry.get("target_column")

        if not predictors_csv.exists():
            parser.error(f"CSV path specified in config.json does not exist: {predictors_csv}")

        dataset_name = args.dataset_name or dataset_entry.get("name", f"dataset{args.dataset_id}")
        out_dir = Path("05_outputs") / dataset_name

        if target_path:
            if not Path(target_path).exists():
                parser.error(f"Target path specified in config.json does not exist: {target_path}")
            X = pd.read_csv(predictors_csv)
            y = pd.read_csv(target_path).iloc[:, 0]
        elif target_column:
            df = pd.read_csv(predictors_csv)
            if target_column not in df.columns:
                parser.error(f"target_column '{target_column}' not found in CSV header")
            y = df[target_column]
            X = df.drop(columns=[target_column])
        else:
            parser.error("Dataset entry must specify either 'target_path' or 'target_column'.")

        # Optionally persist shape back to config for quick reference
        dataset_entry["shape"] = {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}
        try:
            cfg_path.write_text(json.dumps(cfg, indent=2))
        except Exception:
            pass  # non-fatal, just informative

    else:
        parser.error("Either --data & --target OR --dataset-id must be provided.")

    # Hold-out split (20%) for unbiased final evaluation
    from sklearn.model_selection import train_test_split

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE, shuffle=True
    )

    champion, _ = meta_search(
        X_train,
        y_train,
        artifacts_dir=out_dir,
        timeout_per_engine=args.time,
    )

    # Evaluate champion on the unseen hold-out set
    preds = champion.predict(X_valid)
    from sklearn.metrics import (
        r2_score,
        mean_absolute_error as _mae,
    )

    from sklearn.metrics import mean_squared_error as _mse

    r2 = r2_score(y_valid, preds)
    rmse = _rmse(y_valid, preds)
    mae = _mae(y_valid, preds)

    summary = {
        "r2_holdout": r2,
        "rmse_holdout": rmse,
        "mae_holdout": mae,
    }

    # Append to existing metrics.json if present
    metrics_path = out_dir / "metrics.json"
    try:
        if metrics_path.exists():
            existing = json.loads(metrics_path.read_text())
        else:
            existing = {}
        existing["holdout"] = summary
        metrics_path.write_text(json.dumps(existing, indent=2))
    except Exception:
        pass  # non-fatal – CV metrics already persisted inside meta_search

    console.print(
        f"[bold green]Hold-out results[/]: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
    )


# ---------------------------------------------------------------------------
# NEW – Concurrent Meta-Search implementation (spec-compliant)
# ---------------------------------------------------------------------------


import multiprocessing as _mp
from multiprocessing.queues import Queue as _MPQueue


def _get_automl_engine(name: str):
    """Return the *imported* wrapper module for *name* or raise.

    This helper provides a thin abstraction over :pyfunc:`engines.discover_available`
    so that unit-tests can easily monkey-patch engine discovery without touching
    the global orchestrator state.
    """

    all_available = discover_available()
    try:
        return all_available[name]
    except KeyError as exc:
        raise ValueError(f"Unknown or unavailable AutoML engine: '{name}'") from exc


def _score(
    model: Any,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    metric: str = config.DEFAULT_METRIC,
):
    """Compute *metric* on the provided validation set.

    Supported metrics: "r2" (higher better), "rmse" & "mae" (lower better).
    """

    preds = model.predict(X_valid)
    if metric == "r2":
        from sklearn.metrics import r2_score as _r2

        return float(_r2(y_valid, preds))
    if metric == "rmse":
        return float(_rmse(y_valid, preds)) * -1  # negative so higher is better
    if metric == "mae":
        from sklearn.metrics import mean_absolute_error as _mae

        return float(_mae(y_valid, preds)) * -1

    raise ValueError(f"Unsupported metric: {metric}")


def _fit_engine(
    eng_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    timeout_sec: int,
) -> Any:
    """Fit a single AutoML engine and return the trained *model* instance."""

    wrapper = _get_automl_engine(eng_name)

    model = wrapper.fit_engine(
        X_train,
        y_train,
        model_families=config.MODEL_FAMILIES,
        prep_steps=config.PREP_STEPS,
        seed=config.RANDOM_STATE,
        timeout_sec=timeout_sec,
    )

    return model


class _MeanEnsembleRegressor:  # noqa: D401 – simple averaging ensemble
    """A minimalist ensemble that averages predictions of pre-fitted models."""

    def __init__(self, models: list[Any]):
        self.models = models

    # fit retained for API compatibility – does nothing because models are pre-fit
    def fit(self, *_args, **_kwargs):  # noqa: D401 – no-op fit
        return self

    def predict(self, X):  # type: ignore[valid-type] – depends on underlying models
        import numpy as _np

        preds = _np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)


def _blend(champions: list[Any]):
    """Return a simple mean-ensemble over *champions* (already fitted)."""

    if not champions:
        raise ValueError("_blend() requires at least one champion model")
    if len(champions) == 1:
        return champions[0]
    return _MeanEnsembleRegressor(champions)


def _meta_search_concurrent(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    metric: str,
    timeout_per_engine: int,
    artifacts_dir: Path | str = "05_outputs",
):
    """Spec-compliant concurrent meta-search.

    This implementation spawns *one* subprocess per available AutoML engine so
    that searches run in parallel while avoiding nested parallelism inside the
    engines themselves (each wrapper enforces n_jobs=1 at model level).
    """

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    engines_available = list(discover_available().keys())
    if not engines_available:
        raise RuntimeError("No AutoML engine wrappers could be imported – aborting.")

    root = Tree("[bold cyan]AutoML Meta-Search (concurrent)[/bold cyan]")
    root.add(f"Metric: {metric}")
    root.add(f"Timeout per engine: {timeout_per_engine}s")

    # ------------------------------------------------------------------
    # 1️⃣ Spawn processes – each will fit its engine and return the model.
    # ------------------------------------------------------------------
    ctx = _mp.get_context("spawn")
    queue: _MPQueue = ctx.Queue()

    def _runner(name: str, X_obj, y_obj, t_sec, q: _MPQueue):  # noqa: N801 – nested func
        try:
            mdl = _fit_engine(name, X_obj, y_obj, timeout_sec=t_sec)
            q.put((name, mdl, None))
        except Exception as exc:  # noqa: BLE001
            q.put((name, None, traceback.format_exc()))

    procs = []
    for eng_name in engines_available:
        p = ctx.Process(target=_runner, args=(eng_name, X, y, timeout_per_engine, queue))
        p.start()
        procs.append(p)

    # ------------------------------------------------------------------
    # 2️⃣ Collect results – join with timeout & handle stragglers.
    # ------------------------------------------------------------------
    for p in procs:
        p.join(timeout_per_engine + 5)  # small grace period
        if p.is_alive():
            p.terminate()
            root.add(f"[red]✗ {p.name} exceeded timeout and was terminated")

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    while not queue.empty():
        eng_name, model, err = queue.get()
        if err is not None or model is None:
            errors[eng_name] = err or "Unknown error"
            root.add(f"[red]✗ {eng_name} error during fit – see traceback below")
            root.add(errors[eng_name])
        else:
            results[eng_name] = model
            root.add(f"[green]✔ {eng_name} finished")

    if not results:
        console.print(root)
        raise RuntimeError("All AutoML engines failed during fitting phase.")

    # ------------------------------------------------------------------
    # 3️⃣ Cross-validation – identical to legacy implementation.
    # ------------------------------------------------------------------
    cv_node = root.add("5×3 Repeated K-Fold evaluation…")

    rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=config.RANDOM_STATE)

    scoring_map = {
        "r2": {
            "r2": "r2",
        },
        "rmse": {
            "rmse": make_scorer(_rmse, greater_is_better=False),
        },
        "mae": {
            "mae": make_scorer(mean_absolute_error, greater_is_better=False),
        },
    }

    if metric not in scoring_map:
        raise ValueError(f"Unsupported metric: {metric}")

    per_engine_metrics: dict[str, dict[str, float]] = {}

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

            # Extract primary metric – always the first key in dict
            _key = next(iter(scoring_map[metric].keys()))
            mean_val = float(cv_res[f"test_{_key}"].mean())
            std_val = float(cv_res[f"test_{_key}"].std())

            per_engine_metrics[eng_name] = {f"{metric}": mean_val, f"{metric}_std": std_val}

            pretty = f"{metric.upper()}={abs(mean_val):.4f}" if metric != "r2" else f"R²={mean_val:.4f}"
            sub.add(f"[green]✓ {pretty}")
        except Exception as exc:  # noqa: BLE001 – fail fast
            sub.add(f"[red]✗ error during CV: {exc}")
            console.print(root)
            raise

    # ------------------------------------------------------------------
    # 4️⃣ Champion selection
    # ------------------------------------------------------------------
    if metric == "r2":
        champ_name, champ_model = max(results.items(), key=lambda kv: per_engine_metrics[kv[0]][metric])
    else:  # rmse / mae – lower is better but we stored negative values
        champ_name, champ_model = max(results.items(), key=lambda kv: per_engine_metrics[kv[0]][metric])

    champ_score = per_engine_metrics[champ_name][metric]
    root.add(f"🏆 [bold green]Champion → {champ_name}[/]").add(f"mean CV {metric.upper()} = {abs(champ_score):.4f}")

    # ------------------------------------------------------------------
    # 5️⃣ Persistence
    # ------------------------------------------------------------------
    for eng_name, mdl in results.items():
        file_path = Path(artifacts_dir) / f"{eng_name}_champion.pkl"
        try:
            import joblib

            joblib.dump(mdl, file_path)
        except Exception:  # noqa: BLE001 – nonfatal
            pass

    metrics_payload = {
        "timestamp": datetime.utcnow().isoformat(),
        "metric": metric,
        "cv": per_engine_metrics,
        "champion": champ_name,
    }

    try:
        (Path(artifacts_dir) / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
    except Exception:
        pass

    # Persist Rich tree log
    try:
        (Path(artifacts_dir) / "run.log").write_text(console.export_text())
    except Exception:
        pass

    console.print(root)

    return champ_model, results


# ---------------------------------------------------------------------------
# Module execution – ensure all definitions above are available before running
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    try:
        _cli()
    except Exception:
        console.print("[red]Uncaught exception:")
        console.print(traceback.format_exc())
        raise 