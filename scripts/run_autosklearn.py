#!/usr/bin/env python
"""Auto-Sklearn Runner (Python 3.9 – env-as)

This script is meant to be executed from the *env-as* pyenv virtual
environment.  It expects that `auto-sklearn==0.15.0` along with its
supported dependency stack (NumPy, SciPy, scikit-learn 0.24.x, …) is
available.

Usage
-----
python run_autosklearn.py \
    --predictors DataSets/3/predictors.csv \
    --target-column y \
    --artifacts-dir 05_outputs/dataset_name \
    [--timeout 3600]

Parameters
~~~~~~~~~~
--predictors       Path to the predictors CSV file.
--target-column    Name of the target column *within* the predictors CSV.
--target-path      Alternative to --target-column: a CSV containing the
target values. If provided, the target column will be taken as the *only*
column from that file.
--artifacts-dir    Directory where model.pkl, metrics.json, and run.log will
be saved.  It will be created if missing.
--timeout          Optional wall-clock limit per Auto-Sklearn run.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from autosklearn.regression import AutoSklearnRegressor  # type: ignore
import logging
from rich.console import Console
from rich.tree import Tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold

RANDOM_STATE = 42
CV_REPEATS = 3
CV_SPLITS = 5
N_JOBS_CV = 12  # outer CV parallelism

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console(highlight=False, record=True)

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def _parse_args() -> argparse.Namespace:  # noqa: D401
    parser = argparse.ArgumentParser(description="Run Auto-Sklearn (regression)")
    parser.add_argument("--predictors", required=True, help="Path to predictors CSV")
    parser.add_argument("--target-column", help="Name of target column in predictors CSV")
    parser.add_argument("--target-path", help="Separate CSV containing target.")
    parser.add_argument("--artifacts-dir", required=True, help="Output directory for artifacts")
    parser.add_argument("--timeout", type=int, default=3600, help="Wall-clock limit (s)")
    return parser.parse_args()

def _load_data(args: argparse.Namespace):  # noqa: D401
    X: pd.DataFrame
    y: pd.Series

    logger.info("Loading predictors from %s", args.predictors)
    if args.target_path:
        logger.info("Loading target from %s", args.target_path)
        X = pd.read_csv(args.predictors, index_col=None)
        y = pd.read_csv(args.target_path, index_col=None).iloc[:, 0]
    else:
        if not args.target_column:
            logger.error("Either --target-column or --target-path must be provided")
            console.print("[red]Either --target-column or --target-path must be provided")
            sys.exit(1)
        df = pd.read_csv(args.predictors, index_col=None)
        y = df.pop(args.target_column)
        X = df
    logger.info("Loaded dataset with %d rows and %d columns", X.shape[0], X.shape[1])
    return X, y

def main() -> None:  # noqa: D401
    args = _parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Run directory: %s", artifacts_dir)

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    root = Tree("[bold cyan]Auto-Sklearn Runner[/bold cyan]")

    try:

        # ------------------------------------------------------------------
        # Data loading
        # ------------------------------------------------------------------
        root.add("Loading data…")
        X, y = _load_data(args)
        root.add(f"X shape = {X.shape}, y shape = {y.shape}")
        logger.info("Data loaded: %s rows", len(X))

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------
        rkf = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)

        model = AutoSklearnRegressor(
            time_left_for_this_task=args.timeout,
            seed=RANDOM_STATE,
            n_jobs=1,  # avoid nested parallelism
            metric="r2",
        )

        start_fit = time.perf_counter()
        logger.info("Starting Auto-Sklearn fitting")
        model.fit(X, y, dataset_name="AutoML-Harness-Dataset")
        fit_seconds = time.perf_counter() - start_fit
        logger.info("Fit completed in %.1fs", fit_seconds)
        root.add(f"[green]Fit completed in {fit_seconds:.1f}s")

    # ------------------------------------------------------------------
    # Cross-validation evaluation
    # ------------------------------------------------------------------
        cv_node = root.add("[bold]5×3 Repeated K-Fold evaluation…[/bold]")
        r2_scores, rmse_scores, mae_scores = [], [], []

        for fold_idx, (train_idx, test_idx) in enumerate(rkf.split(X), 1):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

            fold_model = model.clone()  # clone ensures same pipeline structure
            fold_model.fit(X_tr, y_tr)
            preds = fold_model.predict(X_te)

            r2 = r2_score(y_te, preds)
            rmse = _rmse(y_te, preds)
            mae = mean_absolute_error(y_te, preds)

            r2_scores.append(r2)
            rmse_scores.append(rmse)
            mae_scores.append(mae)

            logger.info(
                "Fold %02d: R2=%.4f RMSE=%.4f MAE=%.4f",
                fold_idx,
                r2,
                rmse,
                mae,
            )
            cv_node.add(
                f"Fold {fold_idx:02d}: R²={r2:.4f}  RMSE={rmse:.4f}  MAE={mae:.4f}"
            )

        metrics: Dict[str, Any] = {
            "cv": {
                "r2_mean": float(np.mean(r2_scores)),
                "r2_std": float(np.std(r2_scores)),
                "rmse_mean": float(np.mean(rmse_scores)),
                "rmse_std": float(np.std(rmse_scores)),
                "mae_mean": float(np.mean(mae_scores)),
                "mae_std": float(np.std(mae_scores)),
            },
            "fit_seconds": fit_seconds,
            "timestamp": datetime.utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Persist artifacts
    # ------------------------------------------------------------------
        model_path = artifacts_dir / "auto_sklearn_wrapper_champion.pkl"
        joblib.dump(model, model_path)
        root.add(f"[blue]Model saved → {model_path}")
        logger.info("Model saved to %s", model_path)

        metrics_path = artifacts_dir / "metrics_autosklearn.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        root.add(f"[blue]Metrics saved → {metrics_path}")
        logger.info("Metrics saved to %s", metrics_path)

        log_path = artifacts_dir / "autosklearn_run.log"
        log_path.write_text(console.export_text())
        root.add(f"[blue]Verbose log → {log_path}")
        logger.info("Run log written to %s", log_path)

        console.print(root)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Run failed: %s", exc)
        console.print(f"[red]Error: {exc}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
