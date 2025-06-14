"""Hyperparameter tuning utility using RandomizedSearchCV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import make_scorer, r2_score

from scripts.data_loader import load_data

RANDOM_STATE = 42
CV_SPLITS = 5
CV_REPEATS = 2


def tune_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = 20,
) -> Tuple[RandomForestRegressor, Dict[str, Any], float]:
    """Tune a RandomForestRegressor using RandomizedSearchCV."""
    param_dist = {
        "n_estimators": [50, 100, 150, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features": ["sqrt", "log2", None],
    }

    cv = RepeatedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE)
    estimator = RandomForestRegressor(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring=make_scorer(r2_score),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_, search.best_score_


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for RandomForestRegressor")
    parser.add_argument("--data", required=True, help="Path to predictors CSV")
    parser.add_argument("--target", required=True, help="Path to target CSV")
    parser.add_argument("--output", required=True, help="File to write best params as JSON")
    parser.add_argument("--n-iter", type=int, default=20, help="Number of search iterations")
    args = parser.parse_args()

    X, y = load_data(args.data, args.target)
    _, best_params, best_score = tune_random_forest(X, y, n_iter=args.n_iter)

    result = {"best_params": best_params, "best_score": best_score}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Best R^2: {best_score:.4f}")
    print(f"Parameters written to {args.output}")


if __name__ == "__main__":
    _cli() 