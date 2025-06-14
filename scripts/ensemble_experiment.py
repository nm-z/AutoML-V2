#!/usr/bin/env python
"""Simple weighted ensemble experiment using engine champions.

This script demonstrates how to combine the champion models from all
three AutoML engines with a linear-weighted ensemble. It runs the
orchestrator on the provided dataset, collects predictions from each
engine on a hold-out set, fits a linear regression as the meta-model,
and reports the resulting R² score.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from scripts.data_loader import load_data
import orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted ensemble experiment")
    parser.add_argument("--data", required=True, help="Path to predictors CSV")
    parser.add_argument("--target", required=True, help="Path to target CSV")
    parser.add_argument(
        "--time", type=int, default=60, help="Time limit per engine (s)"
    )
    args = parser.parse_args()

    X, y = load_data(args.data, args.target)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Run all engines without ensembling to get individual champions
    champion, engines, metrics = orchestrator._meta_search_concurrent(
        X=X_train,
        y=y_train,
        run_dir=Path("05_outputs") / "ensemble_experiment",
        timeout_per_engine=args.time,
        metric="r2",
        enable_ensemble=False,
        n_cpus=os.cpu_count() or 1,
    )

    # Gather predictions from each engine
    train_preds = []
    test_preds = []
    for name, model in engines.items():
        train_preds.append(model.predict(X_train))
        test_preds.append(model.predict(X_test))
        r2_ind = r2_score(y_test, test_preds[-1])
        print(f"{name} hold-out R²: {r2_ind:.4f}")

    stack_X_train = np.column_stack(train_preds)
    stack_X_test = np.column_stack(test_preds)

    meta_model = LinearRegression()
    meta_model.fit(stack_X_train, y_train)
    ensemble_preds = meta_model.predict(stack_X_test)
    ensemble_r2 = r2_score(y_test, ensemble_preds)
    print(f"Weighted ensemble hold-out R²: {ensemble_r2:.4f}")


if __name__ == "__main__":
    main() 