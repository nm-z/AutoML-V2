import importlib.util
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "hyperparameter_tuner",
    Path(__file__).resolve().parents[1] / "scripts" / "hyperparameter_tuner.py",
)
tuner = importlib.util.module_from_spec(spec)
sys.modules["hyperparameter_tuner"] = tuner
spec.loader.exec_module(tuner)
from hyperparameter_tuner import tune_random_forest


def test_tune_random_forest_runs():
    X = pd.DataFrame({"a": range(10), "b": range(10, 20)})
    y = pd.Series(range(10))
    _, params, score = tune_random_forest(X, y, n_iter=1)
    assert isinstance(params, dict)
    assert isinstance(score, float) 