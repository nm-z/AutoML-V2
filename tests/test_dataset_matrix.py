import json
import sys
from pathlib import Path

import pytest

from test_integration_engines import load_orchestrator

@pytest.mark.parametrize("ds", ["1", "2", "3"])
def test_cli_dataset_matrix(monkeypatch, tmp_path, ds):
    orch = load_orchestrator(monkeypatch)
    monkeypatch.setattr(orch, "_validate_components_availability", lambda: None)
    monkeypatch.setattr(orch, "_extract_pipeline_info", lambda m: [])

    called = {}
    def fake_meta_search(**kwargs):
        run_dir = Path(kwargs.get("run_dir"))
        run_dir.mkdir(parents=True, exist_ok=True)
        called["run_dir"] = run_dir
        class DummyModel:
            def predict(self, X):
                return [0]
        dummy = DummyModel()
        return dummy, {"autosklearn": dummy, "tpot": dummy, "autogluon": dummy}, {
            "autosklearn": {},
            "tpot": {},
            "autogluon": {},
        }
    monkeypatch.setattr(orch, "_meta_search_concurrent", fake_meta_search)
    monkeypatch.setattr(orch, "_meta_search_sequential", fake_meta_search)

    data = tmp_path / f"D{ds}-Predictors.csv"
    target = tmp_path / f"D{ds}-Targets.csv"
    data.write_text("a\n1\n")
    target.write_text("b\n1\n")

    monkeypatch.setattr(sys, "argv", [
        "orchestrator.py",
        "--data",
        str(data),
        "--target",
        str(target),
        "--all",
        "--no-ensemble",
    ])

    orch._cli()
    metrics = json.load(open(called["run_dir"] / "metrics.json"))
    assert set(metrics["run_meta"]["engines_invoked"]) == {"autogluon", "autosklearn", "tpot"}
    assert metrics["dataset"]["n_rows"] == 1
