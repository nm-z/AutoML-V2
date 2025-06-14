import importlib
from pathlib import Path
import sys
import types
import pytest


def load_orchestrator(monkeypatch):
    sys.modules.pop("orchestrator", None)

    # Stub heavy dependencies
    pandas = types.ModuleType("pandas")
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(seed=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    rich_console = types.ModuleType("rich.console")
    class DummyConsole:
        def __init__(self, *a, **k):
            pass
        def log(self, *a, **k):
            pass
    rich_console.Console = DummyConsole
    monkeypatch.setitem(sys.modules, "rich.console", rich_console)

    rich_tree = types.ModuleType("rich.tree")
    rich_tree.Tree = type("Tree", (), {})
    monkeypatch.setitem(sys.modules, "rich.tree", rich_tree)

    sklearn = types.ModuleType("sklearn")
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    pipe_mod = types.ModuleType("sklearn.pipeline")
    pipe_mod.Pipeline = object
    monkeypatch.setitem(sys.modules, "sklearn.pipeline", pipe_mod)
    msel = types.ModuleType("sklearn.model_selection")
    msel.RepeatedKFold = object
    msel.cross_validate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", msel)
    metrics = types.ModuleType("sklearn.metrics")
    metrics.make_scorer = lambda *a, **k: None
    metrics.mean_absolute_error = lambda *a, **k: None
    metrics.r2_score = lambda *a, **k: None
    metrics.mean_squared_error = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics)

    data_loader = types.ModuleType("scripts.data_loader")
    data_loader.load_data = lambda *a, **k: (None, None)
    monkeypatch.setitem(sys.modules, "scripts.data_loader", data_loader)
    fe_mod = types.ModuleType("scripts.feature_engineering")
    fe_mod.engineer_features = lambda X: (X, types.SimpleNamespace(named_steps={'pca': types.SimpleNamespace(n_components_=1)}))
    monkeypatch.setitem(sys.modules, "scripts.feature_engineering", fe_mod)

    engines_mod = types.ModuleType("engines")
    monkeypatch.setitem(sys.modules, "engines", engines_mod)
    for wrapper, cls_name in [
        ("auto_sklearn_wrapper", "AutoSklearnEngine"),
        ("tpot_wrapper", "TPOTEngine"),
        ("autogluon_wrapper", "AutoGluonEngine"),
    ]:
        mod = types.ModuleType(f"engines.{wrapper}")
        mod.__dict__[cls_name] = type(cls_name, (), {})
        monkeypatch.setitem(sys.modules, f"engines.{wrapper}", mod)
    engines_mod.discover_available = lambda: {}

    spec = importlib.util.spec_from_file_location(
        "orchestrator",
        Path(__file__).resolve().parents[1] / "orchestrator.py",
    )
    orch = importlib.util.module_from_spec(spec)
    sys.modules["orchestrator"] = orch
    spec.loader.exec_module(orch)
    return orch


def test_validate_components_availability_success(monkeypatch):
    orch = load_orchestrator(monkeypatch)
    orch._validate_components_availability()


def test_validate_components_availability_missing_model(monkeypatch):
    orch = load_orchestrator(monkeypatch)
    monkeypatch.setattr(
        orch,
        "MODEL_FAMILIES",
        orch.MODEL_FAMILIES + ["NonexistentModel"],
        raising=False,
    )
    with pytest.raises(FileNotFoundError):
        orch._validate_components_availability()


def test_validate_components_availability_missing_preprocessor(monkeypatch):
    orch = load_orchestrator(monkeypatch)
    monkeypatch.setattr(
        orch,
        "PREP_STEPS",
        orch.PREP_STEPS + ["NonexistentPrep"],
        raising=False,
    )
    with pytest.raises(FileNotFoundError):
        orch._validate_components_availability()
