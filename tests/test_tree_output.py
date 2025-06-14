import importlib.util
from pathlib import Path
import sys
import types

import pytest


def load_orchestrator(monkeypatch, printed):
    sys.modules.pop("orchestrator", None)

    pandas = types.ModuleType("pandas")
    monkeypatch.setitem(sys.modules, "pandas", pandas)

    numpy = types.ModuleType("numpy")
    numpy.random = types.SimpleNamespace(seed=lambda *a, **k: None)
    numpy.sqrt = lambda x: x ** 0.5
    numpy.array = lambda x: x
    monkeypatch.setitem(sys.modules, "numpy", numpy)

    rich_console = types.ModuleType("rich.console")
    class DummyConsole:
        def __init__(self, *a, **k):
            pass
        def log(self, *a, **k):
            pass
        def print(self, obj):
            printed.append(obj)
    rich_console.Console = DummyConsole
    monkeypatch.setitem(sys.modules, "rich.console", rich_console)

    rich_tree = types.ModuleType("rich.tree")
    class DummyTree:
        def __init__(self, *a, **k):
            pass
        def add(self, *a, **k):
            return DummyTree()
    rich_tree.Tree = DummyTree
    monkeypatch.setitem(sys.modules, "rich.tree", rich_tree)

    sklearn = types.ModuleType("sklearn")
    monkeypatch.setitem(sys.modules, "sklearn", sklearn)
    pipe_mod = types.ModuleType("sklearn.pipeline")
    pipe_mod.Pipeline = object
    monkeypatch.setitem(sys.modules, "sklearn.pipeline", pipe_mod)
    class DummyX:
        shape = (1, 1)

    class DummyY(list):
        shape = (1,)

    msel = types.ModuleType("sklearn.model_selection")
    msel.RepeatedKFold = object
    msel.cross_validate = lambda *a, **k: None
    def train_test_split(X, y, *a, **k):
        return DummyX(), DummyX(), DummyY([1]), DummyY([1])
    msel.train_test_split = train_test_split
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", msel)

    metrics = types.ModuleType("sklearn.metrics")
    metrics.make_scorer = lambda *a, **k: None
    metrics.mean_absolute_error = lambda *a, **k: 0
    metrics.mean_squared_error = lambda *a, **k: 0
    metrics.r2_score = lambda *a, **k: 0
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics)

    data_loader = types.ModuleType("scripts.data_loader")
    def load_data(*a, **k):
        return DummyX(), DummyY([1])
    data_loader.load_data = load_data
    monkeypatch.setitem(sys.modules, "scripts.data_loader", data_loader)
    fe_mod = types.ModuleType("scripts.feature_engineering")
    fe_mod.engineer_features = lambda X, y=None: (
        X,
        types.SimpleNamespace(named_steps={"pca": types.SimpleNamespace(n_components_=1)}),
    )
    monkeypatch.setitem(sys.modules, "scripts.feature_engineering", fe_mod)

    engines_mod = types.ModuleType("engines")
    monkeypatch.setitem(sys.modules, "engines", engines_mod)
    for wrapper, cls_name in [
        ("auto_sklearn_wrapper", "AutoSklearnEngine"),
        ("tpot_wrapper", "TPOTEngine"),
        ("autogluon_wrapper", "AutoGluonEngine"),
    ]:
        mod = types.ModuleType(f"engines.{wrapper}")
        mod.__all__ = [cls_name]
        mod.__dict__[cls_name] = type(cls_name, (), {})
        monkeypatch.setitem(sys.modules, f"engines.{wrapper}", mod)
    def discover_available():
        return {
            "autosklearn": sys.modules["engines.auto_sklearn_wrapper"],
            "tpot": sys.modules["engines.tpot_wrapper"],
            "autogluon": sys.modules["engines.autogluon_wrapper"],
        }
    engines_mod.discover_available = discover_available

    pickle_stub = types.ModuleType("pickle")
    pickle_stub.dump = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "pickle", pickle_stub)


    spec = importlib.util.spec_from_file_location(
        "orchestrator",
        Path(__file__).resolve().parents[1] / "orchestrator.py",
    )
    orch = importlib.util.module_from_spec(spec)
    sys.modules["orchestrator"] = orch
    spec.loader.exec_module(orch)
    return orch, DummyTree


def test_tree_flag_outputs_tree(monkeypatch, tmp_path):
    printed = []
    orch, TreeCls = load_orchestrator(monkeypatch, printed)

    monkeypatch.setattr(orch, "_validate_components_availability", lambda: None)
    monkeypatch.setattr(orch, "_extract_pipeline_info", lambda m: {})

    def fake_meta_search(**kwargs):
        run_dir = Path(kwargs.get("run_dir"))
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "dummy.txt").write_text("x")
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

    data = tmp_path / "p.csv"
    target = tmp_path / "t.csv"
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
        "--tree",
    ])

    orch._cli()

    assert any(isinstance(obj, TreeCls) for obj in printed)
