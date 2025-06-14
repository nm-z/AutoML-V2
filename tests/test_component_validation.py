import importlib
import importlib.util
import os
import types
import sys
import pytest


def load_orchestrator(monkeypatch):
    # Stub heavy dependencies so orchestrator can be imported
    stub_names = [
        'pandas',
        'numpy',
        'sklearn',
        'sklearn.pipeline',
        'sklearn.model_selection',
        'sklearn.metrics',
        'rich',
        'rich.console',
        'rich.tree',
        'scripts',
        'scripts.data_loader',
        'scripts.feature_engineering',
        'engines',
        'engines.auto_sklearn_wrapper',
        'engines.tpot_wrapper',
        'engines.autogluon_wrapper',
    ]
    for name in stub_names:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)

    # populate minimal attributes used at import time
    sys.modules['numpy'].random = types.SimpleNamespace(seed=lambda *a, **k: None)
    sys.modules['rich.console'].Console = type('Console', (), {'__init__': lambda self, *a, **k: None, 'log': lambda *a, **k: None})
    sys.modules['rich.tree'].Tree = type('Tree', (), {})
    dl_mod = sys.modules['scripts.data_loader']
    def load_data(*args, **kwargs):
        return None, None
    dl_mod.load_data = load_data
    fe_mod = types.ModuleType('scripts.feature_engineering')
    fe_mod.engineer_features = lambda X: (X, types.SimpleNamespace(named_steps={'pca': types.SimpleNamespace(n_components_=1)}))
    monkeypatch.setitem(sys.modules, 'scripts.feature_engineering', fe_mod)
    sys.modules['engines'].discover_available = lambda: []
    sys.modules['engines.auto_sklearn_wrapper'].AutoSklearnEngine = object
    sys.modules['engines.tpot_wrapper'].TPOTEngine = object
    sys.modules['engines.autogluon_wrapper'].AutoGluonEngine = object
    sys.modules['sklearn.pipeline'].Pipeline = object
    sys.modules['sklearn.model_selection'].RepeatedKFold = object
    sys.modules['sklearn.model_selection'].cross_validate = lambda *a, **k: None
    metrics_mod = sys.modules['sklearn.metrics']
    metrics_mod.make_scorer = lambda *a, **k: None
    metrics_mod.mean_absolute_error = lambda *a, **k: None
    metrics_mod.mean_squared_error = lambda *a, **k: None
    metrics_mod.r2_score = lambda *a, **k: None

    # Load orchestrator module directly from file path to avoid import issues
    spec = importlib.util.spec_from_file_location(
        'orchestrator',
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'orchestrator.py'),
    )
    orchestrator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(orchestrator)
    return orchestrator


def test_validate_components_success(monkeypatch):
    orch = load_orchestrator(monkeypatch)
    orch._validate_components_availability()


def test_validate_components_failure(monkeypatch):
    orch = load_orchestrator(monkeypatch)
    monkeypatch.setattr(
        orch,
        'MODEL_FAMILIES',
        orch.MODEL_FAMILIES + ['FakeModel'],
        raising=False,
    )
    with pytest.raises(FileNotFoundError):
        orch._validate_components_availability()
