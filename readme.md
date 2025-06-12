# AutoML Orchestrator

## Purpose

A reproducible, plug‑and‑play **AutoML** harness that can ingest *any* tabular dataset, explore the *exact* model families and preprocessing blocks you approve of, and hand back the highest‑R² pipeline—all without hand‑rolled Optuna loops.

The system wraps three battle‑tested engines—**auto‑sklearn 2**, **TPOT 2**, and **AutoGluon‑Tabular**—behind a thin orchestrator that:
1. Constrains each engine to the approved search‑space (Tables 1 & 2).
2. Runs the engines in parallel, each with its own budget.
3. Scores their champions on a shared validation split.
4. Returns, or ensembles, the true best‑R² model.

---

## Component Tree

```
AutoML‑Harness
├── orchestrator.py           # meta_search controller
├── engines/
│   ├── auto_sklearn_wrapper.py
│   ├── tpot_wrapper.py
│   └── autogluon_wrapper.py
├── components/
│   ├── preprocessors/
│   │   ├── scalers/
│   │   │   ├── RobustScaler
│   │   │   ├── StandardScaler
│   │   │   └── QuantileTransform
│   │   ├── dimensionality/PCA
│   │   └── outliers/
│   │       ├── KMeansOutlier
│   │       ├── IsolationForest
│   │       └── LocalOutlierFactor
│   └── models/
│       ├── Ridge
│       ├── RPOP
│       ├── Lasso
│       ├── ElasticNet
│       ├── SVR
│       ├── DecisionTree
│       ├── RandomForest
│       ├── ExtraTrees
│       ├── GradientBoosting
│       ├── AdaBoost
│       ├── MLP
│       ├── XGBoost
│       └── LightGBM
|── README.md          # ← this file
|
|-----    config.json # generated locator of dataset
     
```

---

## Library Reference

### Table 1 – Model Families & Default Hyperparameters

| Model Family     | Default Hyperparameters                                                                                  |
| ---------------- | -------------------------------------------------------------------------------------------------------- |
| Ridge            | alpha = 1.0                                                                                              |
| RPOP             | alpha = 0.001                                                                                            |
| Lasso            | alpha = 0.1, max\_iter = 2000                                                                            |
| ElasticNet       | alpha = 0.1, l1\_ratio = 0.5, max\_iter = 2000                                                           |
| SVR              | C = 10.0, gamma='scale', kernel='rbf'                                                                    |
| DecisionTree     | max\_depth = 10, random\_state = 42                                                                      |
| RandomForest     | n\_estimators = 100, max\_depth = 10, random\_state = 42, n\_jobs = 1                                    |
| ExtraTrees       | n\_estimators = 100, max\_depth = 10, random\_state = 42, n\_jobs = 1                                    |
| GradientBoosting | n\_estimators = 100, max\_depth = 3, learning\_rate = 0.1, random\_state = 42                            |
| AdaBoost         | n\_estimators = 100, learning\_rate = 1.0, random\_state = 42                                            |
| MLP              | hidden\_layer\_sizes = (128, 64), alpha = 0.001, max\_iter = 500, random\_state = 42                     |
| XGBoost\*        | n\_estimators = 100, max\_depth = 6, learning\_rate = 0.1, random\_state = 42, n\_jobs = 1               |
| LightGBM\*       | n\_estimators = 100, max\_depth = 6, learning\_rate = 0.1, random\_state = 42, n\_jobs = 1, verbose = ‑1 |

\*Included only when the library is available.

### Table 2 – Preprocessing Methods & Parameter Spaces

| Method             | Parameter Space                                                                           |
| ------------------ | ----------------------------------------------------------------------------------------- |
| PCA                | n\_components ∈ \[10, min(50, n\_features − 5)]                                           |
| RobustScaler       | quantile\_range ∈ {(5, 95), (10, 90), (25, 75)}                                           |
| StandardScaler     | –                                                                                         |
| KMeansOutlier      | n\_clusters ∈ \[2, 6], min\_cluster\_size\_ratio ∈ {0.05, 0.10, 0.15}                     |
| IsolationForest    | contamination ∈ {0.05, 0.10, 0.15, 0.20}, n\_estimators ∈ \[50, 200]                      |
| LocalOutlierFactor | n\_neighbors ∈ \[10, 30], contamination ∈ {0.05, 0.10, 0.15, 0.20}                        |
| FeatureSelection   | score\_func ∈ {mutual\_info\_regression, f\_regression}, k ∈ \[20, min(120, n\_features)] |
| QuantileTransform  | output\_distribution ∈ {uniform, normal}, n\_quantiles ∈ \[100, min(1000, n\_samples//2)] |

---

## Implementation Plan (see `TODO.md` for checklist)

1. Wrap any custom transformers so they present a strict scikit‑learn API.
2. Populate `config.py` with `MODEL_FAMILIES` and `PREP_STEPS` arrays.
3. Implement `_get_automl_engine()` (imports auto‑sklearn → TPOT → AutoGluon).
4. Write `_fit_engine()` adapters and `meta_search()` orchestrator.
5. Optional: ensemble the three champions for a small R² bump.
6. Add CLI + CI tests.

---

## Rules Summary (see `RULES.md` for full text)

* All components **must** expose a scikit‑learn‑compatible `.fit/.transform/.predict`.
* The search space is frozen to Tables 1 & 2.
* Default metric is **R²**.
* Each AutoML engine gets its own wall‑clock budget (default 3600 s).
---

### **Module-by-Module Requirements (full, no omissions)**

> **Legend**
> • **Public API** = everything that external callers may import or execute
> • **Internal helpers** = “private” (leading “\_”) but still required
> • **I/O contracts** = filenames, ENV vars, artifacts that **must** be produced/consumed

---

#### **`orchestrator.py`** &#x20;

| Area                 | Requirement                                                                                                                                                                                                                                                                                                           |                                                   |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| **Purpose**          | Single **entry-point** that coordinates dataset loading, preprocessing search-space injection, parallel engine execution, scoring and optional champion ensembling.                                                                                                                                                   |                                                   |
| **Public API**       | `main() -> None` (standard CLI gateway)  · \`meta\_search(config\_path\:str, X\:pd.DataFrame, y\:pd.Series, budget\:int                                                                                                                                                                                               | dict) -> Pipeline\` (returns best final pipeline) |
| **Internal helpers** | `_get_automl_engine(name:str) -> BaseEngine` (selects wrapper) · `_fit_engine(engine:BaseEngine, X_train, X_valid, y_train, y_valid, timeout:int) -> Tuple[Pipeline,float]` · `_score(champion:Pipeline, X_valid,y_valid) -> float` · `_blend(champions:List[Pipeline]) -> Pipeline` (simple mean-regressor ensemble) |                                                   |
| **Concurrency**      | Spawn **one `multiprocessing.Process` per engine**; join with `timeout` slice from the global budget.                                                                                                                                                                                                                 |                                                   |
| **Config ingestion** | Accept absolute/relative path to **`config.json`** (generated by dataset locator) plus optional `--budget-seconds`, `--metric` CLI flags.                                                                                                                                                                             |                                                   |
| **Logging**          | Use `logging.basicConfig(level=logging.INFO)`; every engine emits `search-start`, `search-end`, `best-score` lines.                                                                                                                                                                                                   |                                                   |
| **Artifacts**        | Persist **full fitted pipeline** to `artifacts/best_pipeline.pkl`; persist **per-engine search logs** into `artifacts/logs/<engine>.log`.                                                                                                                                                                             |                                                   |
| **Exit codes**       | • `0` = success & artifact written • `2` = any engine crashed (fail-fast) • `3` = timeout exhausted before at least one engine finished                                                                                                                                                                               |                                                   |

---

#### **`engines/auto_sklearn_wrapper.py`** &#x20;

| Area                         | Requirement                                                                                                                                   |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Public classes**           | `AutoSklearnEngine(BaseEngine)`                                                                                                               |
| **Public methods**           | `fit(X_train,y_train,X_valid,y_valid,budget:int)->None` · `predict(X) -> np.ndarray` · `export(path:str)->None` (calls `automl.dump_model()`) |
| **Internal helpers**         | `_build_search_space(models, preprocessors) -> dict` · `_translate_metric(metric:str) -> autosklearn.metrics.*`                               |
| **Search-space enforcement** | Must honor **Tables 1 & 2** exactly; pass to `include` / `exclude` params.                                                                    |
| **Timeout handling**         | Pass `time_left_for_this_task=budget` and `per_run_time_limit=budget//16`.                                                                    |
| **Artifacts**                | Write `{run_dir}/autosklearn_score_details.csv` & `{run_dir}/model.pkl`.                                                                      |
| **Dependencies**             | `auto-sklearn>=0.15,<0.19`.                                                                                                                   |

---

#### **`engines/tpot_wrapper.py`** &#x20;

| Area                      | Requirement                                                                                                       |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **Public classes**        | `TPOTEngine(BaseEngine)`                                                                                          |
| **Public methods**        | Same signatures as AutoSklearnEngine.                                                                             |
| **Parameter enforcement** | Pass fixed `config_dict` that contains **only** the model & pre-processing primitives enumerated in Tables 1 & 2. |
| **Early stopping**        | Stop when validation R² plateaus > 20 generations.                                                                |
| **Artifacts**             | `exported_pipeline.py`, `evaluation.json`.                                                                        |

---

#### **`engines/autogluon_wrapper.py`** &#x20;

| Area               | Requirement                                                                                                                           |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Public classes** | `AutoGluonEngine(BaseEngine)`                                                                                                         |
| **Public methods** | Identical fit/predict/export.                                                                                                         |
| **Preset**         | Use `presets="medium_quality_faster_train"`; override model-types list so that **only** Ridge, RF, XGB, LGBM, etc. in Table 1 remain. |
| **Artifacts**      | Full AutoGluon directory zipped to `autogluon.zip`.                                                                                   |

---

#### **`components/preprocessors/scalers/RobustScaler.py`** &#x20;

| Area             | Requirement                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| **Class**        | `RobustScalerBlock(BaseEstimator,TransformerMixin)`                              |
| **Public API**   | `fit(X,y=None)`, `transform(X)`, `get_params(deep=True)`, `set_params(**kwargs)` |
| **Hyper-params** | `quantile_range:Tuple[int,int]` limited to {(5,95),(10,90),(25,75)}              |
| **Tagging**      | `.tags_ = {"handles_nan":True}` for AutoGluon compatibility                      |

*(Repeat identical structure for **StandardScalerBlock** and **QuantileTransformBlock**.)*

---

#### **`components/preprocessors/dimensionality/PCA.py`** &#x20;

| Field     | Detail                                                                |
| --------- | --------------------------------------------------------------------- |
| **Class** | `PCABlock(BaseEstimator,TransformerMixin)`                            |
| **Args**  | `n_components:int` ∈ $10, min(50,n_features−5)$                       |
| **Extra** | Expose `explained_variance_ratio_` property for downstream reporting. |

---

#### **`components/preprocessors/outliers/`** &#x20;

| Module                  | Class                  | Key Params                                                  |
| ----------------------- | ---------------------- | ----------------------------------------------------------- |
| `KMeansOutlier.py`      | `KMeansOutlierBlock`   | `n_clusters 2-6`, `min_cluster_size_ratio {0.05,0.10,0.15}` |
| `IsolationForest.py`    | `IsolationForestBlock` | `contamination {0.05…0.20}`, `n_estimators 50-200`          |
| `LocalOutlierFactor.py` | `LOFBlock`             | `n_neighbors 10-30`, `contamination {0.05…0.20}`            |

All three **must** drop detected outliers **before** piping data forward.

---

#### **`components/models/`** &#x20;

Each model file hosts a **thin adapter class** adding uniform `name`, `param_space`, and `sklearn_model` properties, e.g.:

```python
class RidgeRegressor(ModelBase):
    name = "Ridge"
    param_space = {"alpha": [1.0]}           # frozen default
    def _build(self, **hp): return Ridge(**hp)
```

*Repeat for RPOP, Lasso, ElasticNet, SVR, DecisionTree, RandomForest, ExtraTrees, GradientBoosting, AdaBoost, MLP, XGBoost, LightGBM.*

---

#### **`config.py`** &#x20;

| Requirement                                                                                                          |
| -------------------------------------------------------------------------------------------------------------------- |
| Contains two **immutable lists**: `MODEL_FAMILIES` and `PREP_STEPS`, auto-generated from Tables 1 & 2 at build-time. |
| Exposes helper `get_space(kind:str)->dict` returning the exact hyper-parameter grid for models or preprocessors.     |

---

#### **`config.json` (runtime-generated)** &#x20;

| Field            | Meaning                        |
| ---------------- | ------------------------------ |
| `dataset_path`   | Absolute path to CSV/Parquet   |
| `target`         | Column name of y               |
| `metric`         | `"r2"` by default              |
| `budget_seconds` | Integer wall-clock cap per run |

---

#### **Artifact & Logging Directories**

* `artifacts/best_pipeline.pkl` – Pickled `sklearn.pipeline.Pipeline` ready for `.predict`.
* `artifacts/logs/ENGINE.log` – One file per engine containing JSONL records `{event,time,msg}`.
* `artifacts/autogluon.zip`, `artifacts/exported_pipeline.py`, etc., as noted above.
* All artifact paths must be **deterministic** and relative to CWD for reproducibility.

---

#### **Cross-Module Interface Contracts**

1. **Every component** exposes a **`signature` dict** (`{"type":"model"|"preprocessor","name":...}`) consumed by the orchestrator for audit.
2. **Engines expect** `search_space:dict` built from `config.get_space(...)`; they must **raise `ValueError` immediately** if an unknown primitive sneaks in (fail-fast rule).
3. **All `.fit()` methods** accept `timeout` kwarg and must self-terminate with `TimeoutError` if exceeded.
4. **Verbose tracing**: set `os.environ["AUTOML_VERBOSE"]="1"` to propagate debug prints inside wrappers.

---

Each folder under `components/models/` is a self-contained Python package that “adapts” one estimator into our uniform AutoML interface.  In **every** directory (e.g. `components/models/Ridge/`, `components/models/Lasso/`, etc.) you will include:

1. **`__init__.py`** – the sole entry‐point
2. **Imports**

   ```python
   from sklearn.<module> import <EstimatorClass>
   from ..model_base import ModelBase
   ```
3. **Global signature**

   ```python
   signature = {"type": "model", "name": "<ModelName>"}
   ```
4. **Adapter class**

   ```python
   class <ModelName>Regressor(ModelBase):
       name = "<ModelName>"
       param_space = { …default grid… }
       def _build(self, **hp):
           return <EstimatorClass>(**hp)
   ```
5. **Default hyperparameters** (pulled from **Table 1** of `README.md`):

   * **Ridge**:

     ```python
     param_space = {"alpha": [1.0]}
     ```


   * **RPOP**:

     ```python
     param_space = {"alpha": [0.001]}
     ```


   * **Lasso**:

     ```python
     param_space = {"alpha": [0.1], "max_iter": [2000]}
     ```


   * **ElasticNet**:

     ```python
     param_space = {"alpha": [0.1], "l1_ratio": [0.5], "max_iter": [2000]}
     ```


   * **SVR**:

     ```python
     param_space = {"C": [10.0], "gamma": ["scale"], "kernel": ["rbf"]}
     ```


   * **DecisionTree**:

     ```python
     param_space = {"max_depth": [10], "random_state": [42]}
     ```


   * **RandomForest**:

     ```python
     param_space = {"n_estimators": [100], "max_depth": [10], "random_state": [42], "n_jobs": [1]}
     ```


   * **ExtraTrees**:

     ```python
     param_space = {"n_estimators": [100], "max_depth": [10], "random_state": [42], "n_jobs": [1]}
     ```


   * **GradientBoosting**:

     ```python
     param_space = {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1], "random_state": [42]}
     ```


   * **AdaBoost**:

     ```python
     param_space = {"n_estimators": [100], "learning_rate": [1.0], "random_state": [42]}
     ```


   * **MLP**:

     ```python
     param_space = {"hidden_layer_sizes": [(128, 64)], "alpha": [0.001], "max_iter": [500], "random_state": [42]}
     ```


   * **XGBoost** (*if installed*):

     ```python
     param_space = {"n_estimators": [100], "max_depth": [6], "learning_rate": [0.1], "random_state": [42], "n_jobs": [1]}
     ```


   * **LightGBM** (*if installed*):

     ```python
     param_space = {"n_estimators": [100], "max_depth": [6], "learning_rate": [0.1], "random_state": [42], "n_jobs": [1], "verbose": [-1]}
     ```



That pattern is repeated **verbatim** in each of the 13 model folders.

Below is the full structure and API you should implement in **`engines/auto_sklearn_wrapper.py`**—no omissions:

---

### 1. Imports & Dependencies

```python
import os
import logging
import pickle
from autosklearn.regression import AutoSklearnRegressor
from autosklearn.metrics import r2
from ..config import get_space
from .base_engine import BaseEngine
```

• **Dependency**: `auto-sklearn>=0.15,<0.19` must be in your `requirements.txt`.

---

### 2. Public Class & Constructor

```python
class AutoSklearnEngine(BaseEngine):
    """
    Adapter for auto-sklearn, exposing a uniform .fit/.predict/.export API.
    """
    def __init__(self, models: list, preprocessors: list, metric: str, run_dir: str):
        self.models = models
        self.preprocessors = preprocessors
        self.metric = metric
        self.run_dir = run_dir
        os.makedirs(self.run_dir, exist_ok=True)
        self.automl = None
```

---

### 3. fit(...) → None

```python
    def fit(self,
            X_train, y_train,
            X_valid, y_valid,
            budget: int) -> None:
        """
        Runs AutoSklearnRegressor with the exact frozen search-space,
        respecting the full time budget and per-run limits.
        """
        # 3.1 Build search_space dict from config
        space = get_space("models")      # names from Table 1 :contentReference[oaicite:0]{index=0}
        prep  = get_space("preprocessors")  # from Table 2

        # 3.2 Instantiate AutoSklearnRegressor
        self.automl = AutoSklearnRegressor(
            time_left_for_this_task=budget,
            per_run_time_limit=budget // 16,
            include_estimators=space["models"],
            include_preprocessors=prep["preprocessors"],
            metric=r2  # default metric is R²
        )

        logging.info(f"[auto-sklearn] search-start budget={budget}s")
        self.automl.fit(
            X_train, y_train,
            dataset_name="autosklearn_tabular",
            X_test=X_valid, y_test=y_valid
        )
        logging.info("[auto-sklearn] search-end")

        # 3.3 Persist detailed scoring info
        stats = self.automl.sprint_statistics()
        with open(os.path.join(self.run_dir, "autosklearn_score_details.csv"), "w") as f:
            f.write(stats)

        # 3.4 Save best model artifact
        model_path = os.path.join(self.run_dir, "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(self.automl, f)

        logging.info(f"[auto-sklearn] best-score={self.automl.score(X_valid, y_valid)}")
```

---

### 4. predict(...) → `np.ndarray`

```python
    def predict(self, X):
        """
        Delegate to the fitted AutoSklearnRegressor.
        """
        return self.automl.predict(X)
```

---

### 5. export(path: str) → None

```python
    def export(self, path: str) -> None:
        """
        Dump the trained model for downstream consumption.
        """
        with open(path, "wb") as f:
            pickle.dump(self.automl, f)
```

---

### 6. (Optional) Internal Helpers

```python
    def _translate_metric(self, metric_name: str):
        """
        Map string metric names to autosklearn.metrics modules.
        Currently supports: "r2" → r2
        """
        if metric_name.lower() == "r2":
            return r2
        raise ValueError(f"Unsupported metric: {metric_name}")
```

---

#### Key Points & Enforcement

* **Frozen search-space**: must come exactly from `config.get_space("models")` and `...("preprocessors")`&#x20;
* **Timeout handling**: global `budget` and `per_run_time_limit=budget//16`
* **Artifacts**:

  * `autosklearn_score_details.csv`
  * `model.pkl`
* **Logging**: emit `search-start`, `search-end`, and `best-score` to stdout via `logging.info`

This completes the **full verbose** specification for `auto_sklearn_wrapper.py`.





Each folder under `components/models/` is a self-contained Python package that “adapts” one estimator into our uniform AutoML interface.  In **every** directory (e.g. `components/models/Ridge/`, `components/models/Lasso/`, etc.) you will include:

1. **`__init__.py`** – the sole entry‐point
2. **Imports**

   ```python
   from sklearn.<module> import <EstimatorClass>
   from ..model_base import ModelBase
   ```
3. **Global signature**

   ```python
   signature = {"type": "model", "name": "<ModelName>"}
   ```
4. **Adapter class**

   ```python
   class <ModelName>Regressor(ModelBase):
       name = "<ModelName>"
       param_space = { …default grid… }
       def _build(self, **hp):
           return <EstimatorClass>(**hp)
   ```
5. **Default hyperparameters** (pulled from **Table 1** of `README.md`):

   * **Ridge**:

     ```python
     param_space = {"alpha": [1.0]}
     ```


   * **RPOP**:

     ```python
     param_space = {"alpha": [0.001]}
     ```


   * **Lasso**:

     ```python
     param_space = {"alpha": [0.1], "max_iter": [2000]}
     ```


   * **ElasticNet**:

     ```python
     param_space = {"alpha": [0.1], "l1_ratio": [0.5], "max_iter": [2000]}
     ```


   * **SVR**:

     ```python
     param_space = {"C": [10.0], "gamma": ["scale"], "kernel": ["rbf"]}
     ```


   * **DecisionTree**:

     ```python
     param_space = {"max_depth": [10], "random_state": [42]}
     ```


   * **RandomForest**:

     ```python
     param_space = {"n_estimators": [100], "max_depth": [10], "random_state": [42], "n_jobs": [1]}
     ```


   * **ExtraTrees**:

     ```python
     param_space = {"n_estimators": [100], "max_depth": [10], "random_state": [42], "n_jobs": [1]}
     ```


   * **GradientBoosting**:

     ```python
     param_space = {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1], "random_state": [42]}
     ```


   * **AdaBoost**:

     ```python
     param_space = {"n_estimators": [100], "learning_rate": [1.0], "random_state": [42]}
     ```


   * **MLP**:

     ```python
     param_space = {"hidden_layer_sizes": [(128, 64)], "alpha": [0.001], "max_iter": [500], "random_state": [42]}
     ```


   * **XGBoost** (*if installed*):

     ```python
     param_space = {"n_estimators": [100], "max_depth": [6], "learning_rate": [0.1], "random_state": [42], "n_jobs": [1]}
     ```


   * **LightGBM** (*if installed*):

     ```python
     param_space = {"n_estimators": [100], "max_depth": [6], "learning_rate": [0.1], "random_state": [42], "n_jobs": [1], "verbose": [-1]}
     ```



That pattern is repeated **verbatim** in each of the 13 model folders.









