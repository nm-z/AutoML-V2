# 🔍 **Codebase Info:** 
This is an **AutoML Meta-Framework** that runs 3 different AutoML engines (Auto-Sklearn, TPOT, AutoGluon) in competition against each other, then picks the best one. Think of it as "AutoML for AutoML" - it automatically finds the best automated machine learning approach for your data.

---

## **🏗️ Architecture Breakdown**

### **1. 🎯 The Orchestrator (`orchestrator.py`)**
- **THE BOSS** - coordinates everything
- Loads your data, splits it for validation
- Runs all 3 AutoML engines with the same data
- Does **5×3 repeated cross-validation** (15 total folds) for fair comparison
- Picks the engine with highest R² score as champion
- Saves everything to `05_outputs/`

### **2. 🤖 The Three AutoML Engines**
Each engine is a different approach to automated machine learning:

- **🔬 Auto-Sklearn** (`engines/auto_sklearn_wrapper.py`)
  - Uses Bayesian optimization + meta-learning
  - Needs Python ≤3.10 (hence the compatibility hell)
  - Good at finding robust pipelines

- **🧬 TPOT** (`engines/tpot_wrapper.py`) 
  - Uses genetic programming (evolutionary algorithms)
  - Literally evolves ML pipelines like biological evolution
  - Works with Python 3.11+

- **⚡ AutoGluon** (`engines/autogluon_wrapper.py`)
  - Uses ensemble methods + neural architecture search
  - Super fast, often wins competitions
  - Works with Python 3.11+

### **3. 🧩 Component Library (`components/`)**
This is where the **building blocks** live that the AutoML engines can choose from:

**📦 Models:** Ridge, Lasso, Random Forest, XGBoost, LightGBM, Neural Networks, etc.
**🔧 Preprocessors:** 
- **Scalers:** StandardScaler, RobustScaler, QuantileTransform
- **Dimensionality:** PCA 
- **Outliers:** IsolationForest, LocalOutlierFactor, KMeansOutlier

### **4. 🐍 Environment Hell Management**
The **entire reason for the complexity** is Python version incompatibility:

- **`automl-py310`** - Auto-Sklearn environment (Python ≤3.10)
- **`automl-py311`** - TPOT + AutoGluon environment (Python 3.11+)
- **`setup.sh`** - Creates both environments with correct dependencies
- **`activate-*.sh`** - Switches between environments

---

## **🔄 The Workflow (What Actually Happens)**

1. **📥 Data Loading:** `scripts/data_loader.py` reads your CSV files
2. **🎲 Engine Discovery:** `engines/__init__.py` finds available AutoML engines
3. **⚔️ The Competition:** Each engine gets the same time budget to find best pipeline
4. **📊 Cross-Validation:** 5×3 repeated CV ensures fair comparison (15 folds each)
5. **🏆 Champion Selection:** Highest R² score wins
6. **💾 Artifact Saving:** Winner's model + metrics saved to `05_outputs/`

---

## **🤔 Why This Complexity Exists**

### **The Python Version Problem:**
- **Auto-Sklearn:** Only works on Python ≤3.10
- **TPOT + AutoGluon:** Work best on Python 3.11+
- **Solution:** Two separate environments managed by `setup.sh`

### **The Meta-AutoML Concept:**
Instead of manually picking which AutoML tool to use, this framework:
1. Runs ALL of them
2. Compares them fairly with cross-validation
3. Automatically picks the winner
4. Gives you the best result without having to be an AutoML expert

---

## **📁 File Structure Logic**

```
🎯 orchestrator.py          # The conductor of the orchestra
🔧 setup.sh                 # Environment setup wizard
📊 scripts/data_loader.py    # Data ingestion
🤖 engines/                 # The 3 AutoML competitors
🧩 components/              # ML building blocks library
📁 DataSets/                # Your input data
💾 05_outputs/              # Results and artifacts
🐍 automl-py310/py311       # pyenv environments
📋 AGENTS.md                # ChatGPT Codex agent docs (kept!)
```

---

## **🎮 How To Use It**

```bash
# 1. Setup environments (creates both pyenv environments)
./setup.sh

# 2. Activate the main environment
pyenv activate automl-py311

# 3. Run the competition
python orchestrator.py --all --time 3600 \
  --data DataSets/3/predictors.csv \
  --target DataSets/3/targets.csv

# 4. Check results in 05_outputs/dataset_name/
pyenv deactivate
```

---

## **🧠 The Genius Behind This**

This framework solves the "**Which AutoML tool should I use?**" problem by:
- Eliminating the guesswork
- Running scientific comparisons
- Handling compatibility issues automatically  
- Giving you provenance (detailed logs of what worked)
- Being reproducible (same random seeds, same results)

**Bottom Line:** You throw in your data, it runs 3 different AI approaches, picks the winner, and hands you the best machine learning model automatically. It's AutoML choosing the best AutoML! 🤯

## Dataset 2 Results

Running `./run_d2.sh` with the default 5‑second budget produced the following hold‑out metrics when the engines fell back to `LinearRegression`:

| Engine | R² | RMSE | MAE |
|-------|------|------|------|
| TPOT | 0.9601 | 0.0005 | 0.0004 |
| AutoGluon | 0.9601 | 0.0005 | 0.0004 |

Auto-Sklearn could not run because the package is not installed. The wrapper reports the missing dependency and exits gracefully.



