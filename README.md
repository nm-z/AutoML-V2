# AutoML Harness

## Quick Setup

**One-command setup** - Run this to get everything working instantly:

```bash
./setup.sh
```

This automatically creates Python environments, installs all dependencies, and sets up the project structure. If you prefer to manage the environment yourself, install the required packages first:

```bash
pip install -r requirements.txt
```
This step ensures modules like `pandas` are available before running `orchestrator.py`.

## Git Repository Structure

This repository has three main branches:
- **`main`** - Primary development branch
- **`master`** - Mirror of main branch  
- **`V1.1`** - Version 1.1 feature branch

## Python Environment Management

This project uses separate Python virtual environments to handle AutoML library compatibility:

- **`env-as`** - Auto-sklearn environment (Python 3.11+ recommended)
- **`env-tpa`** - TPOT + AutoGluon environment (Python 3.11+ recommended)

### Quick Environment Usage

```bash
# Activate Auto-Sklearn environment
./activate-as.sh

# Activate TPOT + AutoGluon environment  
./activate-tpa.sh

# Deactivate any environment
deactivate
```

### Python 3.13 Compatibility Notes

The setup script supports Python 3.13 but with limitations:
- **Auto-sklearn**: Not compatible with Python 3.13 (will be skipped)
- **AutoGluon**: Not compatible with Python 3.13 (will be skipped)  
- **TPOT**: Works with Python 3.13 with compatibility warnings
- **XGBoost/LightGBM**: Generally compatible with Python 3.13

For best compatibility, install Python 3.11:
```bash
# Arch Linux
sudo pacman -S python311

# Ubuntu/Debian  
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Manual Installation (if setup.sh fails)

```bash
# Create environments
python3.11 -m venv env-as
python3.11 -m venv env-tpa

# Install Auto-Sklearn environment
source env-as/bin/activate
pip install --upgrade pip
pip install auto-sklearn==0.15.0 numpy==1.24.3 scikit-learn==1.3.2 pandas matplotlib seaborn rich joblib
deactivate

# Install TPOT + AutoGluon environment
source env-tpa/bin/activate
pip install --upgrade pip
pip install setuptools tpot autogluon.tabular numpy scikit-learn pandas matplotlib seaborn rich joblib xgboost lightgbm
deactivate
```

## Running the Orchestrator

```bash
# Activate the appropriate environment
./activate-tpa.sh

# Run with all engines
python orchestrator.py --all --time 3600 \
  --data DataSets/3/predictors_Hold\ 1\ Full_20250527_151252.csv \
  --target DataSets/3/targets_Hold\ 1\ Full_20250527_151252.csv

# Run with specific engines
python orchestrator.py --tpot --time 1800 \
  --data DataSets/1/Predictors_Hold-1_2025-04-14_18-28.csv

deactivate
```

## Project Structure

```
AutoML-Harness/
├── orchestrator.py              # Main entry point
├── setup.sh                     # One-command setup script
├── activate-as.sh               # Auto-sklearn environment activation
├── activate-tpa.sh              # TPOT + AutoGluon environment activation
├── engines/                     # AutoML engine wrappers
├── components/                  # Preprocessors and models
├── DataSets/                    # Input datasets
├── 05_outputs/                  # Generated artifacts and results
└── requirements.txt             # Base dependencies
```

## Output Artifacts

All runs generate artifacts in `05_outputs/<dataset_name>/`:
- **`*_champion.pkl`** - Trained pipeline for each engine
- **`metrics.json`** - Comprehensive 5×3 CV performance metrics  
- **`*.log`** - Detailed execution logs

## System Requirements

- Linux (recommended) or macOS
- Python 3.11+ (recommended) or Python 3.13 (with limitations)
- 8GB+ RAM for larger datasets
- Build tools (`build-essential` on Ubuntu, `base-devel` on Arch)

## Running in Docker with Persistent Logs

This repository includes a minimal **Dockerfile** and a
`docker-compose.yml` for reproducible runs. The compose configuration
mounts the host `05_outputs/` directory into the container so that
all logs and artifacts remain available after the container exits.

```bash
# Build the Docker image
docker compose build

# Execute the orchestrator (logs are written to ./05_outputs on the host)
docker compose run automl \
  python orchestrator.py --help
```

All logs are stored under `05_outputs/logs/` on the host machine,
ensuring they persist between runs.
=======
## License

This project is licensed under the [MIT License](LICENSE).

