# AutoML Harness

## Quick Setup

**One-command setup** - Run this to get everything working instantly:

```bash
./setup.sh [--with-as]
```

This automatically creates the `env-tpa` Python environment, installs all dependencies (including `pandas`), and sets up the project structure. Use `--with-as` if you also want the optional Auto-Sklearn environment. After running it, activate the environment before using the orchestrator:

```bash
./activate-tpa.sh
```

If you prefer to manage the environment yourself, install the required packages first:

```bash
pip install -r requirements.txt
```
This step ensures modules like `pandas` are available before running `orchestrator.py`.

> **Note**
> 
> The AutoGluon engine depends on the `autogluon.tabular` package. If this library is missing, `autogluon_wrapper.py` falls back to a simple `LinearRegression`, which severely limits model quality. Run `./setup.sh` or the `pip install` command above to install the full AutoGluon dependencies and avoid the fallback.

## Git Repository Structure

This repository has three main branches:
- **`main`** - Primary development branch
- **`master`** - Mirror of main branch  
- **`V1.1`** - Version 1.1 feature branch

## Python Environment Management

This project uses separate Python virtual environments to handle AutoML library compatibility:

- **`env-as`** - Auto-sklearn environment (optional, Python ≤3.10 recommended)
- **`env-tpa`** - TPOT + AutoGluon environment (Python 3.11+ recommended)

### Quick Environment Usage

```bash
# (Optional) Activate Auto-Sklearn environment
./activate-as.sh

# Activate TPOT + AutoGluon environment (default)
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
python3.11 -m venv env-tpa
# Optional Auto-Sklearn environment
python3.11 -m venv env-as

# Install Auto-Sklearn environment (Python <=3.10 only)
source env-as/bin/activate
pip install --upgrade pip
pip install auto-sklearn==0.15.0 numpy==1.24.3 scikit-learn==1.4.2 pandas matplotlib seaborn rich joblib
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

# Run with all engines (default behavior)
python orchestrator.py --all --time 3600 \
  --data DataSets/3/predictors_Hold\ 1\ Full_20250527_151252.csv \
  --target DataSets/3/targets_Hold\ 1\ Full_20250527_151252.csv

# The orchestrator automatically runs Auto-Sklearn, TPOT and AutoGluon
# together. The `--all` flag is optional but included here for clarity.
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
## Log Aggregation

This project ships a simple **ELK stack** configuration for collecting and searching logs.
Start the stack with:

```bash
docker compose -f docker-compose.logging.yml up -d
```

Set the `LOGSTASH_HOST` environment variable so `orchestrator.py` forwards logs to Logstash:

```bash
export LOGSTASH_HOST=localhost  # or the host running Logstash
export LOGSTASH_PORT=5959       # optional, defaults to 5959
```

Now run the orchestrator as usual and view logs in Kibana at <http://localhost:5601>.

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

## License

This project is licensed under the [MIT License](LICENSE).

