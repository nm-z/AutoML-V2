# AutoML

## Quick Setup

**One-command setup** - Run this to get everything working instantly:

```bash
./setup.sh [--with-as]
```

This automatically creates the `automl-py311` and optional `automl-py310`
environments using **pyenv**. After running it, activate the environment before
using the orchestrator:

```bash
pyenv activate automl-py311
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

This project now uses **pyenv** to maintain two dedicated environments:

- **`automl-py310`** – Auto-Sklearn (Python 3.10)
- **`automl-py311`** – TPOT + AutoGluon (Python 3.11)

### Quick Environment Usage

```bash
# Activate an environment
pyenv activate automl-py310   # or automl-py311

# Deactivate when done
pyenv deactivate
```

## Development Environment Tips

1. **Environment Activation and Deactivation**

   ```bash
   pyenv activate automl-py310  # or automl-py311
   pyenv deactivate
   ```

   Use `pyenv local` in the project root so the environment activates automatically when entering the directory.

2. **Dependency Management**

   Maintain separate `requirements-py310.txt` and `requirements-py311.txt` files.
   After installing or updating packages in an environment run:

   ```bash
   pyenv activate automl-py310
   pip freeze > requirements-py310.txt
   pyenv deactivate
   ```

3. **Running Scripts with Specific Versions**

   You can call scripts without activating an environment by using `pyenv exec`:

   ```bash
   pyenv exec python3.11 orchestrator.py --all
   ```

4. **Troubleshooting**

   - Ensure the correct environment is active if you see `ModuleNotFoundError`.
   - Re-run `./setup.sh` if dependencies are missing.
   - When using `gh` commands, set `env GH_PAGER=cat` to avoid pager errors.

5. **Maintaining `setup.sh`**

   Keep the setup script in sync with new dependencies so others can reproduce the same environments.


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
# Create environments with pyenv
pyenv virtualenv 3.11 automl-py311
# Optional Auto-Sklearn environment
pyenv virtualenv 3.10 automl-py310

# Install Auto-Sklearn environment (Python <=3.10 only)
pyenv activate automl-py310
pip install --upgrade pip
pip install auto-sklearn==0.15.0 numpy==1.24.3 scikit-learn\>=1.4.2,<1.6 pandas matplotlib seaborn rich joblib
pyenv deactivate

# Install TPOT + AutoGluon environment
pyenv activate automl-py311
pip install --upgrade pip
pip install setuptools tpot autogluon.tabular numpy scikit-learn\>=1.4.2,<1.6 pandas matplotlib seaborn rich joblib xgboost lightgbm
pyenv deactivate
```

## Running the Orchestrator

```bash
# Activate the appropriate environment
pyenv activate automl-py311

# Run the orchestrator (AutoGluon, Auto-Sklearn, and TPOT all run)
python orchestrator.py --all --time 3600 \
  --data DataSets/3/predictors_Hold\ 1\ Full_20250527_151252.csv \
  --target DataSets/3/targets_Hold\ 1\ Full_20250527_151252.csv \
  --cpus 4

# Use `--cpus` to limit how many threads each AutoML engine and the underlying
# BLAS libraries may use. This is especially important when running inside a
# Docker container with restricted CPU quotas.

# The orchestrator automatically runs Auto-Sklearn, TPOT and AutoGluon
# together. The `--all` flag is optional but included here for clarity.
pyenv deactivate
```


### Quick Smoke Test
Run the helper script to verify your setup. It activates the default environment and runs all three engines for 60 seconds on the sample dataset:

```bash
./run_all.sh
```
All orchestrations run **AutoGluon**, **Auto-Sklearn**, and **TPOT** simultaneously. The `--all` flag ensures every run evaluates each engine before selecting a champion.

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

## Troubleshooting

- **Environment not activated** – If you encounter `ModuleNotFoundError` or similar issues,
  activate the default environment:

  ```bash
  pyenv activate automl-py311
  ```

  Optionally switch to the Auto-Sklearn environment with `pyenv activate automl-py310`.
  Deactivate the current environment at any time using `pyenv deactivate`.

- **Setup problems** – If `./setup.sh` fails, follow the instructions in the
  *Manual Installation* section to create `env-as` and `env-tpa` manually and
  install the required packages.

- **Python version incompatibilities** – AutoGluon and Auto-Sklearn are skipped
  on Python 3.13. Use Python 3.11 for full functionality.

## License

This project is licensed under the [MIT License](LICENSE).

