# TODO

## Completed Tasks

- Git LFS setup completed, including tracking of `.pkl`, `.json`, `DataSets/`, and `05_outputs/` directories. Git history has been cleaned to properly track large files.
- `orchestrator.py` `AttributeError` for duration calculation fixed.
- Smoke test for `orchestrator.py` passed successfully. All engines (AutoGluon, Auto-Sklearn, TPOT) executed, data loaded, split, and artifacts saved.

## Remaining Action Items

- Update environment setup to ensure required Python packages (e.g., pandas) are installed before running the orchestrator.
- Modify `setup.sh` to either create `env-as` or skip the activation test if it is not needed.

## Status

The setup script now creates the `env-tpa` environment by default and installs all required packages using prebuilt wheels. An optional `env-as` can be created for Auto-Sklearn (Python ≤3.10). Activation scripts use the standard `venv` mechanism so `python orchestrator.py --help` works after running `./setup.sh` and activating the environment.

