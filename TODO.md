# TODO

## Completed Tasks
- Added run_all.sh for 60-second smoke test.

- Git LFS setup completed, including tracking of `.pkl`, `.json`, `DataSets/`, and `05_outputs/` directories. Git history has been cleaned to properly track large files.
- `orchestrator.py` `AttributeError` for duration calculation fixed.
- Smoke test for `orchestrator.py` passed successfully. All engines (AutoGluon, Auto-Sklearn, TPOT) executed, data loaded, split, and artifacts saved.
- Resolved scikit-learn version conflict by specifying `scikit-learn>=1.4.2,<1.6` so Auto-Sklearn and TPOT install together.

## Remaining Action Items

- Update environment setup to ensure required Python packages (e.g., pandas) are installed before running the orchestrator.
- Modify `setup.sh` to either create `env-as` or skip the activation test if it is not needed.
- Update `setup.sh` to create `automl-py310` and `automl-py311` pyenv environments automatically.
- Enhance console logs using `rich.tree` so run progress is shown as a clear tree.
- Add a `--tree` flag to `orchestrator.py` to optionally print artifact directories in tree form.
- Create tests verifying tree-formatted output appears when the flag is used.
- Verify `run_all.sh` smoke test passes after updating dependencies.
- Add a missing `run_all.sh` script to launch the orchestrator with all three engines for a quick smoke test.
- Revise setup or CI to ensure required packages like `rich` install reliably without manual intervention.
- Bundle prebuilt wheels or configure a local PyPI mirror so `make test` can run without internet access.

## Status

The setup script currently creates `env-tpa` and optional `env-as` environments using `venv`. We plan to migrate fully to `pyenv` with `automl-py310` and `automl-py311` for improved version management.

