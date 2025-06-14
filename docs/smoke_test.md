# Smoke Test Guide

The `run_all.sh` script provides a quick 60-second sanity check across every dataset.
It now supports optional environment variables for CI:

- `SKIP_PYENV=1` skips pyenv activation when the environments are preconfigured.
- `ORCHESTRATOR=/path/to/script` allows using a stub orchestrator for testing.

The script iterates over `DataSets/*` and reports failures per dataset.
Any missing predictors or targets result in a warning rather than a crash.
