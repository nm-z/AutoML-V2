#!/usr/bin/env bash
# Run orchestrator on Dataset 2 with all engines
set -euo pipefail

# Load pyenv so that `pyenv activate` works non-interactively
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
if command -v pyenv >/dev/null; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
else
    echo "pyenv not found; aborting" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the default environment. Fall back to automl-py310 if automl-py311
# does not exist.
if pyenv versions --bare | grep -q "automl-py311"; then
    pyenv activate automl-py311
elif pyenv versions --bare | grep -q "automl-py310"; then
    pyenv activate automl-py310
else
    echo "Neither automl-py311 nor automl-py310 environment exists." >&2
    exit 1
fi

export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

python orchestrator.py \
  --data DataSets/2/D2-Predictors.csv \
  --target DataSets/2/D2-Targets.csv \
  --all "$@"
