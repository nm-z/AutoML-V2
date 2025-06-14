#!/bin/bash
# Run orchestrator with all engines for a 60-second smoke test
set -euo pipefail

# Load pyenv so that `pyenv activate` works even when the script is executed
# non-interactively.
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
if command -v pyenv >/dev/null; then
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
else
    echo "pyenv not found; aborting" >&2
    exit 1
fi

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate default environment
pyenv activate automl-py311

# Execute orchestrator with sample dataset
python "$SCRIPT_DIR/orchestrator.py" --all --time 60 \
  --data "$SCRIPT_DIR/DataSets/1/D1-Predictors.csv" \
  --target "$SCRIPT_DIR/DataSets/1/D1-Targets.csv" "$@"

pyenv deactivate

