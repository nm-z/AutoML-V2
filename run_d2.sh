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
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

python orchestrator.py \
  --data DataSets/2/D2-Predictors.csv \
  --target DataSets/2/D2-Targets.csv \
  --all "$@"
