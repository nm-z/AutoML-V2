#!/bin/bash
# Activate TPOT + AutoGluon environment
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

echo "Activating TPOT + AutoGluon environment (automl-py311)..."
pyenv activate automl-py311
echo "✓ TPOT + AutoGluon environment activated"
echo "Use 'pyenv deactivate' to exit the environment"
