#!/bin/bash
# Activate Auto-Sklearn environment
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

echo "Activating Auto-Sklearn environment (automl-py310)..."
pyenv activate automl-py310
echo "✓ Auto-Sklearn environment activated"
echo "Use 'pyenv deactivate' to exit the environment"
