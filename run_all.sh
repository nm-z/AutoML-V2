#!/usr/bin/env bash
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

# Set the PYTHON_PATH to include the current directory so Python can find orchestrator.py
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Loop over all dataset directories and run a short smoke test on each
for ds in "$SCRIPT_DIR"/DataSets/*; do
    predictors=$(find "$ds" -maxdepth 1 -name '*Predictors.csv' | head -n 1)
    targets=$(find "$ds" -maxdepth 1 -name '*Targets.csv' | head -n 1)
    if [[ -f "$predictors" && -f "$targets" ]]; then
        echo "Running smoke test on dataset $(basename "$ds")"
        python orchestrator.py \
            --data "$predictors" \
            --target "$targets" \
            --time 60 \
            --all
    fi
done

pyenv deactivate

