#!/usr/bin/env bash
# Run orchestrator with all engines for a 60-second smoke test
set -euo pipefail

# Load pyenv so that `pyenv activate` works even when the script is executed
# non-interactively. Allow skipping via SKIP_PYENV=1 for CI environments.
export PYENV_ROOT="${PYENV_ROOT:-$HOME/.pyenv}"
export PATH="$PYENV_ROOT/bin:$PYENV_ROOT/shims:$PATH"
if [[ "${SKIP_PYENV:-0}" != "1" ]]; then
    if command -v pyenv >/dev/null; then
        eval "$(pyenv init -)"
        eval "$(pyenv virtualenv-init -)"
    else
        echo "pyenv not found; aborting" >&2
        exit 1
    fi
else
    echo "Skipping pyenv initialization (SKIP_PYENV=1)" >&2
fi

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the default environment. Fall back to automl-py310 if automl-py311
# does not exist.
if [[ "${SKIP_PYENV:-0}" != "1" ]]; then
    if pyenv versions --bare | grep -q "automl-py311"; then
        pyenv activate automl-py311
    elif pyenv versions --bare | grep -q "automl-py310"; then
        pyenv activate automl-py310
    else
        echo "Neither automl-py311 nor automl-py310 environment exists." >&2
        exit 1
    fi
fi

# Set the PYTHON_PATH to include the current directory so Python can find orchestrator.py
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Loop over all dataset directories and run a short smoke test on each
ORCHESTRATOR="${ORCHESTRATOR:-orchestrator.py}"
exit_code=0
for ds in "$SCRIPT_DIR"/DataSets/*; do
    predictors=$(find "$ds" -maxdepth 1 -name '*Predictors.csv' | head -n 1)
    targets=$(find "$ds" -maxdepth 1 -name '*Targets.csv' | head -n 1)
    if [[ -f "$predictors" && -f "$targets" ]]; then
        echo "Running smoke test on dataset $(basename "$ds")"
        if ! python "$ORCHESTRATOR" \
            --data "$predictors" \
            --target "$targets" \
            --time 60 \
            --all; then
            echo "Smoke test failed on $(basename "$ds")" >&2
            exit_code=1
        fi
    else
        echo "Skipping $(basename "$ds") - predictors or targets missing" >&2
    fi
done

if [[ "${SKIP_PYENV:-0}" != "1" ]]; then
    pyenv deactivate
fi

exit $exit_code

