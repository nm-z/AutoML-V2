#!/bin/bash
# Activate TPOT + AutoGluon environment  

# Ensure pyenv is initialized
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "Activating TPOT + AutoGluon environment (env-tpa)..."
pyenv activate env-tpa
echo "✓ TPOT + AutoGluon environment activated"
echo "Use 'pyenv deactivate' to exit the environment"
