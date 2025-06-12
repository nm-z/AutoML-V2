#!/bin/bash
# Activate Auto-Sklearn environment

# Ensure pyenv is initialized
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

echo "Activating Auto-Sklearn environment (env-as)..."
pyenv activate env-as
echo "✓ Auto-Sklearn environment activated"
echo "Use 'pyenv deactivate' to exit the environment"
