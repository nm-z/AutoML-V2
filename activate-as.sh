#!/bin/bash
# Activate Auto-Sklearn environment
echo "Activating Auto-Sklearn environment (env-as)..."
source env-as/bin/activate
echo "✓ Auto-Sklearn environment activated"
echo "Use 'deactivate' to exit the environment"

# Start a new shell with the environment activated
exec $SHELL
