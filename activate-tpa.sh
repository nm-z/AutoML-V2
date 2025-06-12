#!/bin/bash
# Activate TPOT + AutoGluon environment  
echo "Activating TPOT + AutoGluon environment (env-tpa)..."
source env-tpa/bin/activate
echo "✓ TPOT + AutoGluon environment activated"
echo "Use 'deactivate' to exit the environment"

# Start a new shell with the environment activated
exec $SHELL
