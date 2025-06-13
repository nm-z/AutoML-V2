#!/bin/bash
# Run orchestrator with all engines for a 60-second smoke test
set -euo pipefail

# Determine script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate default environment
source "$SCRIPT_DIR/activate-tpa.sh"

# Execute orchestrator with sample dataset
python "$SCRIPT_DIR/orchestrator.py" --all --time 60 \
  --data "$SCRIPT_DIR/DataSets/1/D1-Predictors.csv" \
  --target "$SCRIPT_DIR/DataSets/1/D1-Targets.csv" "$@"

