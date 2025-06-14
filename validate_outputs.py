import json, os
from pathlib import Path

def validate_metrics(output_dir):
    metrics_file = Path(output_dir) / "metrics.json"
    if not metrics_file.exists():
        print("\u2717 metrics.json not found")
        return
    with open(metrics_file) as f:
        metrics = json.load(f)
    required = ['champion_engine', 'r2_score', 'rmse', 'mae']
    missing = [k for k in required if k not in metrics]
    if not missing:
        print("\u2713 All required metrics present")
    else:
        print(f"\u2717 Missing metrics: {missing}")

# Example usage: validate_metrics('path/to/your/output/timestamp_dir')
