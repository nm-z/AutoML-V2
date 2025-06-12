"""Demo script – trains tiny synthetic dataset to illustrate API usage."""
from __future__ import annotations

import signal
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.tree import Tree

from orchestrator import meta_search

console = Console(highlight=False)


class TimeoutException(Exception):
    """Raised when the 5-minute wall-clock budget elapses."""


def _timeout_handler(signum: int, frame: Any) -> None:  # noqa: D401 – simple name
    raise TimeoutException("Wall-clock budget exceeded (5 minutes).")


signal.signal(signal.SIGALRM, _timeout_handler)
signal.alarm(300)

try:
    tree = Tree("[bold cyan]Demo – Synthetic Dataset[/bold cyan]")
    data_path = Path("datasets/dataset3.csv")
    if not data_path.exists():
        raise FileNotFoundError(data_path)
    df = pd.read_csv(data_path)
    tree.add(f"loaded {len(df)} rows from {data_path}")

    X = df.drop(columns=["target"])
    y = df["target"]

    champion, _ = meta_search(X, y, artifacts_dir="05_outputs/demo", timeout_per_engine=60)
    tree.add(f"🏁 R² on training = {getattr(champion, 'best_score_', 'N/A'):.4f}")

    console.print(tree)
except TimeoutException as exc:
    console.print(f"[red]{exc}")
    sys.exit(1)
finally:
    signal.alarm(0)  # Reset timer 