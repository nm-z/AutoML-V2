from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

def load_data(
    predictors_path: str | Path,
    target_path: str | Path,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load predictor and target data from specified paths.

    Supports CSV and conceptually, Parquet files. For Parquet, it's a placeholder
    and would require `pyarrow` or `fastparquet`.

    Parameters
    ----------
    predictors_path : str | Path
        Path to the predictors data file (CSV or Parquet).
    target_path : str | Path
        Path to the target data file (CSV or Parquet).
    **kwargs
        Additional keyword arguments to pass to the underlying data loading function.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features (X) as a DataFrame and the target (y) as a Series.

    Raises
    ------
    ValueError
        If the file format is unsupported or target file contains multiple columns.
    FileNotFoundError
        If the specified paths do not exist.
    """
    predictors_path = Path(predictors_path)
    target_path = Path(target_path)

    if predictors_path.is_dir():
        predictor_files = list(predictors_path.glob("[Pp]redictors*Full_*.csv"))
        if len(predictor_files) != 1:
            raise FileNotFoundError(f"Expected exactly one predictors file in {predictors_path}, found {len(predictor_files)}. Please specify the exact file path or ensure a unique predictor file.")
        predictors_path = predictor_files[0]

    if target_path.is_dir():
        target_files = list(target_path.glob("targets_Hold 1 Full_*.csv"))
        if len(target_files) != 1:
            raise FileNotFoundError(f"Expected exactly one targets file in {target_path}, found {len(target_files)}. Please specify the exact file path or ensure a unique target file.")
        target_path = target_files[0]

    if not predictors_path.exists():
        raise FileNotFoundError(f"Predictors file not found: {predictors_path}")
    if not target_path.exists():
        raise FileNotFoundError(f"Target file not found: {target_path}")

    if predictors_path.suffix == ".csv":
        X = pd.read_csv(predictors_path, **kwargs)
    # elif predictors_path.suffix == ".parquet":
    #     # Placeholder for Parquet support
    #     # X = pd.read_parquet(predictors_path, **kwargs)
    #     raise ValueError("Parquet support is not yet implemented.")
    else:
        raise ValueError(f"Unsupported predictors file format: {predictors_path.suffix}")

    if target_path.suffix == ".csv":
        y = pd.read_csv(target_path, **kwargs).squeeze()
    # elif target_path.suffix == ".parquet":
    #     # Placeholder for Parquet support
    #     # y = pd.read_parquet(target_path, **kwargs).squeeze()
    #     raise ValueError("Parquet support is not yet implemented.")
    else:
        raise ValueError(f"Unsupported target file format: {target_path.suffix}")

    if y.ndim > 1:
        raise ValueError("Target file must contain a single target column.")

    return X, y


__all__ = ["load_data"] 