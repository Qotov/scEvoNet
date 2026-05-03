"""Shared transforms for expression matrices."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
    """Sigmoid squashing; clips input for numerical stability."""
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy(dtype=np.float64, copy=True)
        arr = np.clip(arr, -500.0, 500.0)
        out = 1.0 / (1.0 + np.exp(-arr))
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-x))


def update_df(df: pd.DataFrame, top_features: list) -> pd.DataFrame:
    """Slice to ``top_features`` (unique names, first occurrence) and uniquify duplicate column labels."""
    seen: set[str] = set()
    uniq_features: list[str] = []
    for name in top_features:
        if name not in seen:
            seen.add(name)
            uniq_features.append(name)
    df = df[uniq_features].copy()
    counts: dict[str, int] = {}
    new_cols: list[str] = []
    for col in df.columns:
        base = col
        if base not in counts:
            counts[base] = 1
            new_cols.append(base)
        else:
            new_cols.append(f"{base}-{counts[base]}")
            counts[base] += 1
    df.columns = new_cols
    return df
