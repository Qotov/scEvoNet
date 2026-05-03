"""Validation helpers require LightGBM."""

import importlib

import numpy as np
import pandas as pd
import pytest
from scevonet.net import SampleConfig
from scevonet.validation import leave_batch_out_auc, permutation_importance_null


def _lgbm_loads() -> bool:
    try:
        importlib.import_module("lightgbm")
        return True
    except Exception:
        return False


needs_lightgbm = pytest.mark.skipif(
    not _lgbm_loads(),
    reason="LightGBM could not be imported.",
)


@needs_lightgbm
def test_permutation_importance_null_runs():
    rng = np.random.default_rng(42)
    g = 25
    genes = [f"G{i}" for i in range(g)]
    X = rng.random((60, g))
    mat = pd.DataFrame(X, columns=genes)
    labels = ["T"] * 20 + ["U"] * 20 + ["V"] * 20
    cfg = SampleConfig(top_features_limit=15, n_estimators=30, early_stopping_rounds=5)
    out = permutation_importance_null(
        mat,
        labels,
        "T",
        config=cfg,
        n_perm=5,
        random_seed=1,
    )
    assert "observed" in out and "p_value" in out
    assert len(out["null_values"]) >= 1


@needs_lightgbm
def test_leave_batch_out_auc():
    rng = np.random.default_rng(0)
    g = 20
    genes = [f"G{i}" for i in range(g)]
    X = rng.random((48, g))
    mat = pd.DataFrame(X, columns=genes)
    labels = ["P"] * 24 + ["Q"] * 24
    batches = ["b1"] * 16 + ["b2"] * 16 + ["b3"] * 16
    cfg = SampleConfig(top_features_limit=12, n_estimators=25, early_stopping_rounds=3)
    df = leave_batch_out_auc(mat, labels, batches, "P", config=cfg)
    assert len(df) == 3
    assert "auc" in df.columns
