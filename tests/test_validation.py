"""Validation helpers (LightGBM) using PBMC3k-derived matrices."""

from __future__ import annotations

import importlib

import pytest
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
def test_permutation_importance_null_runs(pbmc_df_labels, pbmc_sample_config):
    mat, labels = pbmc_df_labels
    pos_type = sorted(set(labels))[0]
    out = permutation_importance_null(
        mat,
        labels,
        pos_type,
        config=pbmc_sample_config,
        n_perm=5,
        random_seed=1,
    )
    assert "observed" in out and "p_value" in out
    assert len(out["null_values"]) >= 1


@needs_lightgbm
def test_leave_batch_out_auc(pbmc_df_labels, pbmc_batches, pbmc_sample_config):
    mat, labels = pbmc_df_labels
    pos_type = sorted(set(labels))[0]
    df = leave_batch_out_auc(mat, labels, pbmc_batches, pos_type, config=pbmc_sample_config)
    assert len(df) == 3
    assert "auc" in df.columns
