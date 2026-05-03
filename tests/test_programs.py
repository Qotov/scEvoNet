"""Directionality helpers using PBMC3k-derived expression (no LightGBM)."""

from __future__ import annotations

import pandas as pd
import pytest
from scevonet.tl import classify_transition_genes, cluster_mean_expression


@pytest.fixture
def pbmc_transition_pair(pbmc_bundle):
    return (
        pbmc_bundle["cluster_a"],
        pbmc_bundle["cluster_b"],
        pbmc_bundle["genes_for_transition"],
    )


def test_cluster_mean_expression(pbmc_df_labels, pbmc_transition_pair):
    df, labels = pbmc_df_labels
    cluster, _, _ = pbmc_transition_pair
    m = cluster_mean_expression(df, labels, cluster)
    assert len(m) == df.shape[1]
    mask = pd.Series(labels, index=df.index) == cluster
    pd.testing.assert_series_equal(m, df.loc[mask].mean(), check_names=False)


def test_classify_transition_genes_pbmc(pbmc_df_labels, pbmc_transition_pair):
    df, labels = pbmc_df_labels
    cluster_a, cluster_b, genes = pbmc_transition_pair
    out = classify_transition_genes(
        df,
        labels,
        genes,
        cluster_a,
        cluster_b,
        log2_fc_strong=0.12,
        rank_high=0.52,
        rank_low=0.38,
    )
    assert not out.empty
    assert "category" in out.columns
    allowed = {
        "shared_high",
        "low_both",
        "gain_in_B",
        "loss_from_A",
        "intermediate",
    }
    assert set(out["category"]).issubset(allowed)
