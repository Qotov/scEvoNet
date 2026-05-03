"""Directionality helpers: synthetic unit tests + PBMC integration (no LightGBM for basics)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scevonet.programs import classify_transition_genes, cluster_mean_expression


def test_cluster_mean_expression_synthetic():
    df = pd.DataFrame(np.random.rand(10, 3), columns=["g1", "g2", "g3"])
    labels = ["A"] * 5 + ["B"] * 5
    m = cluster_mean_expression(df, labels, "A")
    assert len(m) == 3
    pd.testing.assert_series_equal(m, df.iloc[:5].mean())


def test_classify_transition_genes_basic():
    rng = np.random.default_rng(0)
    n = 40
    genes = ["early", "shared", "late", "noise"]
    X = np.zeros((n, len(genes)))
    X[:15, 0] = 5.0
    X[:15, 1] = 4.0
    X[:15, 2] = 0.1
    X[:15, 3] = rng.random(15)
    X[15:, 0] = 0.1
    X[15:, 1] = 4.0
    X[15:, 2] = 5.0
    X[15:, 3] = rng.random(25)
    mat = pd.DataFrame(X, columns=genes)
    labels = ["A"] * 15 + ["B"] * 25
    out = classify_transition_genes(
        mat,
        labels,
        genes,
        "A",
        "B",
        log2_fc_strong=0.25,
        rank_high=0.55,
        rank_low=0.35,
    )
    cats = dict(zip(out["gene"], out["category"]))
    assert cats.get("late") == "gain_in_B"
    assert cats.get("early") == "loss_from_A"


def test_cluster_mean_expression_pbmc(pbmc_df_labels, pbmc_transition_pair):
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
