"""Smoke tests for Sample / EvoManager with tiny synthetic data."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from scevonet import EvoManager, Sample, SampleConfig
from scevonet.net import draw_net
from scevonet.utils import sigmoid, update_df


def _lightgbm_loads() -> bool:
    try:
        importlib.import_module("lightgbm")
        return True
    except Exception:
        return False


needs_lightgbm = pytest.mark.skipif(
    not _lightgbm_loads(),
    reason="LightGBM could not be imported (e.g. missing OpenMP / libomp).",
)


@pytest.fixture
def tiny_matrix_and_labels():
    rng = np.random.default_rng(42)
    n, g = 40, 30
    X = rng.random((n, g)) * 2.0
    genes = [f"G{i}" for i in range(g)]
    df = pd.DataFrame(X, columns=genes)
    labels = ["A"] * 15 + ["B"] * 15 + ["C"] * 10
    return df, labels


def test_sigmoid_finite():
    x = pd.DataFrame(np.array([[1000.0, -1000.0]]))
    y = sigmoid(x)
    assert np.all(np.isfinite(y.to_numpy()))


def test_update_df_makes_duplicate_names_unique():
    """If the matrix has repeated column labels, output columns become ``name``, ``name-1``, …"""
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["g0", "g0"])
    out = update_df(df, ["g0"])
    assert list(out.columns) == ["g0", "g0-1"]


@needs_lightgbm
def test_sample_and_evomanager(tiny_matrix_and_labels):
    df, labels = tiny_matrix_and_labels
    cfg = SampleConfig(
        top_features_limit=20,
        n_estimators=30,
        early_stopping_rounds=5,
    )
    s0 = Sample(df, labels, config=cfg)
    s1 = Sample(df, labels, config=cfg)
    assert len(s0.models) == 3
    em = EvoManager(s0, s1, network_top_genes=10)
    assert em.predictions.keys() == {"0_0", "0_1", "1_0", "1_1"}
    assert em.cell_types_similarity.shape[0] > 0
    assert not em.network.empty


def test_prepare_input_df_aligns_genes(tiny_matrix_and_labels):
    df, _ = tiny_matrix_and_labels
    feats = ["G0", "Gmissing", "G1"]
    out = EvoManager.prepare_input_df(df, feats)
    assert list(out.columns) == feats
    assert np.isnan(out["Gmissing"].iloc[0])


def test_draw_net_accepts_false_gene():
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.ioff()
    g = nx.Graph()
    g.add_edge("a", "b")
    draw_net(g, gene=False)
    plt.close("all")


@needs_lightgbm
def test_three_samples_predictions():
    rng = np.random.default_rng(0)
    g = 25
    genes = [f"G{i}" for i in range(g)]
    df = pd.DataFrame(rng.random((24, g)), columns=genes)
    labs = ["X"] * 8 + ["Y"] * 8 + ["Z"] * 8
    cfg = SampleConfig(top_features_limit=15, n_estimators=20, early_stopping_rounds=3)
    samples = [Sample(df, labs, config=cfg) for _ in range(3)]
    em = EvoManager(*samples, network_top_genes=5)
    assert len(em.predictions) == 9
    # Wide similarity table: one column block per training sample
    assert em.cell_types_similarity.shape[1] == sum(len(s.models) for s in samples)
