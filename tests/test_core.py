"""Tests for Sample / EvoManager and utilities."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from scevonet import EvoManager, Sample
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
def test_sample_and_evomanager(pbmc_df_labels, pbmc_second_sample, pbmc_sample_config):
    df, labels = pbmc_df_labels
    df2, labels2 = pbmc_second_sample
    cfg = pbmc_sample_config
    s0 = Sample(df, labels, config=cfg)
    s1 = Sample(df2, labels2, config=cfg)
    assert len(s0.models) == len(set(labels))
    em = EvoManager(s0, s1, network_top_genes=30)
    assert em.predictions.keys() == {"0_0", "0_1", "1_0", "1_1"}
    assert em.cell_types_similarity.shape[0] > 0
    assert not em.network.empty


@needs_lightgbm
def test_prepare_input_df(pbmc_df_labels):
    df, _ = pbmc_df_labels
    feats = [str(df.columns[0]), "NOT_IN_MATRIX", str(df.columns[1])]
    out = EvoManager.prepare_input_df(df, feats)
    assert list(out.columns) == feats
    assert np.isnan(out["NOT_IN_MATRIX"].iloc[0])


def test_draw_net_accepts_false_gene():
    import matplotlib.pyplot as plt
    import networkx as nx

    plt.ioff()
    g = nx.Graph()
    g.add_edge("a", "b")
    draw_net(g, gene=False)
    plt.close("all")


@needs_lightgbm
def test_three_samples_predictions(
    pbmc_df_labels, pbmc_second_sample, pbmc_third_sample, pbmc_sample_config
):
    df, labels = pbmc_df_labels
    df2, labels2 = pbmc_second_sample
    df3, labels3 = pbmc_third_sample
    cfg = pbmc_sample_config
    samples = [
        Sample(df, labels, config=cfg),
        Sample(df2, labels2, config=cfg),
        Sample(df3, labels3, config=cfg),
    ]
    em = EvoManager(*samples, network_top_genes=25)
    assert len(em.predictions) == 9
    assert em.cell_types_similarity.shape[1] == sum(len(s.models) for s in samples)
