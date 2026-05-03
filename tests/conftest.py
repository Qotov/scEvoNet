"""Pytest configuration: headless matplotlib + PBMC session fixtures (requires scanpy)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from typing import Any

import pytest
from scevonet.tl import SampleConfig


@pytest.fixture
def pbmc_sample_config() -> SampleConfig:
    """Light training settings for PBMC-shaped integration tests (faster than defaults)."""
    return SampleConfig(
        top_features_limit=200,
        n_estimators=55,
        early_stopping_rounds=5,
        random_state=42,
    )


@pytest.fixture(scope="session")
def pbmc_bundle() -> dict[str, Any]:
    pytest.importorskip("scanpy")
    from scevonet.pbmc_demo import build_pbmc_demo_bundle

    return build_pbmc_demo_bundle(seed=42)


@pytest.fixture
def pbmc_df_labels(pbmc_bundle: dict[str, Any]):
    """Primary sample: cells × genes and cell-type labels."""
    return pbmc_bundle["df_a"], pbmc_bundle["labels_a"]


@pytest.fixture
def pbmc_second_sample(pbmc_bundle: dict[str, Any]):
    """Second PBMC-derived matrix (bootstrap + noise), same genes."""
    return pbmc_bundle["df_b"], pbmc_bundle["labels_b"]


@pytest.fixture
def pbmc_third_sample(pbmc_bundle: dict[str, Any]):
    """Third PBMC-derived matrix for multi-sample EvoManager tests."""
    return pbmc_bundle["df_c"], pbmc_bundle["labels_c"]


@pytest.fixture
def pbmc_batches(pbmc_bundle: dict[str, Any]) -> list[str]:
    """Batch labels aligned with ``pbmc_df_labels`` rows."""
    return pbmc_bundle["batches"]


@pytest.fixture
def pbmc_transition_pair(pbmc_bundle: dict[str, Any]):
    return (
        pbmc_bundle["cluster_a"],
        pbmc_bundle["cluster_b"],
        pbmc_bundle["genes_for_transition"],
    )
