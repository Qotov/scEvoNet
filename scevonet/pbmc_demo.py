"""PBMC3k-based demonstration matrices (subset, noise, synthetic labels).

Used by tutorials and tests. Requires **scanpy**::

    pip install scanpy

First call downloads PBMC3k (~6 MB) via Scanpy’s cache.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _stratified_batches(labels: list[str], rng: np.random.Generator) -> list[str]:
    """Spread each cell type across ``batch_0`` … ``batch_2`` for leave-batch-out checks."""
    n = len(labels)
    out = ["batch_0"] * n
    df_idx = pd.DataFrame({"ix": np.arange(n), "lab": labels})
    for _, sub in df_idx.groupby("lab"):
        ix = sub["ix"].tolist()
        rng.shuffle(ix)
        for k, i in enumerate(ix):
            out[i] = f"batch_{k % 3}"
    return out


def build_pbmc_demo_bundle(seed: int = 42) -> dict[str, Any]:
    """
    Load Scanpy PBMC3k, subsample cells/genes, add noise, derive labels.

    Returns three expression matrices (bootstrap resamples + noise) sharing the same
    genes (simulating multiple samples), stratified batch IDs, and metadata for
    directionality / enrichment examples.
    """
    import scanpy as sc
    from sklearn.cluster import KMeans

    rng = np.random.default_rng(seed)
    adata = sc.datasets.pbmc3k()

    n_cells_target = min(450, adata.n_obs)
    obs_ix = rng.choice(adata.n_obs, size=n_cells_target, replace=False)
    adata = adata[obs_ix].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    n_genes = min(800, adata.n_vars)
    if adata.n_vars > n_genes:
        xd = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        mean_expr = xd.mean(axis=0).ravel()
        g_ix = np.argsort(mean_expr)[-n_genes:]
        adata = adata[:, g_ix].copy()

    x = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)

    genes = [str(g) for g in adata.var_names]
    cells = [str(c) for c in adata.obs_names]
    df = pd.DataFrame(x, index=cells, columns=genes)

    noise_scale = 0.06
    df = df + rng.normal(0.0, noise_scale, size=df.shape)
    df = df.clip(lower=0.0)

    n_types = 4
    km = KMeans(n_clusters=n_types, random_state=seed, n_init=10)
    lab_ix = km.fit_predict(df.to_numpy())
    base_names = [f"CT{i}" for i in range(n_types)]
    labels: list[str] = [base_names[i] for i in lab_ix]

    labels_arr = np.array(labels)
    n = len(labels_arr)
    n_rare = max(12, int(0.06 * n))
    rare_ix = rng.choice(n, size=n_rare, replace=False)
    labels_arr[rare_ix] = "Rare"
    labels = labels_arr.tolist()

    vc = pd.Series(labels).value_counts()
    if vc.min() < 10:
        merge_from = vc.idxmin()
        donor = vc.idxmax()
        need = int(10 - vc[merge_from])
        donor_mask = np.array(labels) == donor
        donor_ix = np.flatnonzero(donor_mask)
        relab = rng.choice(donor_ix, size=min(need, len(donor_ix)), replace=False)
        for i in relab:
            labels[i] = merge_from

    n = len(labels)
    boot_ix = rng.choice(n, size=n, replace=True)
    df_b = df.iloc[boot_ix].copy()
    df_b.index = [f"s2_{i}" for i in range(n)]
    df_b = df_b + rng.normal(0.0, 0.04, size=df_b.shape)
    df_b = df_b.clip(lower=0.0)
    labels_b = [labels[j] for j in boot_ix]

    boot2_ix = rng.choice(n, size=n, replace=True)
    df_c = df.iloc[boot2_ix].copy()
    df_c.index = [f"s3_{i}" for i in range(n)]
    df_c = df_c + rng.normal(0.0, 0.035, size=df_c.shape)
    df_c = df_c.clip(lower=0.0)
    labels_c = [labels[j] for j in boot2_ix]

    batches = _stratified_batches(labels, rng)

    types_present = [t for t in ["CT0", "CT1", "CT2", "CT3", "Rare"] if t in set(labels)]
    cluster_a, cluster_b = types_present[0], types_present[1]

    genes_transition = genes[: min(40, len(genes))]

    return {
        "df_a": df,
        "labels_a": labels,
        "df_b": df_b,
        "labels_b": labels_b,
        "df_c": df_c,
        "labels_c": labels_c,
        "batches": batches,
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "genes_for_transition": genes_transition,
    }
