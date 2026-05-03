"""Optional helpers for ``AnnData`` / Scanpy workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from scevonet.net import Sample

if TYPE_CHECKING:
    from anndata import AnnData


def sample_from_adata(
    adata: AnnData,
    cluster_key: str,
    *,
    layer: str | None = None,
    gene_symbols: str | None = None,
) -> Sample:
    """
    Build a :class:`Sample` from an ``AnnData`` object (e.g. after Scanpy preprocessing).

    Parameters
    ----------
    adata
        Annotated matrix; expression in ``X`` or ``layers[layer]``.
    cluster_key
        ``adata.obs`` column with cell type / cluster labels.
    layer
        If set, use ``adata.layers[layer]`` instead of ``X``.
    gene_symbols
        If set, use ``adata.var[gene_symbols]`` as column names; otherwise ``var_names``.
    """
    try:
        import anndata  # noqa: F401
    except ImportError as e:
        raise ImportError("sample_from_adata requires the 'anndata' package.") from e

    X = adata.layers[layer] if layer is not None else adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    genes = (
        adata.var[gene_symbols].astype(str).values
        if gene_symbols is not None
        else adata.var_names.astype(str).values
    )
    matrix = pd.DataFrame(X, index=adata.obs_names.astype(str), columns=genes)
    cell_types = adata.obs[cluster_key].astype(str).tolist()
    return Sample(matrix, cell_types)
