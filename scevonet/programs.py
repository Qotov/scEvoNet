"""Directional interpretation of gene lists and optional gene-set enrichment."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def cluster_mean_expression(
    matrix: pd.DataFrame,
    labels: Sequence[str] | pd.Series,
    cluster: str,
    genes: list[str] | None = None,
) -> pd.Series:
    """Mean expression per gene among cells with label ``cluster`` (pseudobulk profile)."""
    lab = pd.Series(labels, index=matrix.index, dtype=str)
    mask = lab == str(cluster)
    if not mask.any():
        raise ValueError(f"No cells with label {cluster!r}")
    sub = matrix.loc[mask]
    if genes is not None:
        use = [g for g in genes if g in sub.columns]
        if len(use) < len(genes):
            missing = set(genes) - set(sub.columns)
            logger.warning("Genes not in matrix (ignored): %s", sorted(missing)[:10])
        sub = sub[use]
    return sub.mean(axis=0)


def classify_transition_genes(
    matrix: pd.DataFrame,
    labels: Sequence[str] | pd.Series,
    genes: list[str],
    cluster_a: str,
    cluster_b: str,
    *,
    eps: float = 1e-6,
    log2_fc_strong: float = 0.5,
    rank_high: float = 0.65,
    rank_low: float = 0.35,
) -> pd.DataFrame:
    """
    Classify candidate genes by **pseudobulk** expression in two cell states.

    This separates **shared** high expression from **gain/loss** along A→B, which
    tree importance alone does not provide.

    Categories (within the supplied ``genes`` list):

    - **shared_high** — high mean in both A and B (rank among genes ≥ ``rank_high`` in each).
    - **gain_in_B** — log2(B/A) > ``log2_fc_strong`` and not shared_low.
    - **loss_from_A** — log2(B/A) < -``log2_fc_strong`` and not shared_low.
    - **low_both** — low rank in both clusters (≤ ``rank_low``).
    - **intermediate** — remaining genes.

    Parameters
    ----------
    matrix
        Cells × genes (same units as used for modeling, e.g. log-normalized counts).
    labels
        Cell type / cluster per row (aligned with ``matrix``).
    genes
        Gene symbols to score (e.g. genes on a shortest-path subnetwork).
    cluster_a, cluster_b
        Names of the two states to compare (e.g. source and target cell types).
    log2_fc_strong
        |log2 fold-change| above this counts as directional gain/loss.
    rank_high, rank_low
        Percentile ranks **within the gene list** for calling high/low in each state.
    """
    genes_use = [g for g in genes if g in matrix.columns]
    if not genes_use:
        raise ValueError("None of the requested genes are present in the matrix columns.")
    mu_a = cluster_mean_expression(matrix, labels, cluster_a, genes_use)
    mu_b = cluster_mean_expression(matrix, labels, cluster_b, genes_use)

    ra = mu_a.rank(pct=True)
    rb = mu_b.rank(pct=True)
    lfc = np.log2((mu_b + eps) / (mu_a + eps))

    rows = []
    for g in genes_use:
        pa, pb = float(ra[g]), float(rb[g])
        fc = float(lfc[g])
        if pa >= rank_high and pb >= rank_high:
            cat = "shared_high"
        elif pa <= rank_low and pb <= rank_low:
            cat = "low_both"
        elif fc > log2_fc_strong:
            cat = "gain_in_B"
        elif fc < -log2_fc_strong:
            cat = "loss_from_A"
        else:
            cat = "intermediate"
        rows.append(
            {
                "gene": g,
                "mean_in_A": mu_a[g],
                "mean_in_B": mu_b[g],
                "log2_fold_change_B_vs_A": fc,
                "rank_pct_in_A": pa,
                "rank_pct_in_B": pb,
                "category": cat,
            }
        )
    return pd.DataFrame(rows).sort_values("log2_fold_change_B_vs_A", ascending=False)


def enrich_genes(
    genes: list[str],
    *,
    organism: str = "human",
    gene_sets: str | list[str] | None = None,
    outdir: str | None = None,
    no_plot: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Run **over-representation analysis** via Enrichr (through ``gseapy``).

    Requires optional dependency: ``uv pip install 'scevonet[enrichment]'`` (or ``uv add 'scevonet[enrichment]'`` in a uv project).

    Parameters
    ----------
    genes
        Gene symbols (HGNC / MGI symbols depending on ``organism``).
    organism
        ``human``, ``mouse``, or other Enrichr-supported labels.
    gene_sets
        Enrichr library names. Defaults to GO Biological Process + MSigDB Hallmark.
    outdir
        If set, write Enrichr tables under this directory (``None`` = no files).
    **kwargs
        Passed to ``gseapy.enrichr``.

    Returns
    -------
    Combined enrichment table (``gseapy`` results dataframe).
    """
    try:
        import gseapy as gp
    except ImportError as e:
        raise ImportError(
            "enrich_genes requires gseapy. Install with: uv pip install 'scevonet[enrichment]'"
        ) from e

    if gene_sets is None:
        gene_sets = ["GO_Biological_Process_2021", "MSigDB_Hallmark_2020"]

    enr = gp.enrichr(
        gene_list=list(genes),
        gene_sets=gene_sets,
        organism=organism,
        outdir=outdir,
        no_plot=no_plot,
        **kwargs,
    )
    return enr.results


def enrich_by_cell_type_programs(
    models_top_features: dict[str, pd.DataFrame],
    *,
    top_n: int = 100,
    organism: str = "human",
    gene_sets: str | list[str] | None = None,
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """
    Run enrichment on the top ``top_n`` important genes per cell type (program-level).

    Parameters
    ----------
    models_top_features
        Mapping cell type → feature table with columns ``gene``, ``importance`` (as on :class:`Sample`).
    """
    out: dict[str, pd.DataFrame] = {}
    for ct, feat in models_top_features.items():
        g = feat["gene"].head(top_n).astype(str).tolist()
        if not g:
            continue
        out[ct] = enrich_genes(g, organism=organism, gene_sets=gene_sets, **kwargs)
    return out
