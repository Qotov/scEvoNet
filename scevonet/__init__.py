"""scEvoNet: gradient boosting-based cell state evolution and gene–cell networks."""

from scevonet.pl import draw_net, draw_network, finish_matplotlib_figure
from scevonet.tl import (
    EvoManager,
    Sample,
    SampleConfig,
    bootstrap_importance_stability,
    classify_transition_genes,
    cluster_mean_expression,
    enrich_by_cell_type_programs,
    enrich_genes,
    fit_ovr_model,
    leave_batch_out_auc,
    permutation_importance_null,
    sample_from_adata,
)

from . import pl, tl

__all__ = [
    "EvoManager",
    "Sample",
    "SampleConfig",
    "bootstrap_importance_stability",
    "classify_transition_genes",
    "cluster_mean_expression",
    "draw_net",
    "draw_network",
    "enrich_by_cell_type_programs",
    "enrich_genes",
    "finish_matplotlib_figure",
    "fit_ovr_model",
    "leave_batch_out_auc",
    "permutation_importance_null",
    "pl",
    "sample_from_adata",
    "tl",
]
__version__ = "2.1.3"
