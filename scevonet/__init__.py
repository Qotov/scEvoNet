"""scEvoNet: gradient boosting-based cell state evolution and gene–cell networks."""

from scevonet.adata import sample_from_adata
from scevonet.net import EvoManager, Sample, SampleConfig, draw_net, fit_ovr_model
from scevonet.programs import (
    classify_transition_genes,
    cluster_mean_expression,
    enrich_by_cell_type_programs,
    enrich_genes,
)
from scevonet.validation import (
    bootstrap_importance_stability,
    leave_batch_out_auc,
    permutation_importance_null,
)

__all__ = [
    "EvoManager",
    "Sample",
    "SampleConfig",
    "draw_net",
    "fit_ovr_model",
    "sample_from_adata",
    "cluster_mean_expression",
    "classify_transition_genes",
    "enrich_genes",
    "enrich_by_cell_type_programs",
    "bootstrap_importance_stability",
    "permutation_importance_null",
    "leave_batch_out_auc",
    "__version__",
]
__version__ = "2.1.0"
