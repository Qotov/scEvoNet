"""Tools: models, programs, validation, and AnnData helpers (Scanpy ``scanpy.tl`` style)."""

from scevonet.tl._adata import sample_from_adata
from scevonet.tl._models import EvoManager, Sample, SampleConfig, fit_ovr_model
from scevonet.tl._programs import (
    classify_transition_genes,
    cluster_mean_expression,
    enrich_by_cell_type_programs,
    enrich_genes,
)
from scevonet.tl._validation import (
    bootstrap_importance_stability,
    leave_batch_out_auc,
    permutation_importance_null,
)

__all__ = [
    "EvoManager",
    "Sample",
    "SampleConfig",
    "bootstrap_importance_stability",
    "classify_transition_genes",
    "cluster_mean_expression",
    "enrich_by_cell_type_programs",
    "enrich_genes",
    "fit_ovr_model",
    "leave_batch_out_auc",
    "permutation_importance_null",
    "sample_from_adata",
]
