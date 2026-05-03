<p align="center">
  <img src="assets/logo.svg" alt="scEvoNet" width="480">
</p>

# scEvoNet

**scEvoNet** builds cross-dataset **cell-state similarity** and **gene–cell** networks from scRNA-seq using gradient-boosted one-vs-rest models (LightGBM), as described in [BMC Bioinformatics, 2023](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-023-05213-3) ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9990205/)).

It is useful for comparing **species**, **developmental stages**, or **tumor vs metastasis** when you can provide **expression matrices** and **cell type labels** per sample.

## Install

```bash
pip install scevonet
```

Optional extras:

```bash
pip install 'scevonet[enrichment]'   # gseapy / Enrichr ORA
pip install 'scevonet[anndata]'     # AnnData / Scanpy helpers
pip install 'scevonet[all]'         # both
```

## Quick start

```python
from scevonet import Sample, EvoManager

# One Sample per matrix + cell type labels; then compare datasets
# em = EvoManager(sample_a, sample_b)
# em.cell_types_similarity
# em.network
```

See **[examples/HowToUse.ipynb](examples/HowToUse.ipynb)** for a full walkthrough (same notebook as the published guide).

## Features (v2.1+)

- **Networks & similarity** — bipartite gene–cell graph, optional shortest-path subnetworks, cross-dataset score correlations.
- **Programs** — `classify_transition_genes` (pseudobulk directionality), optional **Enrichr** enrichment (`enrich_genes`, `enrich_by_cell_type_programs`).
- **Validation** — bootstrap importance stability, label-permutation null, leave-one-batch-out AUC.
- **Plotting** — `draw_net` (quick) and `draw_network` (bipartite-aware styling, edge weights from importance).

## Citation

Kotov et al., *BMC Bioinformatics* (2023). [DOI 10.1186/s12859-023-05213-3](https://doi.org/10.1186/s12859-023-05213-3)

## Links

- [Monsoro-Burq lab](https://curie.fr/equipe/monsoro-burq)
- [PyPI — scevonet](https://pypi.org/project/scevonet)
