"""LGBM-based cell-type models, cross-dataset predictions, and gene–cell networks."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.model_selection import train_test_split

from scevonet.utils import sigmoid, update_df

logger = logging.getLogger(__name__)


def _show_or_close(fig=None) -> None:
    """``plt.show()`` in interactive backends; close figure when running headless (e.g. Agg)."""
    be = matplotlib.get_backend().lower()
    if be in ("agg", "cairo", "pdf", "ps", "svg", "template"):
        plt.close(fig if fig is not None else plt.gcf())
    else:
        plt.show()


def _lgbm_regressor():
    """Import LightGBM only when training (allows importing the package without libomp)."""
    from lightgbm import LGBMRegressor

    return LGBMRegressor


@dataclass
class SampleConfig:
    """Training hyperparameters for one-vs-rest cell-type models."""

    top_features_limit: int = 3000
    n_estimators: int = 500
    test_size: float = 0.25
    random_state: int = 42
    early_stopping_rounds: int = 10


def fit_ovr_model(
    matrix: pd.DataFrame,
    y_binary: pd.Series,
    config: SampleConfig | None = None,
) -> dict[str, Any]:
    """
    Train a single one-vs-rest cell-type model using the same two-stage LightGBM
    pipeline as :class:`Sample` (full genes → top-K → refined model).

    Parameters
    ----------
    matrix
        Expression matrix (cells × genes); index must match ``y_binary``.
    y_binary
        Binary target (1 = positive cell type, 0 = other), index aligned with ``matrix``.
    """
    cfg = config or SampleConfig()
    if not y_binary.index.equals(matrix.index):
        raise ValueError("y_binary index must match matrix.index")
    LGBMRegressor = _lgbm_regressor()
    x_train, x_test, y_train, y_test = train_test_split(
        matrix,
        y_binary,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
    )
    model = LGBMRegressor(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose=False,
    )
    top_features = (
        pd.DataFrame({"gene": x_train.columns, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    genes = list(top_features["gene"][: cfg.top_features_limit])
    x_train_upd = sigmoid(update_df(x_train, genes))
    x_test_upd = sigmoid(update_df(x_test, genes))
    model_upd = LGBMRegressor(
        n_estimators=cfg.n_estimators,
        random_state=cfg.random_state,
        verbose=-1,
    )
    model_upd.fit(
        x_train_upd,
        y_train,
        eval_set=[(x_test_upd, y_test)],
        eval_metric="auc",
        early_stopping_rounds=cfg.early_stopping_rounds,
        verbose=False,
    )
    top_features_upd = (
        pd.DataFrame({"gene": x_train_upd.columns, "importance": model_upd.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {"model": model_upd, "features": top_features_upd}


class Sample:
    """Fit one LGBM regressor per cell type (1 vs rest), with top-feature refinement."""

    def __init__(
        self,
        matrix: pd.DataFrame,
        cell_types: list | pd.Series,
        config: SampleConfig | None = None,
    ):
        self.config = config or SampleConfig()
        self.models: dict[str, Any] = {}
        self.matrix = matrix
        self.cell_types = list(cell_types)
        self.cell_types_df = pd.DataFrame()
        self.models_top_features: dict[str, pd.DataFrame] = {}

        self._generate_ys()
        self._get_all_models()

    def _generate_ys(self) -> None:
        """Encode each cell type as a binary 1/0 target across cells."""
        for cell_type in sorted(set(self.cell_types)):
            states = [1 if ct == cell_type else 0 for ct in self.cell_types]
            self.cell_types_df[cell_type] = states

    def _generate_model(self, cluster_col: pd.Series) -> dict:
        return fit_ovr_model(self.matrix, cluster_col, self.config)

    def _get_all_models(self) -> None:
        for cell_type in self.cell_types_df.columns:
            result = self._generate_model(self.cell_types_df[cell_type])
            self.models[cell_type] = result["model"]
            self.models_top_features[cell_type] = result["features"]
            logger.info("Model for %s is trained", cell_type)


class EvoManager:
    """Cross-predict between datasets, merge gene–cell edges, and similarity matrices."""

    def __init__(
        self,
        *samples: Sample,
        network_top_genes: int = 75,
        filter_krt_genes: bool = True,
    ):
        self.samples = dict(enumerate(samples))
        self.network_top_genes = network_top_genes
        self.filter_krt_genes = filter_krt_genes
        self.predictions: dict[str, pd.DataFrame] = {}
        self._run_predictions()
        self.network = self._get_merged_network()
        self.cell_types_similarity = self._generate_comparison_df()

    @staticmethod
    def prepare_input_df(matrix: pd.DataFrame, top_features: list[str]) -> pd.DataFrame:
        """Align prediction matrix to training features; missing genes become NaN, then sigmoid."""
        n_rows = matrix.shape[0]
        data: dict[str, np.ndarray] = {}
        for gene in top_features:
            if gene in matrix.columns:
                data[gene] = np.asarray(matrix[gene].values, dtype=np.float64)
            else:
                data[gene] = np.full(n_rows, np.nan, dtype=np.float64)
        df = pd.DataFrame(data, index=matrix.index)
        return sigmoid(df)

    @staticmethod
    def get_means(df: pd.DataFrame) -> pd.DataFrame:
        """Mean predicted score per annotated cluster."""
        model_cols = [c for c in df.columns if c != "clusters"]
        out = df.groupby("clusters", as_index=True)[model_cols].mean()
        return out

    def between_two(self, sample1: Sample, sample2: Sample) -> pd.DataFrame:
        """Apply all classifiers from ``sample1`` to cells in ``sample2``."""
        df = pd.DataFrame({"clusters": sample2.cell_types})
        for cell_type_model in sample1.models:
            feats = sample1.models_top_features[cell_type_model]["gene"].tolist()
            mat = self.prepare_input_df(sample2.matrix, feats)
            df[cell_type_model] = sample1.models[cell_type_model].predict(mat)
        return self.get_means(df)

    def _run_predictions(self) -> None:
        n = len(self.samples)
        total = n * n
        done = 0
        for s1 in self.samples:
            for s2 in self.samples:
                done += 1
                key = f"{s1}_{s2}"
                logger.info("Predictions %s / %s (%s)", done, total, key)
                self.predictions[key] = self.between_two(self.samples[s1], self.samples[s2])

    @staticmethod
    def extract_features(
        dct: dict[str, pd.DataFrame],
        top_n: int,
        *,
        krt_remove: bool = True,
    ) -> pd.DataFrame:
        """Flatten per-cell-type feature tables into long format for the bipartite network."""
        parts: list[pd.DataFrame] = []
        for cell_type_, feat_df in dct.items():
            genes = feat_df["gene"].iloc[:top_n].tolist()
            imp = feat_df["importance"].iloc[:top_n].tolist()
            df_ = pd.DataFrame(
                {
                    "Cell_type": [cell_type_] * len(genes),
                    "Gene": genes,
                    "Importance": imp,
                }
            )
            if krt_remove:
                df_ = df_[~df_["Gene"].astype(str).str.contains("KRT", na=False)]
            parts.append(df_)
        return pd.concat(parts, ignore_index=True)

    def _get_merged_network(self) -> pd.DataFrame:
        return pd.concat(
            [
                self.extract_features(
                    self.samples[i].models_top_features,
                    self.network_top_genes,
                    krt_remove=self.filter_krt_genes,
                )
                for i in self.samples
            ],
            ignore_index=True,
        )

    @staticmethod
    def get_shortest_paths(
        network: nx.Graph, source, target, k: int = 10
    ):
        """Yield up to ``k`` shortest simple paths between ``source`` and ``target``."""
        paths = nx.shortest_simple_paths(network, source, target)
        for counter, path in enumerate(paths):
            yield path
            if counter >= k - 1:
                break

    def _generate_comparison_df(self) -> pd.DataFrame:
        """
        Similarity table used for clustermaps: z-scored predictions concatenated
        as in the paper (train-on-i models evaluated on all j).
        """
        n = len(self.samples)
        column_blocks: list[pd.DataFrame] = []
        for train_idx in range(n):
            row_parts = [self.predictions[f"{train_idx}_{j}"].apply(zscore) for j in range(n)]
            column_blocks.append(pd.concat(row_parts, axis=0))
        return pd.concat(column_blocks, axis=1)

    def cluster_map(self, visible_legend: bool = False, c_map: str = "viridis") -> None:
        """Clustermap of correlations between columns (cell-type model scores)."""
        grid = sns.clustermap(
            self.cell_types_similarity.T.corr(),
            cmap=c_map,
            yticklabels=True,
            figsize=(10, 10),
        )
        grid.cax.set_visible(visible_legend)
        _show_or_close(grid.figure)

    def get_closest_cell_types(self, cell_type: str, type_: str = "matrix") -> list:
        """Closest cell types by score column (``model``) or correlation (``matrix``)."""
        if type_ not in ("model", "matrix"):
            raise ValueError("type_ must be 'model' or 'matrix'")
        if type_ == "model":
            return list(self.cell_types_similarity.sort_values(cell_type, ascending=False).index)
        return list(self.cell_types_similarity.T.corr().sort_values(cell_type, ascending=False).index)

    def check_(
        self,
        closest_cell_types: list,
        shortest_path: list,
    ) -> bool:
        net_types = set(self.network["Cell_type"].unique())
        path_cell_types = [x for x in shortest_path if x in net_types]
        allowed = set(closest_cell_types) & net_types
        return all(x in allowed for x in path_cell_types)

    @staticmethod
    def write_connections(
        in_: list,
        out: list,
        raw_lists_for_debug: list,
        shortest_path_: list,
    ) -> None:
        raw_lists_for_debug.append(list(shortest_path_))
        for j in range(len(shortest_path_) - 1):
            in_.append(shortest_path_[j])
            out.append(shortest_path_[j + 1])

    def generate_cell_type_network(
        self,
        cluster1,
        cluster2,
        number_of_shortest_paths: int = 100,
        minimal_number_of_nodes: int = 3,
        get_only_direct: bool = False,
        closest_clusters: int = 10,
        net: bool = True,
    ):
        """Subnetwork between two cell-type nodes via shortest paths and confusion-matrix filtering."""
        in_, out_, raw_lists_for_debug = [], [], []
        subnet = pd.DataFrame()
        graph = nx.from_pandas_edgelist(
            self.network, source="Gene", target="Cell_type", edge_attr=["Importance"]
        )
        for shortest_path_ in self.get_shortest_paths(
            graph, cluster1, cluster2, number_of_shortest_paths
        ):
            if get_only_direct:
                ok = len(shortest_path_) == minimal_number_of_nodes and self.check_(
                    self.get_closest_cell_types(cluster2)[:closest_clusters],
                    shortest_path_,
                )
            else:
                ok = len(shortest_path_) >= minimal_number_of_nodes and self.check_(
                    self.get_closest_cell_types(cluster2)[:closest_clusters],
                    shortest_path_,
                )
            if ok:
                self.write_connections(in_, out_, raw_lists_for_debug, shortest_path_)
        subnet["in"] = in_
        subnet["out"] = out_
        if subnet.empty:
            empty_g = nx.Graph()
            return empty_g if net else subnet
        subnet["importance"] = [
            graph.get_edge_data(row["in"], row["out"])["Importance"]
            for _, row in subnet.iterrows()
        ]
        subgraph = nx.from_pandas_edgelist(subnet, "in", "out", edge_attr=["importance"])
        return subgraph if net else subnet.drop_duplicates()


def draw_net(
    graph,
    gene=None,
    layout: str = "2",
    font_size: int = 8,
) -> None:
    """
    Draw ``graph`` with NetworkX. If ``gene`` is an iterable of nodes, those are labeled;
    otherwise label nodes with degree ≥ 1 or betweenness ≥ 0.08.

    ``gene`` may be ``False`` for backward compatibility (treated as auto-label mode).
    """
    b = nx.betweenness_centrality(graph)
    labels: dict = {}
    use_highlight = gene not in (None, False)
    highlight = set(gene) if use_highlight else None
    for node in graph.nodes():
        if highlight is not None:
            if node in highlight:
                labels[node] = node
        else:
            if graph.degree[node] >= 1 or b.get(node, 0) >= 0.08:
                labels[node] = node
    layouts = {
        "1": nx.spring_layout(graph),
        "2": nx.kamada_kawai_layout(graph),
        "3": nx.shell_layout(graph),
    }
    pos = layouts.get(layout, layouts["2"])
    degrees = dict(graph.degree)
    nx.draw(
        graph,
        pos=pos,
        with_labels=False,
        nodelist=list(degrees.keys()),
        node_size=1500,
        alpha=0.7,
        node_color="lightblue",
        style="dashed",
        width=0.5,
    )
    nx.draw_networkx_labels(graph, pos, labels, font_size=font_size, font_color="black")
    plt.margins(x=0.4, y=0.4)
    _show_or_close(plt.gcf())
