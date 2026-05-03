"""Publication-friendly network figures for gene–cell bipartite graphs."""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.bipartite as nx_bi
import numpy as np


def finish_matplotlib_figure(fig=None) -> None:
    """Call ``show()`` on interactive backends; close when headless (e.g. Agg)."""
    be = matplotlib.get_backend().lower()
    if be in ("agg", "cairo", "pdf", "ps", "svg", "template"):
        plt.close(fig if fig is not None else plt.gcf())
    else:
        plt.show()


def _edge_weights(graph: nx.Graph, attr: str | None) -> list[float]:
    if not graph.edges():
        return []
    if attr is None:
        return [1.0] * graph.number_of_edges()
    out = []
    for _u, _v, data in graph.edges(data=True):
        w = data.get(attr)
        out.append(float(w) if w is not None and np.isfinite(w) else 1.0)
    return out


def _normalize_widths(weights: list[float], lo: float = 0.4, hi: float = 3.0) -> list[float]:
    if not weights:
        return []
    arr = np.asarray(weights, dtype=float)
    if arr.max() <= arr.min():
        return [(lo + hi) / 2.0] * len(weights)
    z = (arr - arr.min()) / (arr.max() - arr.min())
    return list(lo + z * (hi - lo))


def _layout(graph: nx.Graph, layout: str, seed: int) -> dict:
    if layout == "spring":
        return nx.spring_layout(graph, seed=seed)
    if layout == "shell":
        return nx.shell_layout(graph)
    if layout == "planar":
        try:
            return nx.planar_layout(graph)
        except Exception:
            return nx.spring_layout(graph, seed=seed)
    # default: kamada_kawai often nicest for small/medium graphs
    try:
        return nx.kamada_kawai_layout(graph)
    except Exception:
        return nx.spring_layout(graph, seed=seed)


def draw_network(
    graph: nx.Graph,
    *,
    gene=None,
    layout: str = "kamada_kawai",
    font_size: int = 9,
    edge_weight_attr: str | None = "importance",
    node_size_cell: int = 2200,
    node_size_gene: int = 900,
    color_cell: str = "#3d5a80",
    color_gene: str = "#ee6c4d",
    color_default: str = "#90a4ae",
    seed: int = 42,
    figsize: tuple[float, float] = (10.5, 8),
    title: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Draw a gene–cell (or general) network with distinct styling for the two partitions.

    Parameters
    ----------
    gene
        Nodes to always label: a single gene/cell name (``str``), an iterable of node names,
        or ``None`` / ``False`` / ``True`` for automatic labeling of salient nodes (degree or
        betweenness). ``True`` is treated like ``None`` (useful as a legacy boolean flag).
    layout
        ``kamada_kawai`` (default), ``spring``, ``shell``, or ``planar``.
    edge_weight_attr
        Edge attribute for line width (e.g. ``importance``). ``None`` uses uniform widths.
    """
    b = nx.betweenness_centrality(graph)
    labels: dict = {}
    # gene: None / False / True → automatic salient labels; str → one highlighted node;
    # iterable of str → those nodes (not ``set(str)``, which would split characters).
    highlight: set | None
    if gene in (None, False, True):
        highlight = None
    elif isinstance(gene, str):
        highlight = {gene}
    else:
        highlight = set(gene)
    for node in graph.nodes():
        if highlight is not None:
            if node in highlight:
                labels[node] = str(node)
        else:
            if graph.degree[node] >= 1 or b.get(node, 0) >= 0.08:
                labels[node] = str(node)

    # Two-color bipartite partition when possible
    node_color: list[str] = []
    cdict: dict | None = None
    try:
        cdict = nx_bi.color(graph)
        for n in graph.nodes():
            node_color.append(color_gene if cdict[n] == 0 else color_cell)
    except Exception:
        node_color = [color_default] * graph.number_of_nodes()

    sizes = []
    for n in graph.nodes():
        if cdict is not None:
            sizes.append(node_size_gene if cdict[n] == 0 else node_size_cell)
        else:
            sizes.append(1600)

    pos = _layout(graph, layout, seed)
    widths = _normalize_widths(_edge_weights(graph, edge_weight_attr))

    fig, ax = plt.subplots(figsize=figsize, facecolor="#fafafa")
    ax.set_facecolor("#fafafa")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, color="#1b4965", pad=12, fontweight="600")

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=widths if widths else 1.0,
        edge_color="#b0bec5",
        alpha=0.9,
        style="solid",
    )
    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        nodelist=list(graph.nodes()),
        node_size=sizes,
        node_color=node_color,
        linewidths=1.2,
        edgecolors="#ffffff",
        alpha=0.95,
    )
    nx.draw_networkx_labels(graph, pos, labels, ax=ax, font_size=font_size, font_color="#1a1a1a")
    plt.margins(x=0.12, y=0.12)
    fig.tight_layout()
    return fig


def draw_net(
    graph,
    gene=None,
    layout: str = "2",
    font_size: int = 8,
) -> None:
    """
    Plot ``graph`` (backward compatible with legacy layout codes ``1``/``2``/``3``).

    Non-interactive backends close the figure after drawing; others display it.
    """
    layout_map = {"1": "spring", "2": "kamada_kawai", "3": "shell"}
    lay = layout_map.get(layout, "kamada_kawai")
    if layout not in layout_map:
        warnings.warn(
            f"Unknown layout code {layout!r}; use '1','2','3' or pass a "
            "scevonet.pl.draw_network layout name.",
            stacklevel=2,
        )
    fig = draw_network(graph, gene=gene, layout=lay, font_size=font_size)
    finish_matplotlib_figure(fig)
