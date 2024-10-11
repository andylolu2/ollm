from dataclasses import dataclass
from functools import cache
from itertools import product

import networkx as nx
import numpy as np
from absl import logging

from ollm.eval.graph_metrics import (
    edge_prec_recall_f1,
    embed_graph,
    graph_similarity,
    node_prec_recall_f1,
)


@dataclass
class PostProcessHP:
    absolute_percentile: float = 0
    relative_percentile: float = 1
    remove_self_loops: bool = True
    remove_inverse_edges: bool = True


@cache
def self_loops(G: nx.DiGraph) -> set:
    return {(u, v) for u, v in G.edges if u == v}


@cache
def inverse_edges(G: nx.DiGraph) -> set:
    def weight(u, v):
        return G[u][v].get("weight", 1)

    return {
        (u, v) for u, v in G.edges if G.has_edge(v, u) and weight(v, u) > weight(u, v)
    }


@cache
def absolute_percentile_edges(G: nx.DiGraph, percentile: float) -> set:
    if percentile == 1:
        return set(G.edges)
    edges = list(G.edges)
    weights = np.array([G[u][v].get("weight", 1) for u, v in edges])
    bottom_indices = np.argpartition(weights, int(percentile * len(edges)))[
        : int(percentile * len(edges))
    ]
    return {edges[i] for i in bottom_indices}


@cache
def relative_percentile_edges(G: nx.DiGraph, percentile: float) -> set:
    """Nucleus pruning: keep only the top percentile_to_keep of outgoing edges from the node."""
    assert 0 <= percentile <= 1

    if percentile == 1:
        return set()

    def prune_edges_out_from_node(node):
        edges = list(G.out_edges(node))
        if len(edges) == 0:
            return set()

        weights = np.array([G[u][v].get("weight", 1) for u, v in edges])
        weights_sorted = np.sort(weights)[::-1]  # sort in descending order
        prune_idx = np.argmax(
            (weights_sorted / weights_sorted.sum()).cumsum() > percentile
        )
        prune_value = weights_sorted[prune_idx]
        to_remove = {(u, v) for (u, v), w in zip(edges, weights) if w <= prune_value}
        return to_remove

    edges_to_remove = set()
    for n in G.nodes:
        edges_to_remove.update(prune_edges_out_from_node(n))
    return edges_to_remove


def post_process(G: nx.DiGraph, hp: PostProcessHP) -> tuple[nx.DiGraph, int]:
    """Prune edges and nodes from a graph.

    Args:
        G: The input graph.
        edge_percentile: The bottom percentile of edges with the lowest weight are pruned.
        percentile_threshold: Outgoing edges with weight percentile > threshold are pruned.
        remove_self_loops: Remove self loops.
        remove_inverse_edges: Remove any pair (y, x) if p(y, x) < p(x, y).
    """
    edges_to_remove = set()
    edges_to_remove.update(absolute_percentile_edges(G, hp.absolute_percentile))
    edges_to_remove.update(relative_percentile_edges(G, hp.relative_percentile))
    if hp.remove_inverse_edges:
        edges_to_remove.update(inverse_edges(G))
    if hp.remove_self_loops:
        edges_to_remove.update(self_loops(G))

    # This also removes nodes with no incoming/outgoing edges
    G = nx.edge_subgraph(G, G.edges - edges_to_remove)

    return G, len(edges_to_remove)


def hp_search(G: nx.DiGraph, G_true: nx.DiGraph, metric: str = "edge_f1", **kwargs):
    hps = []
    keys = list(kwargs.keys())
    for values in product(*kwargs.values()):
        hps.append(PostProcessHP(**dict(zip(keys, values))))
    assert len(hps) > 0, "No hyperparameters to search over"

    if metric == "edge_f1":
        score_fn = lambda G_pred, G_true: edge_prec_recall_f1(G_pred, G_true)[2]
    elif metric == "node_f1":
        score_fn = lambda G_pred, G_true: node_prec_recall_f1(G_pred, G_true)[2]
    elif metric.startswith("graph_similarity"):
        n_iters = int(metric.split("_")[-1])
        G = embed_graph(G)  # type: ignore
        G_true = embed_graph(G_true)  # type: ignore
        score_fn = lambda G_pred, G_true: graph_similarity(
            G_pred, G_true, direction="undirected", n_iters=n_iters
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    best = (None, None, -float("inf"))  # best hp, best G, best score
    for hp in hps:
        G_pred = post_process(G, hp)
        score = score_fn(G_pred, G_true)
        if score is None:
            continue
        logging.info("Score: %.5f, HP: %s", score, hp)
        if score > best[2]:
            best = (hp, G_pred, score)
    return best
