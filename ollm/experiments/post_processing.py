from dataclasses import dataclass
from functools import cache

import networkx as nx
import numpy as np


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
