from typing import Hashable

import graph_tool as gt
import networkx as nx


def get_gt_type(value):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    match value:
        case bool():
            return "bool"
        case int():
            return "long"
        case float():
            return "float"
        case dict():
            return "object"
        case str():
            return "string"
        case list():
            if len(value) == 0:
                return "object"
            inner_type = get_gt_type(value[0])
            if inner_type in ("bool", "long", "float", "string"):
                return f"vector<{inner_type}>"
            else:
                return "object"
        case _:
            return "object"
            raise ValueError(f"Unsupported type: {type(value)}")


def nx_to_gt(
    G_nx: nx.Graph,
) -> tuple[gt.Graph, dict[Hashable, int], dict[int, Hashable]]:
    """
    Converts a networkx graph to a graph-tool graph.
    """
    G_gt = gt.Graph(directed=G_nx.is_directed())

    # Graph properties
    for key, value in G_nx.graph.items():
        G_gt.gp[key] = G_gt.new_gp(get_gt_type(value))
        G_gt.gp[key] = value

    # Nodes and node properties (including the node id)
    nx_to_gt_map = {}  # mapping nx nodes -> gt vertices for later
    for node, data in G_nx.nodes(data=True):
        v = G_gt.add_vertex()
        nx_to_gt_map[node] = G_gt.vertex_index[v]

        for key, val in data.items():
            if key not in G_gt.vp:
                G_gt.vp[key] = G_gt.new_vp(get_gt_type(val))
            G_gt.vp[key][v] = val

    # Edges and edge properties
    for src, dst, data in G_nx.edges(data=True):
        e = G_gt.add_edge(nx_to_gt_map[src], nx_to_gt_map[dst])
        for key, val in data.items():
            if key not in G_gt.ep:
                G_gt.ep[key] = G_gt.new_ep(get_gt_type(val))
            G_gt.ep[key][e] = val

    gt_to_nx_map = {v: k for k, v in nx_to_gt_map.items()}

    return G_gt, nx_to_gt_map, gt_to_nx_map
