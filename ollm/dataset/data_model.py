import json
from pathlib import Path
from typing import TypeAlias

import networkx as nx
from absl import logging

ID: TypeAlias = int

# --- Data model ----
# {
#     "id": Any,
#     "title": str,
#     "pages": [
#         {
#             "title": str,
#             "abstract": str,
#         }
#     ]
# }


def save_graph(G: nx.Graph, save_file: Path | str, depth: int | None = None):
    """Save graph to file.

    Args:
        G: Graph to save.
        save_file: Path to the file to save the graph to. Must be a json file.
        depth: Maximum depth of the graph.
            Requires the graph to have a "root" graph attribute. Defaults to None.
    """
    save_file = Path(save_file)
    assert save_file.suffix == ".json"

    save_file.parent.mkdir(parents=True, exist_ok=True)

    if depth is not None and depth > 0:
        assert "root" in G.graph
        G = nx.ego_graph(G, G.graph["root"], radius=depth)

    logging.info(
        "Saving graph with %d nodes and %d edges to %s",
        nx.number_of_nodes(G),
        nx.number_of_edges(G),
        save_file,
    )
    with open(save_file, "w") as f:
        json.dump(nx.node_link_data(G, edges="links"), f)


def load_graph(save_file: Path | str, depth: int | None = None) -> nx.DiGraph:
    """Load graph from file. Optionally limit the depth of the graph.

    Args:
        save_file: Path to the file containing the graph. Must be a json file.
        depth: Maximum depth of the graph.
            Requires the graph to have a "root" graph attribute. Defaults to None.
    """
    save_file = Path(save_file)
    assert save_file.suffix == ".json"

    logging.info("Loading graph from %s", save_file)
    with open(save_file, "r") as f:
        G = nx.node_link_graph(json.load(f), edges="links")

    if depth is not None and depth > 0:
        assert "root" in G.graph
        G = nx.ego_graph(G, G.graph["root"], radius=depth)

    return G
