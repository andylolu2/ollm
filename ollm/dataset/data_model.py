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


def clean_up_graph(G: nx.Graph):
    # Remove nodes not reachable from the root.
    nodes_to_keep = nx.descendants(G, G.graph["root"]) | {G.graph["root"]}
    logging.info(
        "Removing %d nodes not reachable from the root", len(G) - len(nodes_to_keep)
    )
    G = G.subgraph(nodes_to_keep)

    # Remove nodes with no pages or descendants that have pages.
    nodes_with_pages = set(n for n, pages in G.nodes(data="pages") if len(pages) > 0)  # type: ignore
    nodes_to_keep = set(
        nx.multi_source_dijkstra_path_length(
            G.reverse() if isinstance(G, nx.DiGraph) else G, nodes_with_pages
        ).keys()
    )
    logging.info(
        "Removing %d nodes with no pages or descendants that have pages",
        len(G) - len(nodes_to_keep),
    )
    G = G.subgraph(nodes_to_keep)

    return G.copy()


def save_graph(
    G: nx.Graph, save_file: Path | str, depth: int | None = None, clean_up: bool = False
):
    """Save graph to file.

    Args:
        G: Graph to save.
        save_file: Path to the file to save the graph to. Must be a json file.
        depth: Maximum depth of the graph.
            Requires the graph to have a "root" graph attribute. Defaults to None.
    """
    save_file = Path(save_file)
    assert "root" in G.graph
    assert save_file.suffix == ".json"

    save_file.parent.mkdir(parents=True, exist_ok=True)

    if depth is not None and depth > 0:
        G = nx.ego_graph(G, G.graph["root"], radius=depth)

    if clean_up:
        G = clean_up_graph(G)

    logging.info("Saving graph to %s", save_file)
    with open(save_file, "w") as f:
        json.dump(nx.node_link_data(G), f)


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
        G = nx.node_link_graph(json.load(f))

    assert "root" in G.graph
    if depth is not None and depth > 0:
        G = nx.ego_graph(G, G.graph["root"], radius=depth)

    return G
