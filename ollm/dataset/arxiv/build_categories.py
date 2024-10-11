from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from arxiv.taxonomy.definitions import ARCHIVES_ACTIVE as ARCHIVES
from arxiv.taxonomy.definitions import CATEGORIES_ACTIVE as CATEGORIES
from arxiv.taxonomy.definitions import GROUPS
from ollm.dataset import arxiv, data_model
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)


def main(_):
    """
    The arXiv taxonomy is a hierarchical classification of arXiv papers into three
    levels: groups -> archives -> categories.

    The taxonomy is hard-coded in the arxiv.taxonomy.definitions module.
    """

    # Set up
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, "build_categories", flags=FLAGS)

    # Build the graph
    G = nx.DiGraph(root=arxiv.ROOT_CATEGORY_ID)
    id_to_name = {arxiv.ROOT_CATEGORY_ID: arxiv.ROOT_CATEGORY_NAME}
    G.add_node(arxiv.ROOT_CATEGORY_ID)

    for key, value in GROUPS.items():
        id_to_name[key] = value["name"]
        if value.get("is_test", False):
            continue
        key = arxiv.normalise(key)
        G.add_node(key)
        G.add_edge(arxiv.ROOT_CATEGORY_ID, key)

    for key, value in ARCHIVES.items():
        id_to_name[key] = value["name"]
        if value.get("is_test", False):
            continue
        key = arxiv.normalise(key)
        parent_key = arxiv.normalise(value["in_group"])
        G.add_node(key)
        G.add_edge(parent_key, key)

    for key, value in CATEGORIES.items():
        id_to_name[key] = value["name"]
        key = arxiv.normalise(key)
        parent_key = arxiv.normalise(value["in_archive"])
        G.add_node(key)
        G.add_edge(parent_key, key)

    for node in G.nodes:
        G.nodes[node]["title"] = id_to_name[node]

    # Remove self loops
    G.remove_edges_from(nx.selfloop_edges(G))

    for node, data in G.nodes(data=True):
        if G.out_degree(node) == 0:
            continue
        logging.info(f"{node}: {data['title']}")
        for child in G.successors(node):
            logging.info(f"\t{child}: {G.nodes[child]['title']}")

    data_model.save_graph(G, out_dir / "raw_categories.json")


if __name__ == "__main__":
    app.run(main)
