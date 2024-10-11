import json
from collections import defaultdict
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_file", None, "Path to the inference ouptut file", required=True
)
flags.DEFINE_multi_string(
    "relations",
    ["subclass of", "instance of", "member of", "part of"],
    "List of relations to extract",
)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)


def parse_triplets(triplets: list[dict[str, str]]) -> set[tuple[str, str]]:
    extracted = set()
    for triplet in triplets:
        match triplet:
            case {"head": head, "type": relation, "tail": tail}:
                if relation in FLAGS.relations:
                    extracted.add((tail, head))
            case _:
                logging.error("Invalid triplet: %s", triplet)
    return extracted


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    relations = defaultdict(int)
    with open(FLAGS.input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            for parent, child in parse_triplets(item["triplets"]):
                relations[(parent, child)] += 1

    G = nx.DiGraph()
    for (parent, child), weight in relations.items():
        G.add_node(parent, title=parent)
        G.add_node(child, title=child)
        G.add_edge(parent, child, weight=weight)
    G.graph["root"] = None

    logging.info("Number of nodes: %d", nx.number_of_nodes(G))
    logging.info("Number of edges: %d", nx.number_of_edges(G))

    data_model.save_graph(G, out_dir / "graph.json")


if __name__ == "__main__":
    app.run(main)
