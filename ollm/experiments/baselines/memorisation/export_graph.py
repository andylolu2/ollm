import json
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from llm_ol.dataset import data_model

FLAGS = flags.FLAGS
flags.DEFINE_string("train_dataset", None, "Train dataset to memorise", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output file", required=True)


def main(_):
    G = nx.DiGraph()

    with open(FLAGS.train_dataset, "r") as f:
        for line in f:
            item = json.loads(line)

            edges = set()
            for path in item["paths"]:
                edges.update(zip(path[:-1], path[1:]))

            for u, v in edges:
                if u not in G:
                    G.add_node(u, title=u)
                if v not in G:
                    G.add_node(v, title=v)
                if not G.has_edge(u, v):
                    G.add_edge(u, v, weight=0)
                G.edges[u, v]["weight"] += 1
    logging.info(
        "Extracted %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
    )
    G.graph["root"] = "Main topic classifications"

    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph.json")


if __name__ == "__main__":
    app.run(main)
