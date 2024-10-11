import json
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from absl import app, flags, logging

from llm_ol.dataset import data_model
from llm_ol.experiments.hearst.svd_ppmi import SvdPpmiModel
from llm_ol.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "input_file", None, "Path to the inference ouptut file", required=True
)
flags.DEFINE_string("graph_true", None, "The ground truth graph", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_multi_integer(
    "k", None, "The number of dimensions for the SVD", required=True
)
flags.DEFINE_multi_string(
    "relations",
    ["subclass of", "instance of", "member of", "part of"],
    "List of relations to extract",
)
flags.DEFINE_float("factor", 10, "Ratio of number of edges to keep vs nodes")


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

    # Use the ground truth graph to get the true concepts
    G_true = data_model.load_graph(FLAGS.graph_true)
    true_concepts = set(G_true.nodes[n]["title"] for n in G_true.nodes)
    root_concept = G_true.nodes[G_true.graph["root"]]["title"]

    matches = []
    with open(FLAGS.input_file, "r") as f:
        for line in f:
            item = json.loads(line)
            for parent, child in parse_triplets(item["triplets"]):
                matches.append((parent, child))
    concepts = set()
    for parent, child in matches:
        concepts.add(parent)
        concepts.add(child)

    vocab = {concept: i + 1 for i, concept in enumerate(concepts)}
    vocab["<OOV>"] = 0
    logging.info("Voabulary size: %d", len(vocab))
    csr_m = sp.dok_matrix((len(vocab), len(vocab)), dtype=np.float64)

    for parent, child in matches:
        csr_m[vocab[parent], vocab[child]] += 1

    csr_m = sp.csr_matrix(csr_m)

    for k in FLAGS.k:
        model = SvdPpmiModel(csr_m, vocab, k)
        nodes = list(true_concepts)
        heads, tails = zip(*product(nodes, nodes))
        weights = model.predict_many(heads, tails)

        n_edges_to_keep = min(int(len(nodes) * FLAGS.factor), len(weights))
        top_indices = np.argpartition(weights, -n_edges_to_keep)[-n_edges_to_keep:]

        # Export to a graph
        G = nx.DiGraph()
        for concept in true_concepts:
            G.add_node(concept, title=concept)
        for idx in top_indices:
            head = heads[idx]
            tail = tails[idx]
            weight = weights[idx]
            G.add_edge(head, tail, weight=weight)
        G.graph["root"] = root_concept

        logging.info(
            "Extracted %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )

        data_model.save_graph(G, Path(FLAGS.output_dir) / f"k_{k}" / "graph.json")


if __name__ == "__main__":
    app.run(main)
