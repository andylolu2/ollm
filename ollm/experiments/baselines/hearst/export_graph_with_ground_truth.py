import re
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import scipy.sparse as sp
from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.experiments.baselines.hearst.svd_ppmi import SvdPpmiModel
from ollm.utils import setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "extraction_dir", None, "Directory containing the extration files", required=True
)
flags.DEFINE_string("graph_true", None, "The ground truth graph", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output file", required=True)
flags.DEFINE_multi_integer(
    "k", None, "The number of dimensions for the SVD", required=True
)
flags.DEFINE_float("factor", 10, "Ratio of number of edges to keep vs nodes")


pattern = re.compile(r"(?P<child>.*)\|\|\|(?P<parent>.*)\|\|\|(?P<rule>.*)")


def main(_):
    setup_logging(Path(FLAGS.output_dir), "export_graph", flags=FLAGS)
    extraction_dir = Path(FLAGS.extraction_dir)
    extraction_files = list(extraction_dir.glob("*.txt.conll"))
    logging.info("Loading extractions from %s", extraction_dir)

    matches = []
    for extraction_file in textqdm(extraction_files):
        with open(extraction_file, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match is None:
                    continue
                child = match.group("child").strip()
                parent = match.group("parent").strip()
                matches.append((parent, child))

    # Use the ground truth graph to get the true concepts
    G_true = data_model.load_graph(FLAGS.graph_true)
    true_concepts = set(G_true.nodes[n]["title"] for n in G_true.nodes)

    concepts = set()
    for parent, child in matches:
        concepts.add(parent)
        concepts.add(child)

    vocab = {concept: i + 1 for i, concept in enumerate(concepts)}
    vocab["<OOV>"] = 0
    csr_m = sp.dok_matrix((len(vocab), len(vocab)), dtype=np.float64)
    logging.info("Voabulary size: %d", len(vocab))

    for parent, child in matches:
        csr_m[vocab[child], vocab[parent]] += 1

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

        logging.info(
            "Extracted %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )

        data_model.save_graph(G, Path(FLAGS.output_dir) / f"k_{k}" / "graph.json")


if __name__ == "__main__":
    app.run(main)
