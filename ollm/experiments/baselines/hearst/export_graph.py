import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.utils import textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "extraction_dir", None, "Directory containing the extration files", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output file", required=True)


def main(_):
    extraction_dir = Path(FLAGS.extraction_dir)
    extraction_files = list(extraction_dir.glob("*.txt.conll"))
    logging.info("Loading extractions from %s", extraction_dir)

    pattern = re.compile(r"(?P<child>.*)\|\|\|(?P<parent>.*)\|\|\|(?P<rule>.*)")

    hyponyms = defaultdict(int)
    for extraction_file in textqdm(extraction_files):
        with open(extraction_file, "r") as f:
            for line in f:
                match = pattern.match(line)
                if match is None:
                    continue
                child = match.group("child").strip()
                parent = match.group("parent").strip()
                hyponyms[(parent, child)] += 1

    # Export to a graph
    G = nx.DiGraph()
    for (parent, child), weight in hyponyms.items():
        G.add_node(parent, title=parent)
        G.add_node(child, title=child)
        G.add_edge(parent, child, weight=weight)
    logging.info(
        "Extracted %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
    )

    G.graph["root"] = None

    data_model.save_graph(G, Path(FLAGS.output_dir) / "graph.json")


if __name__ == "__main__":
    app.run(main)
