import json
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "hierarchy_file", None, "Path to the hierarchy directory", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)

pattern = re.compile(r"Main topic classifications( -> ((?!(\n|->)).)+)+")
empty_pattern = re.compile(r"\s*")


def parse_hierarchy(hierarchy_str: str):
    paths = hierarchy_str.split("\n")
    relations = set()
    total = 0
    num_invalid = 0
    for path in paths:
        path = path.strip()
        if empty_pattern.fullmatch(path) is not None:
            continue

        total += 1
        if pattern.fullmatch(path) is None:
            num_invalid += 1
            logging.debug("Invalid pattern: %s", path)
            continue
        nodes = path.split(" -> ")
        for parent, child in zip(nodes[:-1], nodes[1:]):
            relations.add((parent, child))
    return relations, total, num_invalid


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    results = []
    for hierarchy_file in FLAGS.hierarchy_file:
        with open(hierarchy_file, "r") as f:
            results += [json.loads(line) for line in f.readlines()]

    hypernyms = defaultdict(int)
    num_samples = len(results)
    num_invalid, num_paths, num_invalid_paths = 0, 0, 0
    for item in results:
        relations, total, invalid = parse_hierarchy(item["hierarchy"])
        num_paths += total
        num_invalid_paths += invalid
        num_invalid += 1 if invalid > 0 else 0
        try:
            for parent, child in relations:
                hypernyms[(parent, child)] += 1
        except Exception as e:
            logging.error("Error parsing hierarchy %s: %s", item["title"], e)

    logging.info("Total of %s samples", num_samples)
    logging.info(
        "Total of %s invalid samples (%.2f%%)",
        num_invalid,
        num_invalid / num_samples * 100,
    )
    logging.info("Total of %s paths", num_paths)
    logging.info(
        "Total of %s invalid paths (%.2f%%)",
        num_invalid_paths,
        num_invalid_paths / num_paths * 100,
    )
    logging.info("Total of %s relations", len(hypernyms))

    G = nx.DiGraph()
    for (parent, child), count in hypernyms.items():
        G.add_node(parent, title=parent)
        G.add_node(child, title=child)
        G.add_edge(parent, child, weight=count)

    data_model.save_graph(G, out_dir / "graph.json")


if __name__ == "__main__":
    app.run(main)
