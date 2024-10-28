# There some conflict between graph-tools and torch, need to import gt first
import graph_tool  # noqa # isort:skip

import dataclasses
import json
from itertools import product
from pathlib import Path

import numpy as np
from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.eval.graph_metrics import (
    fuzzy_and_continuous_precision_recall_f1,
)
from ollm.experiments.post_processing import PostProcessHP, post_process
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph", None, "Path to the graph file.", required=True)
flags.DEFINE_string(
    "graph_true", None, "Path to the ground truth graph file.", required=True
)
flags.DEFINE_integer("num_samples", 11, "Number of thresholds to evaluate.")
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "hp_search", flags=FLAGS)

    out_file = out_dir / "hp_search.jsonl"
    if out_file.exists():
        out_file.unlink()

    G = data_model.load_graph(FLAGS.graph)
    G_true = data_model.load_graph(FLAGS.graph_true)

    absolute_percentiles = 1 - np.geomspace(
        1 / G.number_of_edges(), 1, FLAGS.num_samples
    )
    # reverse to start with the most memory-intensive HPs
    absolute_percentiles = absolute_percentiles[::-1]
    relative_percentiles = 1 - np.geomspace(0.1, 1, FLAGS.num_samples) + 0.1

    for absolute_percentile, relative_percentile in product(
        absolute_percentiles, relative_percentiles
    ):
        hp = PostProcessHP(absolute_percentile, relative_percentile)
        G_pruned, n_removed = post_process(G, hp)
        (
            continuous_precision,
            continuous_recall,
            continuous_f1,
            fuzzy_precision,
            fuzzy_recall,
            fuzzy_f1,
        ) = fuzzy_and_continuous_precision_recall_f1(
            G_pruned, G_true, match_threshold=0.436
        )

        item = {
            "continuous_precision": continuous_precision,
            "continuous_recall": continuous_recall,
            "continuous_f1": continuous_f1,
            "fuzzy_precision": fuzzy_precision,
            "fuzzy_recall": fuzzy_recall,
            "fuzzy_f1": fuzzy_f1,
            "hp": dataclasses.asdict(hp),
        }

        logging.info("Results: %s", json.dumps(item, indent=2))
        with out_file.open("a") as f:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)

if __name__ == "__main__":
    app.run(main)
