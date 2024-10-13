import json
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from ollm.dataset.data_model import load_graph
from ollm.eval.graph_metrics import (
    fuzzy_and_continuous_precision_recall_f1,
    graph_precision_recall_f1,
    literal_prec_recall_f1,
    motif_distance,
)
from ollm.experiments.post_processing import PostProcessHP, post_process
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph", None, "Path to the graph to evaluate.", required=True)
flags.DEFINE_string(
    "graph_true", None, "Path to the ground truth graph file.", required=True
)
flags.DEFINE_string(
    "hp_search_result", None, "Path to the hyperparameter search result."
)
flags.DEFINE_string("output_file", None, "Output file.", required=True)
flags.DEFINE_string("best_hp_metric", "continuous_f1", "Metric to use for best HP.")


def evaluate(G, G_true, hp):
    G, _ = post_process(G, hp)
    precision, recall, f1 = literal_prec_recall_f1(G, G_true)
    soft_precision, soft_recall, soft_f1, hard_precision, hard_recall, hard_f1 = (
        fuzzy_and_continuous_precision_recall_f1(
            G, G_true, match_threshold=0.436, skip_if_too_slow=False
        )
    )
    soft_graph_precision, soft_graph_recall, soft_graph_f1 = graph_precision_recall_f1(
        G, G_true, direction="forward", n_iters=2
    )
    motif_wass = motif_distance(G, G_true, n=3)

    return {
        "num_nodes": nx.number_of_nodes(G),
        "num_edges": nx.number_of_edges(G),
        "edge_f1": f1,
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_soft_precision": soft_precision,
        "edge_soft_recall": soft_recall,
        "edge_soft_f1": soft_f1,
        "edge_hard_precision": hard_precision,
        "edge_hard_recall": hard_recall,
        "edge_hard_f1": hard_f1,
        "graph_soft_precision": soft_graph_precision,
        "graph_soft_recall": soft_graph_recall,
        "graph_soft_f1": soft_graph_f1,
        "motif_wass": motif_wass,
    }


def main(_):
    output_file = Path(FLAGS.output_file)
    setup_logging(output_file.parent, "test_metrics", flags=FLAGS)

    # Get the best HPs
    logging.info("Loading HP search results from %s", FLAGS.hp_search_result)
    with open(FLAGS.hp_search_result, "r") as f:
        hp_search_results = [json.loads(line) for line in f]
    best_hp = PostProcessHP(
        **max(hp_search_results, key=lambda x: x[FLAGS.best_hp_metric])["hp"]
    )
    logging.info("Best HP: %s", best_hp)

    # Compute the metrics
    G_true = load_graph(FLAGS.graph_true)

    G = load_graph(FLAGS.graph)
    G, _ = post_process(G, best_hp)

    literal_precision, literal_recall, literal_f1 = literal_prec_recall_f1(G, G_true)
    (
        continuous_precision,
        continuous_recall,
        continuous_f1,
        fuzzy_precision,
        fuzzy_recall,
        fuzzy_f1,
    ) = fuzzy_and_continuous_precision_recall_f1(
        G, G_true, match_threshold=0.436, skip_if_too_slow=False
    )
    graph_precision, graph_recall, graph_f1 = graph_precision_recall_f1(
        G, G_true, direction="forward", n_iters=2
    )
    motif_dist = motif_distance(G, G_true, n=3)

    metrics = {
        "num_nodes": nx.number_of_nodes(G),
        "num_edges": nx.number_of_edges(G),
        "literal_precision": literal_precision,
        "literal_recall": literal_recall,
        "literal_f1": literal_f1,
        "continuous_precision": continuous_precision,
        "continuous_recall": continuous_recall,
        "continuous_f1": continuous_f1,
        "fuzzy_precision": fuzzy_precision,
        "fuzzy_recall": fuzzy_recall,
        "fuzzy_f1": fuzzy_f1,
        "graph_precision": graph_precision,
        "graph_recall": graph_recall,
        "graph_f1": graph_f1,
        "motif_dist": motif_dist,
    }

    # Pretty print the results
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k:<20}: {v:.3f}")
        else:
            print(f"{k:<20}: {v}")

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    app.run(main)
