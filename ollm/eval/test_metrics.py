import dataclasses
import json
from pathlib import Path

import graph_tool.all as gt
import networkx as nx
import numpy as np
from absl import app, flags, logging

from ollm.dataset.data_model import load_graph
from ollm.eval.graph_metrics import (
    edge_prec_recall_f1,
    edge_similarity,
    graph_fuzzy_match,
)
from ollm.experiments.post_processing import PostProcessHP, post_process
from ollm.utils import setup_logging
from ollm.utils.nx_to_gt import nx_to_gt

from metadata import query  # isort:skip

FLAGS = flags.FLAGS
flags.DEFINE_string("output_file", None, "Path to the output file.", required=True)
flags.DEFINE_string("best_hp_metric", "edge_soft_f1", "Metric to use for best HP.")
flags.DEFINE_string("dataset", "wikipedia/v2", "Dataset to evaluate.")


def motifs_wasserstein(G_pred: nx.Graph, G_true: nx.Graph, n: int = 3):
    motifs_pred, counts_pred = gt.motifs(nx_to_gt(G_pred)[0], n)  # type: ignore
    motifs_true, counts_true = gt.motifs(nx_to_gt(G_true)[0], n)  # type: ignore

    all_motifs = motifs_pred[::]
    for motif in motifs_true:
        for existing_motif in all_motifs:
            if gt.isomorphism(motif, existing_motif):
                break
        else:
            all_motifs.append(motif)
    all_counts_pred = np.zeros(len(all_motifs))
    all_counts_true = np.zeros(len(all_motifs))
    for i, motif in enumerate(motifs_pred):
        for j, existing_motif in enumerate(all_motifs):
            if gt.isomorphism(motif, existing_motif):
                all_counts_pred[j] = counts_pred[i]
                break
    for i, motif in enumerate(motifs_true):
        for j, existing_motif in enumerate(all_motifs):
            if gt.isomorphism(motif, existing_motif):
                all_counts_true[j] = counts_true[i]
                break
    all_counts_pred /= all_counts_pred.sum()
    all_counts_true /= all_counts_true.sum()

    wass = np.sum(np.abs(all_counts_true - all_counts_pred)) / 2
    return wass


def evaluate(G, G_true, hp):
    G, _ = post_process(G, hp)
    precision, recall, f1 = edge_prec_recall_f1(G, G_true)
    soft_precision, soft_recall, soft_f1, hard_precision, hard_recall, hard_f1 = (
        edge_similarity(G, G_true, match_threshold=0.436, skip_if_too_slow=False)
    )
    soft_graph_precision, soft_graph_recall, soft_graph_f1 = graph_fuzzy_match(
        G, G_true, direction="forward", n_iters=2
    )
    motif_wass = motifs_wasserstein(G, G_true, n=3)

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

    exps = [
        query(exp="link_prediction", dataset=FLAGS.dataset),
        query(exp="cot", dataset=FLAGS.dataset),
        query(exp="memorisation", dataset=FLAGS.dataset),
        query(exp="hearst", dataset=FLAGS.dataset),
        query(exp="rebel", dataset=FLAGS.dataset),
        query(exp="prompting", k_shot=0, dataset=FLAGS.dataset),
        query(exp="prompting", k_shot=1, dataset=FLAGS.dataset),
        query(exp="prompting", k_shot=3, dataset=FLAGS.dataset),
        query(exp="finetune", reweighted=False, transfer=False, dataset=FLAGS.dataset),
        query(exp="finetune", reweighted=True, transfer=False, dataset=FLAGS.dataset),
    ]
    if FLAGS.dataset == "arxiv/v2":
        exps += [
            query(
                exp="finetune", reweighted=False, transfer=True, dataset=FLAGS.dataset
            ),
            query(
                exp="finetune", reweighted=True, transfer=True, dataset=FLAGS.dataset
            ),
        ]

    with output_file.open("w") as f:
        for exp in exps:
            logging.info("Evaluating %s", exp.name)
            G = load_graph(exp.test_output)
            G_true = load_graph(exp.test_ground_truth)
            hp = PostProcessHP(**exp.best_hp(FLAGS.best_hp_metric))
            logging.info("HP: %s", hp)
            metrics = evaluate(G, G_true, hp)
            logging.info("Metrics: %s", metrics)

            item = {"name": exp.name, "hp": dataclasses.asdict(hp), **metrics}
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    app.run(main)
