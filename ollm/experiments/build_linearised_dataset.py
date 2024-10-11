import json
import random
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import graph_tool.all as gt
import networkx as nx
import numpy as np
from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.utils import setup_logging, textpbar
from ollm.utils.nx_to_gt import nx_to_gt

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "graph_file", None, "Path to the train split of the graph file", required=True
)
flags.DEFINE_string("output_file", None, "Output file", required=True)
flags.DEFINE_integer("cutoff", 5, "Maximum path length from the root to the page")
flags.DEFINE_integer("num_workers", 8, "Number of workers to use")
flags.DEFINE_integer("seed", 0, "Random seed")


def paths_from_root(
    G_gt: gt.Graph,
    root_idx: int,
    page_categories_idxs: list[int],
    cutoff: None | int = None,
):
    """Find the simple paths with len <= cutoff from the root to the page."""

    # Temporarily add the page to the graph
    page_node = G_gt.add_vertex()
    for idx in page_categories_idxs:
        G_gt.add_edge(idx, page_node)

    try:
        paths = gt.all_paths(G_gt, source=root_idx, target=page_node, cutoff=cutoff)
        paths = {tuple(path[:-1]) for path in paths}
        paths = [list(path) for path in paths]
        random.shuffle(paths)
        return paths
    finally:
        G_gt.remove_vertex(page_node)


def make_training_samples(G: nx.Graph):
    G = G.copy()
    G_gt, nx_to_gt_map, gt_to_nx_map = nx_to_gt(G)

    pages = {}
    for node, data in G.nodes(data=True):
        for page in data.pop("pages"):
            id_ = page["id"]
            if id_ not in pages:
                pages[id_] = {**page, "categories": [node]}
            else:
                pages[id_]["categories"].append(node)
    pages = list(pages.values())
    category_idxs = [
        [nx_to_gt_map[category] for category in page["categories"]] for page in pages
    ]

    not_covered_edges = set(G.edges())
    num_paths = []
    path_lengths = []
    pbar = textpbar(len(pages))
    map_fn = partial(
        paths_from_root,
        G_gt,
        nx_to_gt_map[G.graph["root"]],
        cutoff=FLAGS.cutoff,
    )
    with Pool(FLAGS.num_workers) as p:
        for page, paths in zip(pages, p.imap(map_fn, category_idxs, chunksize=5000)):
            if len(paths) == 0:
                logging.warning("No paths found for page %s", page["title"])
                continue

            path_titles = [
                [G.nodes[gt_to_nx_map[v]]["title"] for v in path] for path in paths
            ]
            yield {
                "id": page["id"],
                "title": page["title"],
                "abstract": page["abstract"],
                "paths": path_titles,
            }
            pbar.update()
            num_paths.append(len(paths))
            path_lengths += [len(path) for path in paths]
            for path in paths:
                for u, v in zip(path[:-1], path[1:]):
                    not_covered_edges.discard((gt_to_nx_map[u], gt_to_nx_map[v]))

    logging.info("Number of samples: %d/%d", len(num_paths), len(pages))
    logging.info(
        "Number of paths quantiles: %s (5 | 25 | 50 | 75 | 95). Mean: %.3f",
        np.percentile(num_paths, [5, 25, 50, 75, 95]),
        np.mean(num_paths),
    )
    logging.info(
        "Path length quantiles: %s (5 | 25 | 50 | 75 | 95). Mean: %.3f",
        np.percentile(path_lengths, [5, 25, 50, 75, 95]),
        np.mean(path_lengths),
    )
    logging.info(
        "Edges not covered by any path: %d/%d (%.2f%%)",
        len(not_covered_edges),
        G.number_of_edges(),
        len(not_covered_edges) / G.number_of_edges() * 100,
    )


def main(_):
    random.seed(FLAGS.seed)
    out_file = Path(FLAGS.output_file)
    assert out_file.suffix == ".jsonl"

    if out_file.exists():
        logging.info("Output file already exists, skipping")
        return

    setup_logging(out_file.parent, "build_dataset", flags=FLAGS)

    G = data_model.load_graph(FLAGS.graph_file)

    logging.info("Saving dataset samples to %s", out_file)
    with open(out_file, "w") as f:
        for chat in make_training_samples(G):
            f.write(json.dumps(chat) + "\n")


if __name__ == "__main__":
    app.run(main)
