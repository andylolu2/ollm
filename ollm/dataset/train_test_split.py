import random
from pathlib import Path

import networkx as nx
from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("split_depth", 1, "Depth at which to split the graph")
flags.DEFINE_float(
    "split_prop", 0.1, "Proportion of nodes at the split depth in the test set"
)
flags.DEFINE_integer("seed", 0, "Random seed")


def split_graph(
    G: nx.Graph, split_depth: int, prop: float
) -> tuple[nx.Graph, nx.Graph]:
    assert 0 <= prop <= 1
    dist_to_root = nx.single_source_shortest_path_length(G, G.graph["root"])
    shared_nodes = {n for n, d in dist_to_root.items() if d < split_depth}
    longest_dist = max(dist_to_root.values())

    nodes_to_split = [n for n, d in dist_to_root.items() if d == split_depth]
    random.shuffle(nodes_to_split)
    nodes_1 = set(nodes_to_split[: int(prop * len(nodes_to_split))])
    nodes_2 = set(nodes_to_split[int(prop * len(nodes_to_split)) :])
    logging.info(
        "Splitting graph at depth %d, selecting %d/%d nodes",
        split_depth,
        len(nodes_1),
        len(nodes_to_split),
    )

    reachable_1 = set(
        nx.multi_source_dijkstra_path_length(
            G, nodes_1, cutoff=longest_dist - split_depth
        ).keys()
    )
    reachable_2 = set(
        nx.multi_source_dijkstra_path_length(
            G, nodes_2, cutoff=longest_dist - split_depth
        ).keys()
    )
    G_1 = G.subgraph(reachable_1 | shared_nodes).copy()
    G_2 = G.subgraph(reachable_2 | shared_nodes).copy()
    return G_1, G_2


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "train_test_split", flags=FLAGS)

    G = data_model.load_graph(FLAGS.graph_file)
    assert isinstance(G, nx.DiGraph)
    G_test, G_train = split_graph(G, FLAGS.split_depth, FLAGS.split_prop)

    dist_to_root = nx.single_source_shortest_path_length(G, G.graph["root"])
    for i in range(max(dist_to_root.values()) + 1):
        nodes = {n for n, d in dist_to_root.items() if d == i}
        nodes_train = nodes & G_train.nodes
        nodes_test = nodes & G_test.nodes
        logging.info(
            "Depth %d: %.2f%% nodes in train, %.2f%% nodes in test. %.2f%% shared nodes. %.2f%% covered nodes. Total %d nodes.",
            i,
            len(nodes_train) / len(nodes) * 100,
            len(nodes_test) / len(nodes) * 100,
            len(nodes_train & nodes_test) / len(nodes) * 100,
            len(nodes_train | nodes_test) / len(nodes) * 100,
            len(nodes),
        )
        edges = {
            (u, v) for u, v in G.edges if min(dist_to_root[u], dist_to_root[v]) == i
        }
        edges_train = edges & G_train.edges
        edges_test = edges & G_test.edges
        try:
            logging.info(
                "Depth %d: %.2f%% edges in train, %.2f%% edges in test. %.2f%% shared edges. %.2f%% covered edges. Total %d edges.\n",
                i,
                len(edges_train) / len(edges) * 100,
                len(edges_test) / len(edges) * 100,
                len(edges_train & edges_test) / len(edges) * 100,
                len(edges_train | edges_test) / len(edges) * 100,
                len(edges),
            )
        except ZeroDivisionError:
            logging.info("Depth %d: No edges\n", i)

    logging.info(
        "Overall: %.2f%% nodes in train, %.2f%% nodes in test. %.2f%% shared nodes. %.2f%% covered nodes. Total %d nodes.",
        len(G_train.nodes) / G.number_of_nodes() * 100,
        len(G_test.nodes) / G.number_of_nodes() * 100,
        len(G_train.nodes & G_test.nodes) / G.number_of_nodes() * 100,
        len(G_train.nodes | G_test.nodes) / G.number_of_nodes() * 100,
        G.number_of_nodes(),
    )
    logging.info(
        "Overall: %.2f%% edges in train, %.2f%% edges in test. %.2f%% shared edges. %.2f%% covered edges. Total %d edges.",
        len(G_train.edges) / G.number_of_edges() * 100,
        len(G_test.edges) / G.number_of_edges() * 100,
        len(G_train.edges & G_test.edges) / G.number_of_edges() * 100,
        len(G_train.edges | G_test.edges) / G.number_of_edges() * 100,
        G.number_of_edges(),
    )

    logging.info("Saving train graph to %s", out_dir / "train_graph.json")
    data_model.save_graph(G_train, out_dir / "train_graph.json")
    logging.info("Saving test graph to %s", out_dir / "test_graph.json")
    data_model.save_graph(G_test, out_dir / "test_graph.json")


if __name__ == "__main__":
    app.run(main)
