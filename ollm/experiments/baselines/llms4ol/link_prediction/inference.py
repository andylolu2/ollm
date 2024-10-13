from itertools import product
from pathlib import Path

import networkx as nx
import torch
from absl import app, flags, logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from ollm.dataset import data_model
from ollm.utils import batch, setup_logging, textqdm

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_string("graph_pred", None, "Path to the graph to predict", required=True)
flags.DEFINE_string("model_path", None, "Path to the model checkpoint", required=True)


@torch.no_grad()
def main(_):
    output_dir = Path(FLAGS.output_dir)
    setup_logging(output_dir, "inference", flags=FLAGS)

    model = AutoModelForSequenceClassification.from_pretrained(
        FLAGS.model_path, num_labels=2, device_map="cuda", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_path)

    G = data_model.load_graph(FLAGS.graph_pred)
    nodes = list(G.nodes())
    node_names = [G.nodes[node]["title"] for node in nodes]

    weights = []
    for uv_batch in batch(
        textqdm(product(node_names, node_names), total=len(node_names) ** 2), 2048
    ):
        us, vs = zip(*uv_batch)
        inputs = tokenizer(
            us, vs, return_tensors="pt", padding=True, truncation=True
        ).to(model.device)
        output = model(**inputs)
        probs = torch.softmax(output.logits, dim=1)
        weights.append(probs[:, 1])
    weights = torch.cat(weights).reshape(len(node_names), len(node_names))
    weights = weights.float().cpu().numpy()

    G_pred = nx.DiGraph()
    for node, name in zip(nodes, node_names):
        G_pred.add_node(node, title=name)
    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            G_pred.add_edge(src, dst, weight=float(weights[i, j]))
    G_pred.graph["root"] = G.graph["root"]

    logging.info(
        "Extracted %d nodes and %d edges",
        G_pred.number_of_nodes(),
        G_pred.number_of_edges(),
    )

    data_model.save_graph(G_pred, Path(FLAGS.output_dir) / "graph.json")


if __name__ == "__main__":
    app.run(main)
