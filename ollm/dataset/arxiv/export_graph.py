import json
from pathlib import Path

from absl import app, flags, logging

from ollm.dataset import arxiv, data_model
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "categories_file", None, "File containing categories", required=True, short_name="c"
)
flags.DEFINE_string(
    "pages_file", None, "File containing pages", required=True, short_name="p"
)
flags.DEFINE_integer("min_citations", 0, "Minimum number of citations for a paper")
flags.DEFINE_string(
    "output_dir", None, "Directory to save output files", required=True, short_name="o"
)


def main(_):
    # Set up
    out_dir = Path(FLAGS.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    with open(FLAGS.pages_file, "r") as f:
        papers = [json.loads(line) for line in f]

    G = data_model.load_graph(FLAGS.categories_file)

    for node in G.nodes:
        G.nodes[node]["pages"] = []

    for paper in papers:
        if paper["citation_count"] < FLAGS.min_citations:
            continue
        for category in paper["categories"]:
            category = arxiv.normalise(category)
            if category not in G:
                logging.warning("Unknown leaf category: %s", category)
                continue
            G.nodes[category]["pages"].append(
                {
                    "id": paper["id"],
                    "title": paper["title"],
                    "abstract": paper["abstract"],
                }
            )

    data_model.save_graph(G, out_dir / "full_graph.json")


if __name__ == "__main__":
    app.run(main)
