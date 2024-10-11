from pathlib import Path

from absl import app, flags, logging

from ollm.dataset import data_model
from ollm.utils import batch, setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_multi_string("graph_file", None, "Path to the graph file", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("group_size", 1000, "Number of abstracts per file")
flags.DEFINE_integer("max_depth", None, "Maximum depth of the graph")


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "make_txt", flags=FLAGS)

    seen = set()
    abstracts = []
    for graph_file in FLAGS.graph_file:
        G = data_model.load_graph(graph_file, FLAGS.max_depth)
        for _, data in G.nodes(data=True):
            for page in data["pages"]:
                if page["id"] in seen:
                    continue
                seen.add(page["id"])
                abstracts.append(page["abstract"])

    for i, abstract_batch in enumerate(batch(abstracts, FLAGS.group_size)):
        with open(out_dir / f"{i}.txt", "w") as f:
            f.write("\n".join(abstract_batch))

    logging.info("Wrote %d abstracts to %s", len(abstracts), out_dir)


if __name__ == "__main__":
    app.run(main)
