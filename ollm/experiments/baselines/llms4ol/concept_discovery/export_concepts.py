import json
from collections import Counter
from pathlib import Path

from absl import app, flags, logging

from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_multi_string(
    "raw_prediction_file", None, "Path to the hierarchy directory", required=True
)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("top_k", None, "Top k concepts to consider", required=True)


def parse_concepts(concept_str: str):
    return [concept.strip(" \n\t\"',.") for concept in concept_str.split(",")]


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "export_graph", flags=FLAGS)

    results = []
    for pred_file in FLAGS.raw_prediction_file:
        with open(pred_file, "r") as f:
            results += [json.loads(line) for line in f.readlines()]

    concepts = Counter()
    for item in results:
        for concept in parse_concepts(item["concepts"]):
            concepts[concept] += 1
    logging.info(
        "Found %d unique concepts, exporting %d of them", len(concepts), FLAGS.top_k
    )

    top_concepts = dict(concepts.most_common(FLAGS.top_k))
    with open(out_dir / "concepts.json", "w") as f:
        json.dump(top_concepts, f, indent=2)


if __name__ == "__main__":
    app.run(main)
