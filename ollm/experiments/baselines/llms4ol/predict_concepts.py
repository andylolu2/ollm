"""This script is idempotent."""

import json
import random
from pathlib import Path

from absl import app, flags, logging
from vllm import LLM, SamplingParams

from ollm.experiments.llm.templates import PRED_CONCEPT_TEMPLATE
from ollm.utils import batch, setup_logging, textpbar

FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", None, "Path to the test dataset", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("seed", 0, "Random seed.")


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

    logging.info(
        "Example prompt:\n%s",
        PRED_CONCEPT_TEMPLATE.render(title="TITLE", abstract="ABSTRACT"),
    )

    computed = set()
    if out_file.exists():
        with open(out_file, "r") as f:
            computed.update({json.loads(line)["id"] for line in f})
    logging.info("Loaded %d computed pages", len(computed))

    with open(FLAGS.test_dataset, "r") as f:
        test_pages = [json.loads(line) for line in f.readlines()]
        test_pages = [
            {
                "id": sample["id"],
                "title": sample["title"],
                "abstract": sample["abstract"],
            }
            for sample in test_pages
            if sample["id"] not in computed
        ]
    logging.info("Computing responses for %d pages", len(test_pages))

    llm = LLM(
        model="mistralai/Mistral-7B-Instruct-v0.2",
        download_dir="out/models",
        max_num_seqs=512,
        max_paddings=512,
        max_model_len=4096,
        seed=FLAGS.seed,
    )
    tokenizer = llm.get_tokenizer()
    pbar = textpbar(len(test_pages))

    for pages in batch(test_pages, 5000):
        prompts = []
        for page in pages:
            messages = [
                {
                    "role": "user",
                    "content": PRED_CONCEPT_TEMPLATE.render(
                        title=page["title"], abstract=page["abstract"]
                    ),
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)
        outputs = llm.generate(
            prompts,
            sampling_params=SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=256,
                stop=["\n\n"],
            ),
        )
        for page, out in zip(pages, outputs):
            with open(out_file, "a") as f:
                response = out.outputs[0].text
                item = {**page, "concepts": response}
                f.write(json.dumps(item) + "\n")
            pbar.update()


if __name__ == "__main__":
    app.run(main)
