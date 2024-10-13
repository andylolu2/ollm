"""This script is idempotent."""

import json
import random
from pathlib import Path

import torch
from absl import app, flags, logging
from vllm import LLM, SamplingParams

from ollm.experiments.templates import (
    COT_ROUND_ONE_TEMPLATE,
    COT_ROUND_TWO_TEMPLATE,
)
from ollm.utils import batch, setup_logging, textpbar

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "train_dataset", None, "Path to the training dataset", required=True
)
flags.DEFINE_string("test_dataset", None, "Path to the test dataset", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_integer("k_shot", 1, "Number of samples to provide.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def main(_):
    random.seed(FLAGS.seed)
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

    with open(FLAGS.train_dataset, "r") as f:
        examples = [json.loads(line) for line in f.readlines()]
    logging.info(
        "Example prompt (round 1):\n%s",
        COT_ROUND_ONE_TEMPLATE.render(
            title="TITLE",
            abstract="ABSTRACT",
            examples=random.sample(examples, FLAGS.k_shot),
        ),
    )
    logging.info("Example prompt (round 2):\n%s", COT_ROUND_TWO_TEMPLATE.render())

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
        tensor_parallel_size=torch.cuda.device_count(),
        max_num_batched_tokens=4096,
        max_model_len=8192,
        max_seq_len_to_capture=4096,
        max_num_seqs=512,
        block_size=32,
        enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()
    pbar = textpbar(len(test_pages))

    for pages in batch(test_pages, 2000):
        few_shot_examples = [random.sample(examples, FLAGS.k_shot) for _ in pages]
        messages = []
        for page, few_shot in zip(pages, few_shot_examples):
            messages.append(
                [
                    {
                        "role": "user",
                        "content": COT_ROUND_ONE_TEMPLATE.render(
                            title=page["title"],
                            abstract=page["abstract"],
                            examples=few_shot,
                        ),
                    }
                ]
            )

        # Round one
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        r1_outputs = llm.generate(
            prompts,  # type: ignore
            sampling_params=SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=512,
                seed=FLAGS.seed,
            ),
        )
        for i, output in enumerate(r1_outputs):
            messages[i] += [
                {
                    "role": "assistant",
                    "content": output.outputs[0].text,
                },
                {
                    "role": "user",
                    "content": COT_ROUND_TWO_TEMPLATE.render(),
                },
            ]

        # Round two
        prompts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]
        r2_outputs = llm.generate(
            prompts,  # type: ignore
            sampling_params=SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=1024,
                stop=["\n\n"],
                seed=FLAGS.seed,
            ),
        )

        for page, few_shot, r1_out, r2_out in zip(
            pages, few_shot_examples, r1_outputs, r2_outputs
        ):
            with open(out_file, "a") as f:
                item = {
                    **page,
                    "hierarchy": r2_out.outputs[0].text,
                    "few_shot_ids": [ex["id"] for ex in few_shot],
                    "description": r1_out.outputs[0].text,
                }
                f.write(json.dumps(item) + "\n")
            pbar.update()


if __name__ == "__main__":
    app.run(main)
