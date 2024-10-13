"""This script is idempotent."""

import json
from pathlib import Path

import torch
from absl import app, flags, logging
from vllm import LLM, SamplingParams

from ollm.experiments.templates import PROMPT_TEMPLATE
from ollm.utils import batch, setup_logging, textpbar

FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", None, "Path to the test dataset", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_string("model", None, "Model to use for inference.", required=True)


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "inference", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

    logging.info(
        "Example prompt:\n%s",
        PROMPT_TEMPLATE.render(title="TITLE", abstract="ABSTRACT"),
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
        model=FLAGS.model,
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

    for pages in batch(test_pages, 5000):
        prompts = []
        for page in pages:
            messages = [
                {
                    "role": "user",
                    "content": PROMPT_TEMPLATE.render(
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
                max_tokens=1024,
                seed=0,
                stop=["\n\n"],
            ),
        )
        for page, out in zip(pages, outputs):
            with open(out_file, "a") as f:
                response = out.outputs[0].text
                item = {**page, "hierarchy": response}
                f.write(json.dumps(item) + "\n")
            pbar.update()


if __name__ == "__main__":
    app.run(main)
