"""This script is idempotent."""

import json
from pathlib import Path

import torch
from absl import app, flags, logging
from transformers import AutoTokenizer, BartForConditionalGeneration

from llm_ol.utils import batch, setup_logging, textpbar

FLAGS = flags.FLAGS
flags.DEFINE_string("test_dataset", None, "Path to the test dataset", required=True)
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)

model_id = "Babelscape/rebel-large"


# Function to parse the generated text and extract the triplets
def extract_triplets(text: str) -> list[dict[str, str]]:
    triplets = []
    relation, subject, relation, object_ = "", "", "", ""
    text = text.strip()
    current = "x"
    for token in (
        text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split()
    ):
        if token == "<triplet>":
            current = "t"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
                relation = ""
            subject = ""
        elif token == "<subj>":
            current = "s"
            if relation != "":
                triplets.append(
                    {
                        "head": subject.strip(),
                        "type": relation.strip(),
                        "tail": object_.strip(),
                    }
                )
            object_ = ""
        elif token == "<obj>":
            current = "o"
            relation = ""
        else:
            if current == "t":
                subject += " " + token
            elif current == "s":
                object_ += " " + token
            elif current == "o":
                relation += " " + token
    if subject != "" and relation != "" and object_ != "":
        triplets.append(
            {"head": subject.strip(), "type": relation.strip(), "tail": object_.strip()}
        )
    return triplets


def main(_):
    out_dir = Path(FLAGS.output_dir)
    setup_logging(out_dir, "main", flags=FLAGS)
    out_file = out_dir / "categorised_pages.jsonl"

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

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    model = BartForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    pbar = textpbar(len(test_pages))

    for pages in batch(test_pages, 128):
        abstracts = [page["abstract"] for page in pages]
        inputs = tokenizer(
            abstracts,
            max_length=model.config.max_length,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        generated_tokens = model.generate(
            **inputs,
            length_penalty=0.0,
            max_length=256,
            min_length=12,
            no_repeat_ngram_size=0,
            num_beams=4,
        )
        outputs = tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

        for page, output in zip(pages, outputs):
            triplets = extract_triplets(output)
            with open(out_file, "a") as f:
                item = {**page, "triplets": triplets}
                f.write(json.dumps(item) + "\n")
            pbar.update()


if __name__ == "__main__":
    app.run(main)
