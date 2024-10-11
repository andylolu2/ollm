import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import wandb
from absl import app, flags
from datasets import Dataset, load_metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from ollm.dataset import data_model
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
flags.DEFINE_string("output_dir", None, "Path to the output directory", required=True)
flags.DEFINE_string(
    "train_graph", None, "Path to the training graph file", required=True
)
flags.DEFINE_string(
    "eval_graph", None, "Path to the evaluation graph file", required=True
)
flags.DEFINE_string("model_id", "bert-base-uncased", "Model ID to use for training")
flags.DEFINE_integer("logging_steps", 100, "Logging steps for the trainer")
flags.DEFINE_integer("save_eval_steps", 500, "Evaluation steps for the trainer")


def main(_):
    output_dir = Path(FLAGS.output_dir)
    setup_logging(output_dir, "train", flags=FLAGS)

    accuracy = load_metric("accuracy", trust_remote_code=True)
    precision = load_metric("precision", trust_remote_code=True)
    recall = load_metric("recall", trust_remote_code=True)
    f1 = load_metric("f1", trust_remote_code=True)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        acc = accuracy.compute(predictions=predictions, references=labels)["accuracy"]
        prec = precision.compute(predictions=predictions, references=labels)[
            "precision"
        ]
        rec = recall.compute(predictions=predictions, references=labels)["recall"]
        f1_score = f1.compute(predictions=predictions, references=labels)["f1"]
        return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1_score}

    def build_dataset(G: nx.DiGraph, tokenizer) -> Dataset:
        def title(node):
            return G.nodes[node]["title"]

        nodes = list(G.nodes())
        edges = list(G.edges())

        pos_samples = [(title(src), title(dst)) for src, dst in edges]
        neg_samples = set()
        for u in nodes:
            for v in nodes:
                if not G.has_edge(u, v):
                    neg_samples.add((title(u), title(v)))
        neg_samples = list(neg_samples)
        random.shuffle(neg_samples)
        neg_samples = neg_samples[: 10 * len(pos_samples)]

        labels = [1] * len(pos_samples) + [0] * len(neg_samples)
        parents, children = zip(*pos_samples + neg_samples)
        ds = Dataset.from_dict(
            {"parents": parents, "children": children, "labels": labels}
        )

        def preprocess(examples):
            return tokenizer(
                examples["parents"],
                examples["children"],
                padding=True,
                truncation=True,
            )

        return ds.map(preprocess)

    model = AutoModelForSequenceClassification.from_pretrained(
        FLAGS.model_id, num_labels=2, device_map="cuda", torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_id)

    G_train = data_model.load_graph(FLAGS.train_graph)
    G_eval = data_model.load_graph(FLAGS.eval_graph)
    ds_train = build_dataset(G_train, tokenizer)
    ds_eval = build_dataset(G_eval, tokenizer).shuffle(seed=0).select(range(4096))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        optim="adamw_torch_fused",
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        logging_strategy="steps",
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_steps=FLAGS.logging_steps,
        eval_steps=FLAGS.save_eval_steps,
        save_steps=FLAGS.save_eval_steps,
        num_train_epochs=100,
        report_to=["wandb"],
        load_best_model_at_end=True,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        seed=0,
        data_seed=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    wandb.init(project="link_prediction", save_code=True)

    trainer.evaluate()
    trainer.train()
    trainer.save_model(str(Path(FLAGS.output_dir) / "checkpoint-final"))


if __name__ == "__main__":
    app.run(main)
