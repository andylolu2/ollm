import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from absl import app, flags, logging
from accelerate import PartialState
from datasets import Dataset, load_dataset
from ml_collections import config_flags
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

import wandb
from ollm.experiments.ollm.training.utils import GenerateSamplesCallback
from ollm.experiments.templates import (
    _MISTRAL_TEMPLATE,
    PROMPT_TEMPLATE,
    RESPONSE_TEMPLATE,
)
from ollm.utils import setup_logging

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config")


class Trainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss."""
        if not model.training:
            outputs = model(input_ids=inputs["input_ids"], labels=inputs["labels"])
        else:
            outputs = model(
                input_ids=inputs["input_ids"], labels=inputs["labels_detailed"]
            )
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def _prepare_dataset(
        self,
        dataset: Dataset,
        tokenizer,
        packing,
        dataset_text_field,
        max_seq_length,
        formatting_func,
        num_of_sequences,
        chars_per_token,
        remove_unused_columns=True,
        append_concat_token=True,
        add_special_tokens=True,
    ):
        edge_counts = defaultdict(int)
        for example in dataset:
            for path in example["paths"]:  # type: ignore
                for u, v in zip(path[:-1], path[1:]):
                    edge_counts[(u, v)] += 1
        # rescale such that each edge occurs on average `mean_edge_count` times
        mean_edge_count = np.mean(list(edge_counts.values()))
        edge_weights = {k: mean_edge_count / v for k, v in edge_counts.items()}

        def tokenize_one(example: dict[str, Any]):
            prompt = PROMPT_TEMPLATE.render(
                title=example["title"], abstract=example["abstract"]
            )
            response = RESPONSE_TEMPLATE.render(paths=example["paths"])
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=True)

            inst_end = [733, 28748, 16289, 28793]  # _[/INST]
            arrow = 3193  # _->
            linebreak = 13  # \n

            def find_index(list_, sublist):
                for i in range(len(list_) - len(sublist) + 1):
                    if list_[i : i + len(sublist)] == sublist:
                        return i
                raise ValueError(f"Sublist {sublist} not found in list")

            resp_start_idx = find_index(input_ids, inst_end) + len(inst_end)
            ignores = [True] * resp_start_idx + [False] * (
                len(input_ids) - resp_start_idx
            )
            ignores_detailed = [True] * resp_start_idx

            prev_word = None
            word_ids = []
            for token_id in input_ids[resp_start_idx:]:
                if token_id == arrow or token_id == linebreak:
                    word = tokenizer.decode(word_ids)
                    assert "->" not in word
                    if prev_word is not None:
                        ignore = random.random() > edge_weights[(prev_word, word)]
                        ignores_detailed += [ignore] * len(word_ids)
                    else:  # First word in the path
                        ignores_detailed += [False] * len(word_ids)
                    ignores_detailed += [False]  # "->" or "\n"

                    word_ids = []
                    prev_word = word if token_id == arrow else None
                elif token_id == tokenizer.eos_token_id:
                    assert len(word_ids) == 0
                    ignores_detailed += [False]
                else:  # token is part of a word
                    word_ids.append(token_id)

            assert len(input_ids) == len(ignores_detailed)
            return {
                "input_ids": input_ids,
                "ignores": ignores,
                "ignores_detailed": ignores_detailed,
            }

        def tokenize(examples: dict[str, list[Any]]):
            # dict of lists -> list of dicts
            examples_list = [dict(zip(examples, t)) for t in zip(*examples.values())]

            results = []
            for example in examples_list:
                try:
                    results.append(tokenize_one(example))
                except Exception as e:
                    logging.warning(f"Error reweighting example: {example} {repr(e)}")

            # list of dicts -> dict of lists
            return {k: [d[k] for d in results] for k in results[0]}

        logging.info("Dataset size: %d", len(dataset))
        dataset = dataset.map(
            tokenize,
            num_proc=self.dataset_num_proc,
            batched=True,
            remove_columns=dataset.column_names,
        )
        dataset = dataset.filter(
            lambda ex: len(ex["input_ids"]) <= max_seq_length,
            num_proc=self.dataset_num_proc,
        )
        logging.info("Dataset size after tokenization and filtering: %d", len(dataset))
        return dataset


class DataCollator(DataCollatorForCompletionOnlyLM):
    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        max_length = max(len(ex["input_ids"]) for ex in examples)
        if self.pad_to_multiple_of is not None:
            max_length = (
                (max_length // self.pad_to_multiple_of) + 1
            ) * self.pad_to_multiple_of
        max_length = min(max_length, self.tokenizer.model_max_length)

        input_ids = []
        ignores_detailed = []
        ignores = []
        for ex in examples:
            diff = max_length - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.tokenizer.pad_token_id] * diff)
            ignores.append(ex["ignores"] + [True] * diff)
            ignores_detailed.append(ex["ignores_detailed"] + [True] * diff)
        input_ids = torch.tensor(input_ids)
        ignores = torch.tensor(ignores)
        ignores_detailed = torch.tensor(ignores_detailed)
        return {
            "input_ids": input_ids,
            "labels": torch.where(ignores, -100, input_ids),
            "labels_detailed": torch.where(ignores_detailed, -100, input_ids),
        }


def dataset_from_file(
    data_file: str | Path, size: int | None = None, seed: int = 0
) -> Dataset:
    dataset = load_dataset("json", data_files=str(data_file), split="train")
    assert isinstance(dataset, Dataset)
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(size))
    return dataset


def main(_):
    config = FLAGS.config
    logging.info("Config:\n%s", config)
    random.seed(config.seed)
    setup_logging(config.output_dir, "main")

    print("LOCAL RANK: ", os.environ.get("LOCAL_RANK", None))

    device_string = PartialState().process_index
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        use_cache=False,
        device_map={"": device_string} if torch.cuda.is_available() else "auto",
        torch_dtype="auto",
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config.train.lora.rank,
            lora_alpha=config.train.lora.alpha,
            lora_dropout=config.train.lora.dropout,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    tokenizer.chat_template = _MISTRAL_TEMPLATE
    tokenizer.padding_side = "right"
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.unk_token

    collator = DataCollator(
        response_template=config.model.response_template,
        instruction_template=config.model.instruction_template,
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=collator,
        train_dataset=dataset_from_file(
            config.data.train_file, config.data.train_size, config.seed
        ),
        eval_dataset={
            "in_domain": dataset_from_file(
                config.data.train_file, config.data.eval_size, config.seed + 1
            ),
            "out_of_domain": dataset_from_file(
                config.data.eval_file, config.data.eval_size, config.seed
            ),
        },
        formatting_func=lambda: None,
        callbacks=[
            GenerateSamplesCallback(
                config.eval.num_generate_samples, config.model.response_template
            )
        ],
        args=SFTConfig(
            output_dir=config.output_dir,
            overwrite_output_dir=False,
            optim="adamw_torch_fused",
            learning_rate=config.train.learning_rate,
            lr_scheduler_type="constant_with_warmup",
            warmup_steps=config.train.warmup_steps,
            report_to=["wandb", "tensorboard"],
            max_seq_length=config.train.max_seq_length,
            num_train_epochs=config.train.epochs,
            logging_steps=config.train.logging_steps,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            gradient_accumulation_steps=config.train.grad_acc_steps,
            ddp_find_unused_parameters=False,
            group_by_length=config.train.group_by_length,
            dataset_num_proc=16,
            dataset_kwargs={
                "add_special_tokens": False,
            },
            remove_unused_columns=False,
            fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            evaluation_strategy="steps",
            eval_steps=config.eval.eval_steps,
            save_steps=config.eval.eval_steps,
            per_device_train_batch_size=config.train.batch_size,
            per_device_eval_batch_size=config.eval.batch_size,
            seed=config.seed,
            data_seed=config.seed,
        ),
    )

    if trainer.state.is_world_process_zero:
        wandb.init(
            project=config.wandb.project,
            notes=config.wandb.notes,
            config=config.to_dict(),
            save_code=True,
        )

    resume_from_checkpoint = get_last_checkpoint(config.output_dir) is not None
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # type: ignore

    # Save the final model
    trainer.save_model(str(Path(config.output_dir) / "checkpoint-final"))


if __name__ == "__main__":
    app.run(main)
