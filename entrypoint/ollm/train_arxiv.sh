#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# Assume 2 GPUs. Adjust batch size if using less/more GPUs.
accelerate launch --multi_gpu \
    ollm/experiments/ollm/training/main_weighted.py \
    --config ollm/experiments/ollm/training/config.py \
    --config.wandb.notes "Arxiv masked adaptation" \
    --config.model.name andylolu24/ollm-wikipedia \
    --config.train.epochs 3 \
    --config.train.batch_size 8 \
    --config.train.lora.rank 8 \
    --config.train.lora.alpha 8 \
    --config.train.learning_rate 3e-6 \
    --config.train.warmup_steps 10 \
    --config.train.logging_steps 32 \
    --config.train.group_by_length=False \
    --config.data.train_size 2048 \
    --config.data.eval_size 256 \
    --config.eval.eval_steps 32 \
    --config.data.train_file out/linearised_datasets/arxiv/train_dataset.jsonl \
    --config.data.eval_file out/linearised_datasets/arxiv/eval_dataset.jsonl \
    --config.output_dir out/experiments/ollm/arxiv/train