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
    --config.wandb.notes "Masked loss" \
    --config.model.name mistralai/Mistral-7B-Instruct-v0.2 \
    --config.train.epochs 2 \
    --config.train.batch_size 8 \
    --config.data.train_file out/linearised_datasets/wikipedia/train_dataset.jsonl \
    --config.data.eval_file out/linearised_datasets/wikipedia/eval_dataset.jsonl \
    --config.output_dir out/experiments/ollm/wikipedia/train
