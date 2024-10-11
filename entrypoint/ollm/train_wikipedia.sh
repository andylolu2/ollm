#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

data_dir=out/linearised_datasets/wikipedia

accelerate launch --multi_gpu --mixed_precision=bf16 \
    ollm/experiments/ollm/training/main_weighted.py \
    --config ollm/experiments/ollm/training/config.py \
    --config.wandb.notes "Masked loss" \
    --config.model.name mistralai/Mistral-7B-Instruct-v0.2 \
    --config.train.epochs 2 \
    --config.train.batch_size 8 \
    --config.data.train_file $data_dir/train_dataset.jsonl \
    --config.data.eval_file $data_dir/eval_dataset.jsonl \
    --config.output_dir out/experiments/ollm/run
