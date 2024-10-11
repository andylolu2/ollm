#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=arxiv
split=test

python ollm/experiments/baselines/prompting/main.py \
    --train_dataset out/linearised_datasets/$dataset/train_dataset.jsonl \
    --test_dataset out/linearised_datasets/$dataset/${split}_dataset.jsonl \
    --k_shot 3 \
    --output_dir out/experiments/baselines/prompting/$dataset/${split}
