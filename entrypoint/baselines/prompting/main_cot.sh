#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# dataset=arxiv
dataset=v2
split=test

N_DEVICES=4

# split dataset into N_DEVICES parts
split -n l/$N_DEVICES -d --additional-suffix=.jsonl \
    out/experiments/llm/$dataset/${split}_dataset.jsonl \
    out/experiments/llm/$dataset/${split}_dataset_

for i in $(seq 0 $((N_DEVICES-1))); do
    CUDA_VISIBLE_DEVICES=$i python llm_ol/experiments/llm/prompting/main_cot.py \
        --train_dataset out/experiments/llm/$dataset/train_dataset.jsonl \
        --test_dataset out/experiments/llm/$dataset/${split}_dataset_0$i.jsonl \
        --k_shot 0 \
        --output_dir out/experiments/cot/wikipedia/v2/${split}_$i &
done

