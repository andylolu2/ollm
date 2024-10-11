#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/finetune/v4
# step=final
# split=test
# dataset=v2
# model=$exp_dir/train/checkpoint-$step

step=13000
split=test
dataset=v2
model=$exp_dir/train/checkpoint-$step

# exp_dir=out/experiments/finetune/arxiv/v5
# step=final
# split=eval
# dataset=arxiv
# model=out/experiments/finetune/v4/train/checkpoint-$step

echo "Running inference on $model"

if [ ! -d "$model/merged" ]; then
    python llm_ol/experiments/llm/finetune/export_model.py \
        --checkpoint_dir $model
fi

python llm_ol/experiments/llm/finetune/inference.py \
    --test_dataset out/experiments/llm/$dataset/${split}_dataset.jsonl \
    --model $model/merged \
    --size 10000 \
    --output_dir $exp_dir/$step/$split
