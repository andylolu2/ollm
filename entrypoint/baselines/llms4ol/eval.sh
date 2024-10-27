#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

split=eval
dataset=wikipedia/v2
exp_dir=out/experiments/llms4ol/wikipedia/v1/$split

python llm_ol/eval/hp_search.py \
    --graph $exp_dir/graph_pruned.json \
    --graph_true out/data/$dataset/train_${split}_split/test_graph.json \
    --num_samples 21 \
    --output_dir $exp_dir \