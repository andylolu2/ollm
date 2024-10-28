#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/baselines/llms4ol/wikipedia

python ollm/eval/hp_search.py \
    --graph $exp_dir/eval/graph.json \
    --graph_true out/data/wikipedia/final/eval_graph.json \
    --num_samples 21 \
    --output_dir $exp_dir