#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/baselines/llms4ol/wikipedia

python ollm/eval/test_metrics.py \
    --graph $exp_dir/test/graph.json \
    --graph_true out/data/wikipedia/final/test_graph.json \
    --hp_search_result $exp_dir/hp_search.jsonl \
    --output_file $exp_dir/test_metrics.json
