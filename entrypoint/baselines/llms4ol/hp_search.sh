#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]); then
echo \
"Usage: $0 <dataset>

dataset: wikipedia | arxiv
"
exit 1
fi

dataset=$1
exp_dir=out/experiments/baselines/llms4ol/$dataset

python ollm/eval/hp_search.py \
    --graph $exp_dir/eval/graph.json \
    --graph_true out/data/$dataset/final/eval_graph.json \
    --num_samples 21 \
    --output_dir $exp_dir