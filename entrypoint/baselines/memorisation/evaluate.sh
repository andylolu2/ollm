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

exp_dir=out/experiments/baselines/memorisation/$dataset

python ollm/eval/hp_search.py \
    --graph out/data/$dataset/final/train_graph.json \
    --graph_true out/data/$dataset/final/eval_graph.json \
    --num_samples 21 \
    --output_dir $exp_dir

python ollm/eval/test_metrics.py \
    --graph out/data/$dataset/final/train_graph.json \
    --graph_true out/data/$dataset/final/test_graph.json \
    --hp_search_result $exp_dir/hp_search.jsonl \
    --output_file $exp_dir/test_metrics.json

