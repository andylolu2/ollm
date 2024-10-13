#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || [ -z "$2" ] || \
    ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]); then
echo \
"Usage: $0 <dataset> <k_shot>

dataset: wikipedia | arxiv
k_shot: number of shots for prompting
"
exit 1
fi

dataset=$1
k_shot=$2

exp_dir=out/experiments/baselines/prompting_${k_shot}shot/$dataset

python ollm/experiments/baselines/prompting/export_graph.py \
    --hierarchy_file $exp_dir/test/categorised_pages.jsonl \
    --output_dir $exp_dir/test

python ollm/eval/test_metrics.py \
    --graph $exp_dir/test/graph.json \
    --graph_true out/data/$dataset/final/test_graph.json \
    --hp_search_result $exp_dir/hp_search.jsonl \
    --output_file $exp_dir/test_metrics.json
