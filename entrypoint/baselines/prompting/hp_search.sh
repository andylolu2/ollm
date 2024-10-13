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
split: eval | test
"
exit 1
fi

dataset=$1
k_shot=$2

exp_dir=out/experiments/baselines/prompting_${k_shot}shot/$dataset

python ollm/experiments/baselines/prompting/export_graph.py \
    --hierarchy_file $exp_dir/eval/categorised_pages.jsonl \
    --output_dir $exp_dir/eval

python ollm/eval/hp_search.py \
    --graph $exp_dir/eval/graph.json \
    --graph_true out/data/$dataset/final/eval_graph.json \
    --num_samples 21 \
    --output_dir $exp_dir