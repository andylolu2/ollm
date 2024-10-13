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
output_dir=out/experiments/ollm/$dataset

python ollm/experiments/ollm/inference.py \
    --test_dataset out/linearised_datasets/$dataset/eval_dataset.jsonl \
    --model andylolu24/ollm-$dataset \
    --output_dir $output_dir/eval

python ollm/experiments/ollm/export_graph.py \
    --hierarchy_file $output_dir/eval/categorised_pages.jsonl \
    --output_dir $output_dir/eval

python ollm/eval/hp_search.py \
    --graph $output_dir/eval/graph.json \
    --graph_true out/data/$dataset/final/eval_graph.json \
    --num_samples 21 \
    --output_dir $output_dir