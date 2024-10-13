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
    --test_dataset out/linearised_datasets/$dataset/test_dataset.jsonl \
    --model andylolu24/ollm-$dataset \
    --output_dir $output_dir/test

python ollm/experiments/ollm/export_graph.py \
    --hierarchy_file $output_dir/test/categorised_pages.jsonl \
    --output_dir $output_dir/test

python ollm/eval/test_metrics.py \
    --graph $output_dir/test/graph.json \
    --graph_true out/data/$dataset/final/test_graph.json \
    --hp_search_result $output_dir/hp_search.jsonl \
    --output_file $output_dir/test_metrics.json
