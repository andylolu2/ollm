#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi


dataset=arxiv  # or wikipedia

if [ $dataset == "arxiv" ]; then
    k=150
elif [ $dataset == "wikipedia" ]; then
    k=5
else
    echo "Unknown dataset $dataset"
    exit 1
fi

dir=out/experiments/hearst/$dataset/test

python llm_ol/experiments/hearst/export_graph_with_ground_truth.py \
    --extraction_dir $dir/extractions \
    --graph_true out/data/$dataset/final/test_graph.json \
    --k $k \
    --output_dir $dir