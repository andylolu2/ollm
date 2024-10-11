#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

dataset=arxiv # or wikipedia

exp_dir=out/experiments/hearst/$dataset/eval

python llm_ol/experiments/hearst/export_graph_with_ground_truth.py \
    --extraction_dir $exp_dir/extractions \
    --graph_true out/data/$dataset/final/eval_graph.json \
    --k 5 \
    --k 10 \
    --k 15 \
    --k 20 \
    --k 25 \
    --k 50 \
    --k 100 \
    --k 150 \
    --k 200 \
    --k 250 \
    --output_dir $exp_dir

for k in 5 10 15 20 25 50 100 150 200 250; do
    python llm_ol/eval/hp_search.py \
        --graph $exp_dir/k_$k/graph.json \
        --graph_true out/data/$dataset/final/eval_graph.json \
        --num_samples 21 \
        --output_dir $exp_dir/k_$k
done