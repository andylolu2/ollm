#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

for split in train eval test; do
    python ollm/experiments/build_linearised_dataset.py \
        --graph_file out/data/wikipedia/final/${split}_graph.json \
        --cutoff 5 \
        --num_workers 16 \
        --output_file out/linearised_datasets/wikipedia/${split}_dataset.jsonl
done