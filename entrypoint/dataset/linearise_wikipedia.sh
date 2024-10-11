#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

cutoff=5
src_dir=out/data/wikipedia/final
output_dir=out/linearised_datasets/wikipedia

for split in train eval test; do
    python ollm/experiments/build_linearised_dataset.py \
        --graph_file $src_dir/${split}_graph.json \
        --cutoff $cutoff \
        --num_workers 16 \
        --output_file $output_dir/${split}_dataset.jsonl
done