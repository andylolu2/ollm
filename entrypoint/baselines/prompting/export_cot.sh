#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/cot/wikipedia/v2/test

python llm_ol/experiments/llm/prompting/export_graph.py \
    --hierarchy_file ${exp_dir}_0/categorised_pages.jsonl \
    --hierarchy_file ${exp_dir}_1/categorised_pages.jsonl \
    --hierarchy_file ${exp_dir}_2/categorised_pages.jsonl \
    --hierarchy_file ${exp_dir}_3/categorised_pages.jsonl \
    --output_dir $exp_dir
