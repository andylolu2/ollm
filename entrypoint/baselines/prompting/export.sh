#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

exp_dir=out/experiments/prompting/arxiv/v2/test

python llm_ol/experiments/llm/prompting/export_graph.py \
    --hierarchy_file $exp_dir/categorised_pages.jsonl \
    --output_dir $exp_dir
