#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || \
    ([ $1 != "eval" ] && [ $1 != "test" ]); then
echo \
"Usage: $0 <split>

split: eval | test
"
exit 1
fi

split=$1

python ollm/experiments/baselines/llms4ol/concept_discovery/predict_concepts.py \
    --test_dataset out/linearised_datasets/wikipedia/${split}_dataset.jsonl \
    --output_dir out/experiments/baselines/llms4ol/wikipedia/$split/concept_discovery
