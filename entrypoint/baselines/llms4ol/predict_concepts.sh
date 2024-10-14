#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || [ -z "$2" ] || \
    ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]) || \
    ([ $2 != "eval" ] && [ $2 != "test" ]); then
echo \
"Usage: $0 <dataset> <split>

dataset: wikipedia | arxiv
split: eval | test
"
exit 1
fi

dataset=$1
split=$2

python ollm/experiments/baselines/llms4ol/concept_discovery/predict_concepts.py \
    --test_dataset out/linearised_datasets/$dataset/${split}_dataset.jsonl \
    --output_dir out/experiments/baselines/llms4ol/$dataset/$split/concept_discovery
