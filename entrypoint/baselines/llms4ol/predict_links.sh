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

exp_dir=out/experiments/baselines/llms4ol/$dataset

root="Main topic classifications"

python ollm/experiments/baselines/llms4ol/link_prediction/predict_links.py \
    --output_dir $exp_dir/$split \
    --concepts_file $exp_dir/$split/concepts.json \
    --model $exp_dir/train/link_prediction/checkpoint-final \
    --factor 10
