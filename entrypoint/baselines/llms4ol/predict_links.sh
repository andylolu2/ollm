#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || \
    ([ $1 != "eval" ] && [ $1 != "test" ]); then
echo \
"Usage: $0 <dataset> <split>

split: eval | test
"
exit 1
fi

split=$1

exp_dir=out/experiments/baselines/llms4ol/wikipedia

python ollm/experiments/baselines/llms4ol/link_prediction/predict_links.py \
    --output_dir $exp_dir/$split \
    --concepts_file $exp_dir/$split/concepts.json \
    --model $exp_dir/train/link_prediction/checkpoint-final \
    --factor 10
