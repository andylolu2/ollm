#!/bin/bash

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ] || \
    ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]) || \
    ([ $2 != "eval" ] && [ $2 != "test" ]); then
echo \
"Usage: $0 <dataset> <split> <k_shot>

dataset: wikipedia | arxiv
split: eval | test
k_shot: number of prompts to use
"
exit 1
fi

dataset=$1
split=$2
k_shot=$3

python ollm/experiments/baselines/prompting/main.py \
    --train_dataset out/linearised_datasets/$dataset/train_dataset.jsonl \
    --test_dataset out/linearised_datasets/$dataset/${split}_dataset.jsonl \
    --k_shot $k_shot \
    --output_dir out/experiments/baselines/prompting_${k_shot}shot/$dataset/$split
