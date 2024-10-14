#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

if [ -z "$1" ] || ([ $1 != "wikipedia" ] && [ $1 != "arxiv" ]); then
echo \
"Usage: $0 <dataset>

dataset: wikipedia | arxiv
"
exit 1
fi

dataset=$1

python ollm/experiments/baselines/llms4ol/link_prediction/train.py \
    --model google-bert/bert-base-cased \
    --train_graph out/data/$dataset/final/train_graph.json \
    --eval_graph out/data/$dataset/final/eval_graph.json \
    --output_dir out/experiments/baselines/llms4ol/$dataset/train/link_prediction
