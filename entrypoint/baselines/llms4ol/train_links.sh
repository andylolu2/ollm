#!/bin/bash
set -e

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

python ollm/experiments/baselines/llms4ol/link_prediction/train.py \
    --model google-bert/bert-base-cased \
    --train_graph out/data/wikipedia/final/train_graph.json \
    --eval_graph out/data/wikipedia/final/eval_graph.json \
    --output_dir out/experiments/baselines/llms4ol/wikipedia/train/link_prediction
